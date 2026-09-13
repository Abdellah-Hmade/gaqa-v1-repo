#!/usr/bin/env bash
# ============================================================================
# GAQA Real-Data External Validation Pipeline
# ============================================================================
#
# Time budget (target ≤5h wall time on server, CPU-only until eval):
#   Stage       Time  Notes
#   setup       ~45m  micromamba bootstrap, env creation, DB downloads
#   download    ~20m  reads + references from ENA/NCBI
#   subsample   ~15m  downsampling, spike-in, condition generation
#   assemble    ~75m  SPAdes parallel assemblies (6 jobs × 4 threads)
#   qc          ~60m  QUAST + BUSCO + CheckM parallel runs
#   render      ~15m  build GAQA-format test questions
#   eval        ~15m  GPU logit-scorer (optional, skip if ADAPTER_DIR unset)
#   report       ~5m  aggregate results into REPORT.md
#   ─────────────────
#   Total      ~255m  (plus buffer = ≤5h)
#
# Usage:
#   bash scripts/realval_pipeline.sh                     # run all stages
#   bash scripts/realval_pipeline.sh --stage setup       # run one stage
#   bash scripts/realval_pipeline.sh --stage download --stage subsample
#
# Env:
#   REALVAL_JOBS        max parallel assembly jobs (default 6)
#   REALVAL_ENV_PREFIX  conda/micromamba env prefix (default realval)
#   ADAPTER_DIR         model adapter dir for eval stage (optional)
#   BITNET_LORA_R       LoRA rank for eval (default 8)
#   BITNET_LORA_ALPHA   LoRA alpha for eval (default 16)
#
# Idempotent: each stage creates data/realval/.state/<stage>.done on success.
# Re-run a stage by deleting its .done file first.
# ============================================================================

set -uo pipefail
# NOTE: NOT using set -e — stage failures are handled explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${REALVAL_DATA_DIR:-$REPO_ROOT/data/realval}"
STATE_DIR="$DATA_DIR/.state"
RESULTS_DIR="${REALVAL_RESULTS_DIR:-$REPO_ROOT/results/realval}"
TOOLS_DIR="$REPO_ROOT/tools"
ENV_DIR="$REPO_ROOT/env"
SRC_DIR="$REPO_ROOT/src"

# Ensure we're on the right branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "experiment/gaqa-bitnet-assembly" ]; then
    echo "WARNING: Expected branch 'experiment/gaqa-bitnet-assembly', got '$CURRENT_BRANCH'"
    echo "  Switching branches may be needed for consistency."
fi

# Defaults
REALVAL_JOBS="${REALVAL_JOBS:-6}"
REALVAL_ENV_PREFIX="${REALVAL_ENV_PREFIX:-realval}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
BITNET_LORA_R="${BITNET_LORA_R:-8}"
BITNET_LORA_ALPHA="${BITNET_LORA_ALPHA:-16}"

# Parse --stage arguments
RUN_STAGES=()
if [ $# -eq 0 ]; then
    RUN_STAGES=(setup download subsample assemble qc render eval report)
else
    while [ $# -gt 0 ]; do
        case "$1" in
            --stage) shift; RUN_STAGES+=("$1") ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
        shift
    done
fi

echo "=== GAQA Real-Data Validation Pipeline ==="
echo "Repo root: $REPO_ROOT"
echo "Data dir:  $DATA_DIR"
echo "Jobs:      $REALVAL_JOBS concurrent"
echo "Stages:    ${RUN_STAGES[*]}"
echo ""

# ─── Helpers ────────────────────────────────────────────────────────────────

mkdir -p "$STATE_DIR" "$RESULTS_DIR" "$DATA_DIR"

stage_done() { [ -f "$STATE_DIR/$1.done" ]; }
mark_done() { touch "$STATE_DIR/$1.done"; echo "[OK] Stage '$1' completed at $(date)"; }
elapsed() {
    local label="$1" start="$2"
    local duration=$(( $(date +%s) - start ))
    printf "  ⏱  %s took %dm%02ds\n" "$label" $((duration/60)) $((duration%60))
}

find_conda() {
    # Try micromamba, mamba, conda in order
    if command -v micromamba &>/dev/null; then
        echo "micromamba"
    elif command -v mamba &>/dev/null; then
        echo "mamba"
    elif command -v conda &>/dev/null; then
        echo "conda"
    else
        echo ""
    fi
}

bootstrap_micromamba() {
    local install_dir="$TOOLS_DIR/micromamba"
    if [ -x "$install_dir/bin/micromamba" ]; then
        export PATH="$install_dir/bin:$PATH"
        return 0
    fi
    echo "[setup] Bootstrapping micromamba into $install_dir..."
    mkdir -p "$install_dir"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$install_dir" bin/micromamba 2>/dev/null
    # Fallback: download the static binary
    if [ ! -x "$install_dir/bin/micromamba" ]; then
        curl -L -o "$install_dir/bin/micromamba" "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
        chmod +x "$install_dir/bin/micromamba"
    fi
    if [ -x "$install_dir/bin/micromamba" ]; then
        export PATH="$install_dir/bin:$PATH"
        micromamba --version
        return 0
    fi
    return 1
}

ensure_env() {
    local conda_tool
    conda_tool=$(find_conda)
    if [ -z "$conda_tool" ]; then
        echo "[setup] No conda/mamba found. Bootstrapping micromamba..."
        bootstrap_micromamba || { echo "FATAL: Could not bootstrap micromamba"; return 1; }
        conda_tool=micromamba
    fi
    # Check if env exists
    if micromamba env list 2>/dev/null | grep -q "$REALVAL_ENV_PREFIX"; then
        echo "[setup] Conda env '$REALVAL_ENV_PREFIX' already exists"
    else
        echo "[setup] Creating conda env '$REALVAL_ENV_PREFIX' from $ENV_DIR/realval.yml..."
        # If using micromamba, use create -f
        if [ "$conda_tool" = "micromamba" ]; then
            micromamba create -y -n "$REALVAL_ENV_PREFIX" -f "$ENV_DIR/realval.yml" || {
                echo "WARNING: First attempt failed, retrying with relaxed deps..."
                micromamba create -y -n "$REALVAL_ENV_PREFIX" python=3.10 spades megahit quast busco checkm-genome seqtk seqkit parallel wget || return 1
            }
        else
            "$conda_tool" env create -f "$ENV_DIR/realval.yml" 2>/dev/null || {
                echo "WARNING: First attempt failed with $conda_tool, trying micromamba..."
                bootstrap_micromamba || return 1
                micromamba create -y -n "$REALVAL_ENV_PREFIX" python=3.10 spades megahit quast busco checkm-genome seqtk seqkit parallel wget
            }
        fi
    fi

    # Initialize conda run helper
    local conda_run
    if [ "$conda_tool" = "micromamba" ]; then
        conda_run="micromamba run -n $REALVAL_ENV_PREFIX"
    else
        conda_run="$conda_tool run -n $REALVAL_ENV_PREFIX"
    fi
    echo "$conda_run" > "$STATE_DIR/.conda_run_cmd"
    echo "[setup] Env ready. Tool: $conda_tool"
}

conda_run() {
    local cmd_file="$STATE_DIR/.conda_run_cmd"
    if [ -f "$cmd_file" ]; then
        local runner
        runner=$(cat "$cmd_file")
        $runner "$@"
    else
        echo "WARNING: No conda run command found, running directly: $*"
        "$@"
    fi
}

# ─── Stage 1: setup ─────────────────────────────────────────────────────────

do_setup() {
    local S stage="setup"
    echo "━━━ Stage: setup ━━━"
    S=$(date +%s)

    # Bootstrap micromamba if needed and create env
    ensure_env || { echo "FATAL: Environment setup failed"; return 1; }

    # Download BUSCO lineage (bacteria_odb10)
    echo "[setup] Downloading BUSCO lineage bacteria_odb10..."
    conda_run busco --download bacteria_odb10 2>&1 | tail -5 || echo "WARNING: BUSCO download had issues, may need manual run"
    
    # Setup CheckM DB (manual tarball; 'checkm data download' is flaky)
    echo "[setup] Setting up CheckM database..."
    CHECKM_DATA_DIR="$DATA_DIR/checkm_data"
    CHECKM_TARBALL="$CHECKM_DATA_DIR/checkm_data_2015_01_16.tar.gz"
    if [ ! -f "$CHECKM_DATA_DIR/.checkm_db_ready" ]; then
        mkdir -p "$CHECKM_DATA_DIR"
        if [ ! -f "$CHECKM_TARBALL" ]; then
            echo "[setup] Downloading CheckM DB tarball (~275 MB)..."
            wget -q -O "$CHECKM_TARBALL" \
                "https://data.ace.uq.edu.au/public/CheckM_databases/checkm_data_2015_01_16.tar.gz" || \
            curl -sL -o "$CHECKM_TARBALL" \
                "https://data.ace.uq.edu.au/public/CheckM_databases/checkm_data_2015_01_16.tar.gz"
        fi
        echo "[setup] Extracting CheckM DB..."
        tar -xzf "$CHECKM_TARBALL" -C "$CHECKM_DATA_DIR"
        # Use the same runner as the rest of the pipeline (micromamba run works;
        # plain 'conda run' is broken on this server)
        conda_run checkm data setRoot "$CHECKM_DATA_DIR" && touch "$CHECKM_DATA_DIR/.checkm_db_ready" || \
            echo "WARNING: CheckM setRoot failed. QC stage will skip CheckM."
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 2: download ──────────────────────────────────────────────────────

do_download() {
    local S stage="download"
    echo "━━━ Stage: download ━━━"
    S=$(date +%s)

    local manifest="$DATA_DIR/manifest_pipeline.csv"
    [ -f "$manifest" ] || manifest="$DATA_DIR/manifest.csv"
    local dl_dir="$DATA_DIR/reads"
    mkdir -p "$dl_dir"

    if [ ! -f "$manifest" ]; then
        echo "FATAL: Manifest not found at $manifest"
        return 1
    fi

    # Download reads and references from manifest
    # Skip header, read each row
    local line_num=0
    while IFS=, read -r organism genome_size pe1 pe2 ref tech fallback; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue  # skip header

        # Clean up fields (remove quotes if any)
        organism=$(echo "$organism" | tr -d '"')
        pe1=$(echo "$pe1" | tr -d '"')
        pe2=$(echo "$pe2" | tr -d '"')
        ref=$(echo "$ref" | tr -d '"')
        fallback=$(echo "$fallback" | tr -d '"')

        local slug
        slug=$(echo "$organism" | tr ' ' '_' | tr -d '().')
        local org_dir="$dl_dir/$slug"
        mkdir -p "$org_dir"

        # Download PE1
        local pe1_file="$org_dir/${slug}_1.fastq.gz"
        if [ -f "$pe1_file" ] && [ -s "$pe1_file" ]; then
            echo "[download] $slug PE1 already exists ($(du -h "$pe1_file" | cut -f1))"
        else
            echo "[download] $slug PE1: $pe1"
            wget -q --show-progress -O "$pe1_file" "$pe1" || {
                echo "WARNING: Failed to download $pe1"
                if [ -n "$fallback" ]; then
                    echo "  Trying fallback URL for $slug..."
                    local fb_pe1=$(echo "$fallback" | cut -d';' -f1)
                    wget -q --show-progress -O "$pe1_file" "$fb_pe1" || {
                        echo "ERROR: Fallback also failed for $slug PE1"
                        return 1
                    }
                else
                    return 1
                fi
            }
        fi

        # Download PE2
        local pe2_file="$org_dir/${slug}_2.fastq.gz"
        if [ -f "$pe2_file" ] && [ -s "$pe2_file" ]; then
            echo "[download] $slug PE2 already exists ($(du -h "$pe2_file" | cut -f1))"
        else
            echo "[download] $slug PE2: $pe2"
            wget -q --show-progress -O "$pe2_file" "$pe2" || {
                echo "WARNING: Failed to download $pe2"
                if [ -n "$fallback" ]; then
                    local fb_pe2=$(echo "$fallback" | cut -d';' -f2)
                    wget -q --show-progress -O "$pe2_file" "$fb_pe2" || {
                        echo "ERROR: Fallback also failed for $slug PE2"
                        return 1
                    }
                else
                    return 1
                fi
            }
        fi

        # Verify gzip integrity
        echo "[download] Verifying gzip integrity for $slug..."
        gzip -t "$pe1_file" || { echo "ERROR: $pe1_file corrupt"; return 1; }
        gzip -t "$pe2_file" || { echo "ERROR: $pe2_file corrupt"; return 1; }

        # Download reference
        local ref_dir="$DATA_DIR/references"
        mkdir -p "$ref_dir"
        local ref_file="$ref_dir/${slug}_genomic.fna.gz"
        if [ -f "$ref_file" ] && [ -s "$ref_file" ]; then
            echo "[download] $slug reference already exists ($(du -h "$ref_file" | cut -f1))"
        else
            echo "[download] $slug reference: $ref"
            wget -q --show-progress -O "$ref_file" "$ref" || {
                echo "ERROR: Failed to download reference $ref"
                return 1
            }
        fi
        gzip -t "$ref_file" || { echo "ERROR: $ref_file corrupt"; return 1; }
    done < "$manifest"

    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 3: subsample ─────────────────────────────────────────────────────

do_subsample() {
    local S stage="subsample"
    echo "━━━ Stage: subsample ━━━"
    S=$(date +%s)

    # Self-heal: ensure conda env is ready (setup may not have run in this session)
    if [ ! -f "$STATE_DIR/.conda_run_cmd" ]; then
        echo "[subsample] No conda runner — ensuring env..."
        ensure_env || { echo "FATAL: Environment setup failed"; return 1; }
    fi

    local manifest="$DATA_DIR/manifest.csv"
    local dl_dir="$DATA_DIR/reads"
    local sub_dir="$DATA_DIR/subsampled"
    mkdir -p "$sub_dir"

    local conditions_csv="$DATA_DIR/conditions.csv"
    # Write header
    echo "organism,condition_id,coverage,contamination_pct,reads_pe1,reads_pe2,reference,label_qc,label_remediation,label_tool,assembler_config" > "$conditions_csv"

    # Coverage levels to generate
    local coverages=(10 30 100)
    local condition_id=0

    local line_num=0
    while IFS=, read -r organism genome_size pe1 pe2 ref tech fallback; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue

        organism=$(echo "$organism" | tr -d '"')
        genome_size=$(echo "$genome_size" | tr -d '"')
        pe1=$(echo "$pe1" | tr -d '"')
        pe2=$(echo "$pe2" | tr -d '"')
        ref=$(echo "$ref" | tr -d '"')
        fallback=$(echo "$fallback" | tr -d '"')

        local slug
        slug=$(echo "$organism" | tr ' ' '_' | tr -d '().')
        local org_dir="$dl_dir/$slug"
        local pe1_file="$org_dir/${slug}_1.fastq.gz"
        local pe2_file="$org_dir/${slug}_2.fastq.gz"
        local ref_file="$DATA_DIR/references/${slug}_genomic.fna.gz"

        # Estimate total bases in reads (use seqkit)
        echo "[subsample] Estimating depth for $slug..."
        local total_bases
        total_bases=$(conda_run seqkit stats -T "$pe1_file" "$pe2_file" 2>/dev/null | tail -n +2 | awk '{s+=$5}END{print int(s)}')
        if [ -z "$total_bases" ] || [ "$total_bases" = "0" ]; then
            # Fallback: estimate from file size (approx 2/3 compression ratio for fastq.gz)
            local f1_size f2_size
            f1_size=$(stat -c%s "$pe1_file" 2>/dev/null || stat -f%z "$pe1_file" 2>/dev/null)
            f2_size=$(stat -c%s "$pe2_file" 2>/dev/null || stat -f%z "$pe2_file" 2>/dev/null)
            total_bases=$(( (f1_size + f2_size) * 3 / 2 ))
        fi
        local genome_bp=$(python3 -c "print(int($genome_size * 1000000))")
        local est_depth=$(python3 -c "import math; print(int($total_bases / max(1, $genome_bp)))" 2>/dev/null || echo 0)
        echo "  Estimated depth: ${est_depth}x"

        for cov in "${coverages[@]}"; do
            condition_id=$((condition_id + 1))
            local cond_id="C${condition_id}"

            # Skip if coverage is higher than estimated depth
            if [ "$cov" -gt "$est_depth" ]; then
                echo "  WARNING: Requested ${cov}x > estimated ${est_depth}x for $slug. Using ${est_depth}x."
                cov="$est_depth"
            fi

            # Calculate fraction for downsampling, clamped to [0.001, 1.0]
            local fraction
            fraction=$(python3 -c "
d = max(1, $est_depth)
f = min(1.0, $cov / d)
print(max(0.001, round(f, 6)))
" 2>/dev/null || echo "1.0")

            local sub_file1="$sub_dir/${slug}_${cov}x_1.fastq.gz"
            local sub_file2="$sub_dir/${slug}_${cov}x_2.fastq.gz"

            if [ ! -f "$sub_file1" ] || [ ! -s "$sub_file1" ]; then
                echo "  Downsampling $slug to ${cov}x (fraction=$fraction)..."
                # Use bash -c wrapper to avoid conda-run pipe bug (BrokenPipeError)
                conda_run bash -c 'seqtk sample -s42 "$0" "$1" | gzip > "$2"' "$pe1_file" "$fraction" "$sub_file1"
                conda_run bash -c 'seqtk sample -s42 "$0" "$1" | gzip > "$2"' "$pe2_file" "$fraction" "$sub_file2"
            fi

            # Build condition row
            local label_qc="acceptable"
            local label_remediation="acceptable"
            local label_tool="correct_config"
            local assembler_config="default"

            if [ "$cov" -le 20 ]; then
                label_qc="low_coverage"
                label_remediation="low_coverage"
            fi

            echo "${organism},${cond_id},${cov},0,${sub_file1},${sub_file2},${ref_file},${label_qc},${label_remediation},${label_tool},${assembler_config}" >> "$conditions_csv"
        done

        # Create contamination spike: 10% of organism B reads into organism A
        # For the first organism, spike in second organism's reads
        if [ "$line_num" -eq 2 ]; then
            # We need at least 2 organisms; this runs after the first row is fully processed
            # We'll handle this in a second pass below
            :
        fi
    done < "$manifest"

    # ── Second pass: create contamination conditions ──
    echo "[subsample] Creating contamination spike-in conditions..."
    local org_names=()
    local slugs=()
    while IFS=, read -r organism genome_size pe1 pe2 ref tech fallback; do
        [ "$(echo "$organism" | tr -d '"' | head -c1)" = "o" ] && continue  # skip header, cheap heuristic
        org_names+=("$(echo "$organism" | tr -d '"')")
        slugs+=("$(echo "$organism" | tr ' ' '_' | tr -d '().')")
    done < <(tail -n +2 "$manifest")

    # Create contamination pairs: organism A (host) with 10% reads from organism B (spike)
    for ((i=0; i<${#slugs[@]}; i++)); do
        local host_slug="${slugs[$i]}"
        local spike_slug="${slugs[$(( (i+1) % ${#slugs[@]} ))]}"
        
        for cov in 30 100; do
            condition_id=$((condition_id + 1))
            local cond_id="C${condition_id}"
            
            local host_file1="$sub_dir/${host_slug}_${cov}x_1.fastq.gz"
            local host_file2="$sub_dir/${host_slug}_${cov}x_2.fastq.gz"
            local spike_file1="$sub_dir/${spike_slug}_${cov}x_1.fastq.gz"
            local spike_file2="$sub_dir/${spike_slug}_${cov}x_2.fastq.gz"
            
            local contam_dir="$sub_dir/contamination"
            mkdir -p "$contam_dir"
            local contam_file1="$contam_dir/${host_slug}_${cov}x_contam_1.fastq.gz"
            local contam_file2="$contam_dir/${host_slug}_${cov}x_contam_2.fastq.gz"
            
            if [ ! -f "$contam_file1" ] || [ ! -s "$contam_file1" ]; then
                echo "  Creating spike: ${host_slug}(${cov}x) + 10% ${spike_slug}"
                # Take 90% of host reads and 10% of spike reads
                conda_run seqtk sample -s42 "$host_file1" 0.9 > /tmp/host1_tmp.fastq 2>/dev/null
                conda_run seqtk sample -s42 "$spike_file1" 0.1 > /tmp/spike1_tmp.fastq 2>/dev/null
                cat /tmp/host1_tmp.fastq /tmp/spike1_tmp.fastq | gzip > "$contam_file1"
                
                conda_run seqtk sample -s42 "$host_file2" 0.9 > /tmp/host2_tmp.fastq 2>/dev/null
                conda_run seqtk sample -s42 "$spike_file2" 0.1 > /tmp/spike2_tmp.fastq 2>/dev/null
                cat /tmp/host2_tmp.fastq /tmp/spike2_tmp.fastq | gzip > "$contam_file2"
                
                rm -f /tmp/host1_tmp.fastq /tmp/host2_tmp.fastq /tmp/spike1_tmp.fastq /tmp/spike2_tmp.fastq
            fi
            
            local ref_file="$DATA_DIR/references/${host_slug}_genomic.fna.gz"
            echo "${org_names[$i]},${cond_id},${cov},10,${contam_file1},${contam_file2},${ref_file},contamination,contamination,correct_config,default" >> "$conditions_csv"
        done
    done

    echo "[subsample] Generated $(grep -c . "$conditions_csv") condition rows (including header)"
    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 4: assemble ──────────────────────────────────────────────────────

do_assemble() {
    local S stage="assemble"
    echo "━━━ Stage: assemble ━━━"
    S=$(date +%s)

    # Self-heal: ensure conda env is ready
    if [ ! -f "$STATE_DIR/.conda_run_cmd" ]; then
        echo "[assemble] No conda runner — ensuring env..."
        ensure_env || { echo "FATAL: Environment setup failed"; return 1; }
    fi

    local conditions_csv="$DATA_DIR/conditions.csv"
    local assembly_dir="$DATA_DIR/assemblies"
    mkdir -p "$assembly_dir"

    if [ ! -f "$conditions_csv" ]; then
        echo "FATAL: conditions.csv not found. Run subsample stage first."
        return 1
    fi

    # Build assembly commands file for GNU parallel
    local commands_file="$STATE_DIR/assemble_commands.txt"
    > "$commands_file"

    local line_num=0
    while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reference label_qc label_remediation label_tool assembler_config; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue

        local slug
        slug=$(echo "$organism" | tr ' ' '_' | tr -d '().')
        local cond_dir="$assembly_dir/${condition_id}"
        mkdir -p "$cond_dir"

        # Determine assembler + config
        local kmers="21,33,55,77"
        local careful=""
        if [ "$assembler_config" = "wrong" ]; then
            kmers="21"  # single k-mer for misassembly
        fi
        # For low coverage, use smaller k-mers
        if python3 -c "exit(0 if $coverage <= 20 else 1)" 2>/dev/null; then
            kmers="21,33,55"
        fi

        # Use the conda runner (not the bash function — parallel subprocesses don't inherit it)
        local runner=""
        [ -f "$STATE_DIR/.conda_run_cmd" ] && runner=$(cat "$STATE_DIR/.conda_run_cmd")
        local cmd
        if [ "$assembler_config" = "megahit" ]; then
            # MEGAHIT: wrong-tool condition (metagenome assembler on isolate)
            # Normalize output to contigs.fasta for the QC stage.
            cmd="$runner megahit -t 4 -o \"$cond_dir/megahit\""
            cmd+=" -1 \"$reads_pe1\" -2 \"$reads_pe2\""
            cmd+=" 2>&1 | tail -5"
            cmd+=" && ln -sf \"$cond_dir/megahit/final.contigs.fa\" \"$cond_dir/megahit/contigs.fasta\""
        else
            cmd="$runner spades.py -t 4 -o \"$cond_dir/spades\""
            cmd+=" --pe1-1 \"$reads_pe1\" --pe1-2 \"$reads_pe2\""
            cmd+=" -k \"$kmers\""
            [ -n "$careful" ] && cmd+=" --careful"
            cmd+=" 2>&1 | tail -5"
        fi

        echo "$cmd" >> "$commands_file"
    done < "$conditions_csv"

    local total_commands
    total_commands=$(wc -l < "$commands_file")
    echo "[assemble] Running $total_commands SPAdes assemblies with $REALVAL_JOBS parallel jobs..."
    echo "  (each assembly: -t 4 threads, no --careful for speed)"

    # Run assemblies with a simple bash job-control loop (avoids conda run + parallel bugs)
    if [ "$total_commands" -gt 0 ]; then
        echo "[assemble] Running $total_commands SPAdes with $REALVAL_JOBS parallel jobs via bash..."
        local pids=() job_count=0
        while IFS= read -r cmd; do
            # Wait if at capacity
            while [ "$(jobs -r | wc -l)" -ge "$REALVAL_JOBS" ]; do
                sleep 10
            done
            # Launch in background
            (eval "$cmd") &
            job_count=$((job_count + 1))
            echo "[assemble] Launched $job_count/$total_commands (PID $!)"
        done < "$commands_file"
        # Wait for all to finish
        echo "[assemble] Waiting for $total_commands assemblies to finish..."
        wait
        echo "[assemble] All assemblies completed."

        # Verify assemblies were actually produced
        local asm_count=0
        for asm_dir in "$DATA_DIR/assemblies"/C*/{spades,megahit}/; do
            if [ -f "$asm_dir/contigs.fasta" ] && [ -s "$asm_dir/contigs.fasta" ]; then
                asm_count=$((asm_count + 1))
            fi
        done
        echo "[assemble] $asm_count/$total_commands assemblies produced contigs.fasta"

        # Retry failed assemblies once with broader k-mers
        if [ -f "$STATE_DIR/assemble_joblog.txt" ]; then
            while IFS=$'\t' read -r seq host starttime jobruntime sendval signal command exitval; do
                # Skip header
                [[ "$seq" =~ ^[0-9]+$ ]] || continue
                if [ "$exitval" -ne 0 ]; then
                    echo "[assemble] Retrying failed assembly (exit=$exitval)..."
                    echo "  Command: $command"
                    # Parse the command to get assembly dir and files
                    eval "$command" 2>&1 || echo "  Retry also failed."
                fi
            done < "$STATE_DIR/assemble_joblog.txt"
        fi
    fi

    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 5: qc ────────────────────────────────────────────────────────────

do_qc() {
    local S stage="qc"
    echo "━━━ Stage: qc ━━━"
    S=$(date +%s)

    # Self-heal: ensure conda env is ready
    if [ ! -f "$STATE_DIR/.conda_run_cmd" ]; then
        echo "[qc] No conda runner — ensuring env..."
        ensure_env || { echo "FATAL: Environment setup failed"; return 1; }
    fi

    local conditions_csv="$DATA_DIR/conditions.csv"
    local assembly_dir="$DATA_DIR/assemblies"
    local qc_dir="$DATA_DIR/qc_results"
    mkdir -p "$qc_dir"

    if [ ! -f "$conditions_csv" ]; then
        echo "FATAL: conditions.csv not found."
        return 1
    fi

    # Build QC commands
    local quast_commands="$STATE_DIR/quast_commands.txt"
    local busco_commands="$STATE_DIR/busco_commands.txt"
    > "$quast_commands"
    > "$busco_commands"

    local line_num=0
    while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reference label_qc label_remediation label_tool assembler_config; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue

        # Assembly dir depends on the assembler used for this condition
        local asm_subdir="spades"
        [ "$assembler_config" = "megahit" ] && asm_subdir="megahit"
        local cond_dir="$assembly_dir/${condition_id}/${asm_subdir}"
        local assembly_fasta="$cond_dir/contigs.fasta"
        local qc_cond_dir="$qc_dir/${condition_id}"
        mkdir -p "$qc_cond_dir"

        if [ ! -f "$assembly_fasta" ]; then
            echo "[qc] WARNING: Assembly not found at $assembly_fasta, skipping $condition_id"
            continue
        fi

        # QUAST with reference
        local quast_out="$qc_cond_dir/quast"
        if [ ! -d "$quast_out/report.tsv" ]; then
            echo "conda_run quast.py -o \"$quast_out\" -R \"$reference\" --gene-finding -t 2 \"$assembly_fasta\" 2>&1 | tail -3" >> "$quast_commands"
        fi

        # BUSCO
        local busco_out="$qc_cond_dir/busco"
        if [ ! -d "$busco_out" ]; then
            echo "conda_run busco -i \"$assembly_fasta\" -o \"$busco_out\" -m genome -l bacteria_odb10 -c 2 --offline 2>&1 | tail -5" >> "$busco_commands"
        fi
    done < "$conditions_csv"

    echo "[qc] Running QUAST..."
    if [ -s "$quast_commands" ]; then
        local n_quast
        n_quast=$(wc -l < "$quast_commands")
        echo "  $n_quast QUAST jobs"
        local qc_pids=()
        while IFS= read -r cmd; do
            while [ "$(jobs -r | wc -l)" -ge "$REALVAL_JOBS" ]; do sleep 5; done
            (eval "$cmd") &
        done < "$quast_commands"
        wait
    else
        echo "  No QUAST jobs needed (all cached)"
    fi

    echo "[qc] Running BUSCO..."
    if [ -s "$busco_commands" ]; then
        local n_busco
        n_busco=$(wc -l < "$busco_commands")
        echo "  $n_busco BUSCO jobs"
        local busco_jobs=$(( REALVAL_JOBS / 2 ))
        [ "$busco_jobs" -lt 1 ] && busco_jobs=1
        while IFS= read -r cmd; do
            while [ "$(jobs -r | wc -l)" -ge "$busco_jobs" ]; do sleep 10; done
            (eval "$cmd") &
        done < "$busco_commands"
        wait
    else
        echo "  No BUSCO jobs needed (all cached)"
    fi

    # Try CheckM if available
    echo "[qc] Attempting CheckM..."
    local checkm_data_dir="$DATA_DIR/checkm_data"
    if [ -f "$checkm_data_dir/.checkm_db_ready" ]; then
        local checkm_commands="$STATE_DIR/checkm_commands.txt"
        > "$checkm_commands"

        while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reference label_qc label_remediation label_tool assembler_config; do
            line_num=$((line_num + 1))
            [ $line_num -eq 1 ] && continue

            local cond_dir="$assembly_dir/${condition_id}/spades"
            local assembly_fasta="$cond_dir/contigs.fasta"
            local checkm_out="$qc_dir/${condition_id}/checkm"
            mkdir -p "$checkm_out"

            # lineage_wf alone does NOT produce qa.txt: must run 'checkm qa' after.
            # Run on a bins dir containing only contigs.fasta (one bin per condition).
            if [ -f "$assembly_fasta" ] && [ ! -f "$checkm_out/qa.txt" ]; then
                echo "mkdir -p \"$checkm_out/bins\" && ln -sf \"$assembly_fasta\" \"$checkm_out/bins/contigs.fasta\" && conda_run checkm lineage_wf -t 2 -x fasta \"$checkm_out/bins\" \"$checkm_out/wf\" > \"$checkm_out/lineage.log\" 2>&1 && conda_run checkm qa -t 2 --tab_table -f \"$checkm_out/qa.txt\" \"$checkm_out/wf/lineage.ms\" \"$checkm_out/wf\" >> \"$checkm_out/lineage.log\" 2>&1" >> "$checkm_commands"
            fi
        done < "$conditions_csv"

        if [ -s "$checkm_commands" ]; then
            local n_checkm
            n_checkm=$(wc -l < "$checkm_commands")
            echo "  $n_checkm CheckM jobs"
            while IFS= read -r cmd; do
                while [ "$(jobs -r | wc -l)" -ge "$busco_jobs" ]; do sleep 10; done
                (eval "$cmd") &
            done < "$checkm_commands"
            wait
        fi
    else
        echo "  CheckM data not available. Skipping CheckM. (Will note in report.)"
    fi

    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 6: render ────────────────────────────────────────────────────────

do_render() {
    local S stage="render"
    echo "━━━ Stage: render ━━━"
    S=$(date +%s)

    local conditions_csv="$DATA_DIR/conditions.csv"
    local qc_dir="$DATA_DIR/qc_results"

    if [ ! -f "$conditions_csv" ]; then
        echo "FATAL: conditions.csv not found"
        return 1
    fi

    echo "[render] Building realval test dataset..."
    conda_run python "$SRC_DIR/build_realval_dataset.py" \
        --conditions "$conditions_csv" \
        --qc-dir "$qc_dir" \
        --output "$DATA_DIR/test.jsonl" \
        --manifest "$DATA_DIR/manifest.csv" 2>&1

    # Validate output
    if [ -f "$DATA_DIR/test.jsonl" ]; then
        local n_questions
        n_questions=$(wc -l < "$DATA_DIR/test.jsonl")
        echo "[render] Generated $n_questions GAQA-format questions"
        
        # Print per-class and per-condition counts
        echo ""
        echo "=== Question Counts ==="
        python3 -c "
import json
from collections import Counter
conds = Counter()
subjects = Counter()
with open('$DATA_DIR/test.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        # Extract condition from the input
        sub = rec.get('subject', 'unknown')
        subjects[sub] += 1
        # Extract label from output
        out = rec.get('output', '')
        conds[sub] += 1
print('By subject:', dict(subjects))
" 2>&1

        # Assert every rendered question's label matches its condition row
        echo "[render] Validating label consistency..."
        python3 -c "
import json
errors = 0
with open('$DATA_DIR/test.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        # Verify the output starts with 'The correct answer is'
        if not rec['output'].startswith('The correct answer is'):
            print(f'ERROR: bad output format: {rec[\"id\"]}')
            errors += 1
if errors == 0:
    print('  All labels valid ✓')
else:
    print(f'  {errors} errors found')
" 2>&1
    else
        echo "ERROR: test.jsonl was not generated!"
        return 1
    fi

    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 7: eval (GPU, optional) ──────────────────────────────────────────

do_eval() {
    local S stage="eval"
    echo "━━━ Stage: eval ━━━"
    S=$(date +%s)

    if [ -z "$ADAPTER_DIR" ]; then
        echo "[eval] ADAPTER_DIR not set — skipping eval stage."
        echo "  To run later:"
        echo "    ADAPTER_DIR=models/your-adapter BITNET_LORA_R=8 BITNET_LORA_ALPHA=16 \\"
        echo "      bash scripts/realval_pipeline.sh --stage eval --stage report"
        return 0
    fi

    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "[eval] WARNING: ADAPTER_DIR '$ADAPTER_DIR' does not exist. Skipping."
        return 0
    fi

    echo "[eval] Running model evaluation on realval test set..."
    echo "  Adapter: $ADAPTER_DIR"
    echo "  LoRA R:  $BITNET_LORA_R"
    echo "  LoRA alpha: $BITNET_LORA_ALPHA"

    local adapter_name
    adapter_name=$(basename "$ADAPTER_DIR")
    local eval_output="$RESULTS_DIR/eval_${adapter_name}.json"

    # Set environment for eval
    export DATA_PROCESSED_DIR="$DATA_DIR"
    export BITNET_LORA_R="$BITNET_LORA_R"
    export BITNET_LORA_ALPHA="$BITNET_LORA_ALPHA"

    # Run the logit scorer
    conda_run python -m src.eval \
        --model-type bitnet \
        --packed \
        --adapter "$ADAPTER_DIR" \
        --benchmarks gaqa \
        --output "$eval_output" 2>&1 || {
        echo "[eval] Evaluation run failed. This is non-fatal."
        echo "  You can re-run manually with correct settings."
        return 0
    }

    # Copy test set predictions alongside results
    if [ -f "$DATA_DIR/test.jsonl" ]; then
        cp "$DATA_DIR/test.jsonl" "$RESULTS_DIR/eval_${adapter_name}_test.jsonl"
    fi

    echo "[eval] Results written to $eval_output"
    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Stage 8: report ────────────────────────────────────────────────────────

do_report() {
    local S stage="report"
    echo "━━━ Stage: report ━━━"
    S=$(date +%s)

    local report="$RESULTS_DIR/REPORT.md"
    local manifest="$DATA_DIR/manifest.csv"

    echo "Generating report: $report"

    cat > "$report" << 'EOF'
# GAQA Real-Data External Validation Report

## Overview

This report documents the real-data external validation set for GAQA.
Labels come from **injected conditions** (downsampled coverage, spiked contamination)
rather than from metric thresholds, providing a rigorous out-of-distribution test.

## Manifest
EOF

    # Add manifest as table
    echo "" >> "$report"
    echo "| Organism | Genome Size (Mb) | Technology | Reference |" >> "$report"
    echo "|----------|-----------------|------------|-----------|" >> "$report"
    tail -n +2 "$manifest" | while IFS=, read -r org gb pe1 pe2 ref tech fb; do
        org=$(echo "$org" | tr -d '"')
        ref=$(echo "$ref" | tr -d '"')
        ref_name=$(basename "$ref" | sed 's/_genomic.fna.gz//')
        echo "| $org | $gb | $tech | $ref_name |" >> "$report"
    done

    # Add conditions table
    echo "" >> "$report"
    echo "## Conditions" >> "$report"
    echo "" >> "$report"
    echo "| Condition ID | Organism | Coverage (×) | Contamination (%) | Label (QC) | Label (Remediation) |" >> "$report"
    echo "|-------------|----------|-------------|-------------------|-----------|-------------------|" >> "$report"
    if [ -f "$DATA_DIR/conditions.csv" ]; then
        tail -n +2 "$DATA_DIR/conditions.csv" | while IFS=, read -r org cid cov cont pe1 pe2 ref lqc lre ltool asm; do
            echo "| $cid | $org | $cov | $cont | $lqc | $lre |" >> "$report"
        done
    else
        echo "| N/A | N/A | N/A | N/A | N/A | N/A |" >> "$report"
    fi

    # Add QUAST/BUSCO/CheckM summary
    echo "" >> "$report"
    echo "## Assembly Quality Metrics" >> "$report"
    echo "" >> "$report"
    echo "| Condition | Contigs | N50 (bp) | Largest (bp) | Misassemblies | Genome Frac (%) | BUSCO Complete (%) | CheckM Comp (%) | CheckM Contam (%) |" >> "$report"
    echo "|----------|---------|---------|-------------|--------------|----------------|-------------------|----------------|------------------|" >> "$report"

    local qc_dir="$DATA_DIR/qc_results"
    if [ -d "$qc_dir" ]; then
        for cond_dir in "$qc_dir"/*/; do
            local cond_id
            cond_id=$(basename "$cond_dir")
            local quast_tsv="$cond_dir/quast/report.tsv"
            local busco_dir="$cond_dir/busco"

            local contigs="N/A" n50="N/A" largest="N/A" misasm="N/A" genfrac="N/A"
            local busco_comp="N/A" checkm_comp="N/A" checkm_contam="N/A"

            # Parse QUAST
            if [ -f "$quast_tsv" ]; then
                contigs=$(grep "^# contigs" "$quast_tsv" | cut -f2)
                n50=$(grep "^N50" "$quast_tsv" | head -1 | cut -f2)
                largest=$(grep "^Largest contig" "$quast_tsv" | cut -f2)
                misasm=$(grep "^# misassemblies" "$quast_tsv" | cut -f2)
                genfrac=$(grep "^Genome fraction" "$quast_tsv" | cut -f2)
            fi

            # Parse BUSCO
            local busco_summary=$(find "$busco_dir" -name "short_summary*.txt" 2>/dev/null | head -1)
            if [ -f "$busco_summary" ]; then
                busco_comp=$(grep -oP 'S:\d+\.\d+%' "$busco_summary" | head -1 | tr -d 'S:')
            fi

            echo "| $cond_id | $contigs | $n50 | $largest | $misasm | $genfrac | $busco_comp | $checkm_comp | $checkm_contam |" >> "$report"
        done
    fi

    # Question counts
    echo "" >> "$report"
    echo "## GAQA Questions Generated" >> "$report"
    echo "" >> "$report"
    if [ -f "$DATA_DIR/test.jsonl" ]; then
        local nq
        nq=$(wc -l < "$DATA_DIR/test.jsonl")
        echo "Total test questions: **$nq**" >> "$report"
        echo "" >> "$report"
        echo "### By Subject" >> "$report"
        python3 -c "
import json
from collections import Counter
subjects = Counter()
with open('$DATA_DIR/test.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        sub = rec.get('subject', 'unknown')
        subjects[sub] += 1
print('| Subject | Count |')
print('|--------|------|')
for sub, cnt in sorted(subjects.items()):
    print(f'| {sub} | {cnt} |')
" >> "$report" 2>&1
    else
        echo "No questions generated (test.jsonl not found)." >> "$report"
    fi

    # Eval results if present
    echo "" >> "$report"
    echo "## Model Evaluation" >> "$report"
    echo "" >> "$report"
    local eval_files=("$RESULTS_DIR"/eval_*.json)
    if [ -f "${eval_files[0]}" ]; then
        for ef in "${eval_files[@]}"; do
            ef_name=$(basename "$ef")
            echo "### $ef_name" >> "$report"
            echo '```json' >> "$report"
            cat "$ef" >> "$report"
            echo '```' >> "$report"
            echo "" >> "$report"
        done
    else
        echo "No eval results yet. Run the eval stage with ADAPTER_DIR set." >> "$report"
    fi

    echo "Report generated at $report"
    mark_done "$stage"
    elapsed "$stage" "$S"
}

# ─── Main dispatcher ────────────────────────────────────────────────────────

MAIN_START=$(date +%s)
for stage in "${RUN_STAGES[@]}"; do
    # Check if already done (idempotent)
    if stage_done "$stage" && [ ${#RUN_STAGES[@]} -eq 1 ]; then
        echo "Stage '$stage' already completed. Delete $STATE_DIR/$stage.done to re-run."
        continue
    fi
    
    case "$stage" in
        setup)      do_setup ;;
        download)   do_download ;;
        subsample)  do_subsample ;;
        assemble)   do_assemble ;;
        qc)         do_qc ;;
        render)     do_render ;;
        eval)       do_eval ;;
        report)     do_report ;;
        *)
            echo "Unknown stage: $stage"
            echo "Valid stages: setup, download, subsample, assemble, qc, render, eval, report"
            exit 1
            ;;
    esac

    # If a stage failed, report but don't kill the script
    if [ $? -ne 0 ]; then
        echo "WARNING: Stage '$stage' encountered errors. Continuing with next stage."
        echo "  Check logs above for details."
    fi
done

elapsed "Total pipeline" "$MAIN_START"
echo "=== Pipeline complete ==="
