#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${REALVAL_DATA_DIR:-$REPO_ROOT/data/realval}"
DATA_DIR="$(realpath "$DATA_DIR")"
STATE_DIR="$DATA_DIR/.state"
RESULTS_DIR="${REALVAL_RESULTS_DIR:-$REPO_ROOT/results/realval}"
TOOLS_DIR="$REPO_ROOT/tools"
ENV_DIR="$REPO_ROOT/env"
SRC_DIR="$REPO_ROOT/src"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "experiment/gaqa-bitnet-assembly" ]; then
    echo "WARNING: Expected branch 'experiment/gaqa-bitnet-assembly', got '$CURRENT_BRANCH'"
    echo "  Switching branches may be needed for consistency."
fi

REALVAL_JOBS="${REALVAL_JOBS:-12}"
REALVAL_THREADS="${REALVAL_THREADS:-12}"
REALVAL_ENV_PREFIX="${REALVAL_ENV_PREFIX:-realval}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
BITNET_LORA_R="${BITNET_LORA_R:-8}"
BITNET_LORA_ALPHA="${BITNET_LORA_ALPHA:-16}"

RUN_STAGES=()
if [ $# -eq 0 ]; then
    RUN_STAGES=(setup download subsample assemble qc render eval)
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

mkdir -p "$STATE_DIR" "$RESULTS_DIR" "$DATA_DIR"

stage_done() { [ -f "$STATE_DIR/$1.done" ]; }
mark_done() { touch "$STATE_DIR/$1.done"; echo "[OK] Stage '$1' completed at $(date)"; }
elapsed() {
    local label="$1" start="$2"
    local duration=$(( $(date +%s) - start ))
    printf "  ⏱  %s took %dm%02ds\n" "$label" $((duration/60)) $((duration%60))
}

find_conda() {
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
    if micromamba env list 2>/dev/null | grep -q "$REALVAL_ENV_PREFIX"; then
        echo "[setup] Conda env '$REALVAL_ENV_PREFIX' already exists"
    else
        echo "[setup] Creating conda env '$REALVAL_ENV_PREFIX' from $ENV_DIR/realval.yml..."
        if [ "$conda_tool" = "micromamba" ]; then
        micromamba create -y -n "$REALVAL_ENV_PREFIX" -f "$ENV_DIR/realval.yml" || {
            echo "WARNING: First attempt failed, retrying with relaxed deps..."
            micromamba create -y -c conda-forge -c bioconda -n "$REALVAL_ENV_PREFIX" \
                python=3.10 spades megahit quast hmmer prodigal checkm-genome \
                seqtk seqkit parallel wget bbmap flye canu unicycler kraken2 pilon assembly-stats && \
            micromamba run -n "$REALVAL_ENV_PREFIX" pip install \
                "busco @ git+https://gitlab.com/ezlab/busco.git@5.8.3" || return 1
        }
        else
            "$conda_tool" env create -f "$ENV_DIR/realval.yml" 2>/dev/null || {
                echo "WARNING: First attempt failed with $conda_tool, trying micromamba..."
                bootstrap_micromamba || return 1
                micromamba create -y -c conda-forge -c bioconda -n "$REALVAL_ENV_PREFIX" \
                    python=3.10 spades megahit quast hmmer prodigal checkm-genome \
                    seqtk seqkit parallel wget bbmap flye canu unicycler kraken2 pilon assembly-stats && \
                micromamba run -n "$REALVAL_ENV_PREFIX" pip install \
                    "busco @ git+https://gitlab.com/ezlab/busco.git@5.8.3"
            }
        fi
    fi

    local conda_run
    if [ "$conda_tool" = "micromamba" ]; then
        conda_run="micromamba run -n $REALVAL_ENV_PREFIX"
    else
        local conda_base
        conda_base=$("$conda_tool" info --base 2>/dev/null) || conda_base="$conda_tool"
        local wrapper="$STATE_DIR/run_in_env.sh"
        printf '#!/bin/bash\n' > "$wrapper"
        printf 'eval "$(%s/bin/conda shell.posix hook)"\n' "$conda_base" >> "$wrapper"
        printf 'conda activate %s 2>/dev/null\n' "$REALVAL_ENV_PREFIX" >> "$wrapper"
        printf 'exec "$@"\n' >> "$wrapper"
        chmod +x "$wrapper"
        conda_run="bash $wrapper"
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

do_setup() {
    local S stage="setup"
    echo "━━━ Stage: setup ━━━"
    S=$(date +%s)

    ensure_env || { echo "FATAL: Environment setup failed"; return 1; }

    echo "[setup] Ensuring BUSCO deps (biopython, numpy, pandas, requests, bbmap)..."
    conda_run pip install biopython numpy pandas requests 2>/dev/null || true
    conda_run conda install -c bioconda -n $REALVAL_ENV_PREFIX bbmap -y 2>/dev/null || \
        echo "[setup] bbmap install skipped (BUSCO may fail for genome mode — install manually)"
    if conda_run python -c "import busco" 2>/dev/null; then
        echo "[setup] BUSCO already installed."
    else
        conda_run pip install "git+https://gitlab.com/ezlab/busco.git@5.8.3" || {
            echo "WARNING: BUSCO pip install failed. QC stage will skip BUSCO."
        }
    fi

    echo "[setup] Downloading BUSCO lineage bacteria_odb10..."
    conda_run busco --download bacteria_odb10 2>&1 | tail -5 || echo "WARNING: BUSCO download had issues, may need manual run"

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
        conda_run checkm data setRoot "$CHECKM_DATA_DIR" && touch "$CHECKM_DATA_DIR/.checkm_db_ready" || \
            echo "WARNING: CheckM setRoot failed. QC stage will skip CheckM."
    fi

    mkdir -p "$RESULTS_DIR"

    mark_done "$stage"
    elapsed "$stage" "$S"
}

do_download() {
    local S stage="download"
    echo "━━━ Stage: download ━━━"
    S=$(date +%s)

    local manifest="$DATA_DIR/manifest.csv"
    local dl_dir="$DATA_DIR/reads"
    mkdir -p "$dl_dir"

    if [ ! -f "$manifest" ]; then
        echo "FATAL: Manifest not found at $manifest"
        return 1
    fi

    local line_num=0
    while IFS=, read -r organism genome_size pe1 pe2 ref tech fallback long_reads_url; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue  # skip header

        organism=$(echo "$organism" | tr -d '"')
        pe1=$(echo "$pe1" | tr -d '"')
        pe2=$(echo "$pe2" | tr -d '"')
        ref=$(echo "$ref" | tr -d '"')
        fallback=$(echo "$fallback" | tr -d '"')
        long_reads_url=$(echo "$long_reads_url" | tr -d '"')
        local org_dir="$dl_dir/$slug"
        mkdir -p "$org_dir"

        local pe1_file="$org_dir/${slug}_1.fastq.gz"
        if [ -f "$pe1_file" ] && [ -s "$pe1_file" ] && gzip -t "$pe1_file" 2>/dev/null; then
            echo "[download] $slug PE1 OK ($(du -h "$pe1_file" | cut -f1))"
        else
            [ -f "$pe1_file" ] && rm -f "$pe1_file"
            echo "[download] $slug PE1: $pe1"
            curl --retry-all-errors --retry 20 --retry-delay 30 -L -o "$pe1_file" "${pe1/https/http}" || {
                echo "WARNING: Failed to download $pe1"
                if [ -n "$fallback" ]; then
                    echo "  Trying fallback URL for $slug..."
                    local fb_pe1=$(echo "$fallback" | cut -d';' -f1)
                    [ -n "$fb_pe1" ] && curl --retry-all-errors --retry 20 --retry-delay 30 -L -o "$pe1_file" "$fb_pe1" || {
                        echo "ERROR: Fallback also failed for $slug PE1"
                        return 1
                    }
                else
                    return 1
                fi
            }
        fi

        sleep 5

        local pe2_file="$org_dir/${slug}_2.fastq.gz"
        if [ -f "$pe2_file" ] && [ -s "$pe2_file" ] && gzip -t "$pe2_file" 2>/dev/null; then
            echo "[download] $slug PE2 OK ($(du -h "$pe2_file" | cut -f1))"
        else
            [ -f "$pe2_file" ] && rm -f "$pe2_file"
            echo "[download] $slug PE2: $pe2"
            curl --retry-all-errors --retry 20 --retry-delay 30 -L -o "$pe2_file" "${pe2/https/http}" || {
                echo "WARNING: Failed to download $pe2"
                if [ -n "$fallback" ]; then
                    local fb_pe2=$(echo "$fallback" | cut -d';' -f2)
                    [ -n "$fb_pe2" ] && curl --retry-all-errors --retry 20 --retry-delay 30 -L -o "$pe2_file" "$fb_pe2" || {
                        echo "ERROR: Fallback also failed for $slug PE2"
                        return 1
                    }
                else
                    return 1
                fi
            }
        fi

        echo "[download] Verifying gzip integrity for $slug..."
        gzip -t "$pe1_file" || { echo "ERROR: $pe1_file corrupt"; return 1; }
        gzip -t "$pe2_file" || { echo "ERROR: $pe2_file corrupt"; return 1; }

        local ref_dir="$DATA_DIR/references"
        mkdir -p "$ref_dir"
        local ref_file="$ref_dir/${slug}_genomic.fna.gz"
        if [ -f "$ref_file" ] && [ -s "$ref_file" ]; then
            echo "[download] $slug reference already exists ($(du -h "$ref_file" | cut -f1))"
        else
            echo "[download] $slug reference: $ref"
            wget -c -q --show-progress --tries=5 --timeout=60 -O "$ref_file" "$ref" || {
                echo "ERROR: Failed to download reference $ref"
                return 1
            }
        fi
        gzip -t "$ref_file" || { echo "ERROR: $ref_file corrupt"; return 1; }

        if [ -n "$long_reads_url" ]; then
            local lr_dir="$DATA_DIR/longreads"
            mkdir -p "$lr_dir"
            local lr_file="$lr_dir/${slug}_long.fastq.gz"
            if [ -f "$lr_file" ] && [ -s "$lr_file" ] && gzip -t "$lr_file" 2>/dev/null; then
                echo "[download] $slug long-reads OK ($(du -h "$lr_file" | cut -f1))"
            else
                [ -f "$lr_file" ] && rm -f "$lr_file"
                echo "[download] $slug long-reads: $long_reads_url"
                curl --retry-all-errors --retry 20 --retry-delay 30 -L -o "$lr_file" "${long_reads_url/https/http}" || {
                    echo "ERROR: Failed to download long-reads for $slug: $long_reads_url"
                    return 1
                }
                gzip -t "$lr_file" || { echo "ERROR: $lr_file corrupt"; return 1; }
            fi
        fi
    done < "$manifest"

    mark_done "$stage"
    elapsed "$stage" "$S"
}

do_subsample() {
    local S stage="subsample"
    echo "━━━ Stage: subsample ━━━"
    S=$(date +%s)

    if [ ! -f "$STATE_DIR/.conda_run_cmd" ]; then
        echo "[subsample] No conda runner — ensuring env..."
        ensure_env || { echo "FATAL: Environment setup failed"; return 1; }
    fi

    local manifest="$DATA_DIR/manifest.csv"
    local dl_dir="$DATA_DIR/reads"
    local sub_dir="$DATA_DIR/subsampled"
    mkdir -p "$sub_dir"

    local conditions_csv="$DATA_DIR/conditions.csv"
    echo "organism,condition_id,coverage,contamination_pct,reads_pe1,reads_pe2,reference,label_qc,label_remediation,label_tool,assembler_config" > "$conditions_csv"

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

        echo "[subsample] Estimating depth for $slug..."
        local total_bases
        total_bases=$(conda_run seqkit stats -T "$pe1_file" "$pe2_file" 2>/dev/null | tail -n +2 | awk '{s+=$5}END{print int(s)}')
        if [ -z "$total_bases" ] || [ "$total_bases" = "0" ]; then
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

            if [ "$cov" -gt "$est_depth" ]; then
                echo "  WARNING: Requested ${cov}x > estimated ${est_depth}x for $slug. Using ${est_depth}x."
                cov="$est_depth"
            fi

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
                conda_run bash -c 'seqtk sample -s42 "$0" "$1" | gzip > "$2"' "$pe1_file" "$fraction" "$sub_file1"
                conda_run bash -c 'seqtk sample -s42 "$0" "$1" | gzip > "$2"' "$pe2_file" "$fraction" "$sub_file2"
            fi

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

        if [ "$line_num" -eq 2 ]; then
            :
        fi
    done < "$manifest"

    echo "[subsample] Creating contamination spike-in conditions..."
    local org_names=()
    local slugs=()
    while IFS=, read -r organism genome_size pe1 pe2 ref tech fallback; do
        [ "$(echo "$organism" | tr -d '"' | head -c1)" = "o" ] && continue  # skip header, cheap heuristic
        org_names+=("$(echo "$organism" | tr -d '"')")
        slugs+=("$(echo "$organism" | tr ' ' '_' | tr -d '().')")
    done < <(tail -n +2 "$manifest")

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

do_assemble() {
    local S stage="assemble"
    echo "━━━ Stage: assemble ━━━"
    S=$(date +%s)

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

    local runner=""
    [ -f "$STATE_DIR/.conda_run_cmd" ] && runner=$(cat "$STATE_DIR/.conda_run_cmd")
    local error_log="$DATA_DIR/assembly_errors.log"
    > "$error_log"

    local commands_file="$STATE_DIR/assemble_commands.txt"
    > "$commands_file"

    local total=0 pending=0 skip=0
    local line_num=0
    while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reads_long long_read_type reference label_qc label_remediation label_tool assembler_config; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue
        total=$((total + 1))
        assembler_config="${assembler_config%$'\r'}"

        local cond_dir="$assembly_dir/${condition_id}"
        local asm_subdir="spades"
        [ "$assembler_config" = "megahit" ] && asm_subdir="megahit"
        [ "$assembler_config" = "flye" ] && asm_subdir="flye"
        [ "$assembler_config" = "canu" ] && asm_subdir="canu"
        [ "$assembler_config" = "unicycler" ] && asm_subdir="unicycler"

        if { [ -f "$cond_dir/spades/contigs.fasta" ] && [ -s "$cond_dir/spades/contigs.fasta" ]; } || \
           { [ -f "$cond_dir/megahit/contigs.fasta" ] && [ -s "$cond_dir/megahit/contigs.fasta" ]; } || \
           { [ -f "$cond_dir/flye/contigs.fasta" ] && [ -s "$cond_dir/flye/contigs.fasta" ]; } || \
           { [ -f "$cond_dir/canu/contigs.fasta" ] && [ -s "$cond_dir/canu/contigs.fasta" ]; } || \
           { [ -f "$cond_dir/unicycler/contigs.fasta" ] && [ -s "$cond_dir/unicycler/contigs.fasta" ]; }; then
            skip=$((skip + 1))
            continue
        fi

        pending=$((pending + 1))
        mkdir -p "$cond_dir"

        local kmers="21,33,55,77"
        if [ "$assembler_config" = "wrong" ]; then
            kmers="21"
        fi
        if python3 -c "exit(0 if $coverage <= 20 else 1)" 2>/dev/null; then
            kmers="21,33,55"
        fi

        local log_file="$cond_dir/assembly.log"
        local err_file="$cond_dir/assembly.err.log"

        local actual_assembler="$assembler_config"
        if [ "$actual_assembler" != "megahit" ] && [ "$actual_assembler" != "wrong" ] && \
           [ "$actual_assembler" != "flye" ] && [ "$actual_assembler" != "canu" ] && \
           [ "$actual_assembler" != "unicycler" ]; then
            local rlen
            rlen=$(zcat "$reads_pe1" 2>/dev/null | head -4 | tail -1 | tr -d '\n' | wc -c 2>/dev/null)
            if [ -n "$rlen" ] && [ "$rlen" -gt 0 ] && [ "$rlen" -lt 75 ]; then
                echo "[assemble] $condition_id: short reads (~${rlen}bp) — using MEGAHIT"
                actual_assembler="megahit"
            fi
        fi

        if [ "$actual_assembler" = "megahit" ]; then
            asm_subdir="megahit"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$condition_id" "$organism" "$coverage" "$contamination" \
                   "$reads_pe1" "$reads_pe2" "$asm_subdir" \
                   "rm -rf \"$cond_dir/megahit\" && $runner megahit -t $REALVAL_THREADS -o \"$cond_dir/megahit\" -1 \"$reads_pe1\" -2 \"$reads_pe2\" > \"$log_file\" 2> \"$err_file\" && ln -sf \"$cond_dir/megahit/final.contigs.fa\" \"$cond_dir/megahit/contigs.fasta\"" \
                >> "$commands_file"
        elif [ "$actual_assembler" = "flye" ]; then
            asm_subdir="flye"
            local gsize_mb=5.0
            local gsize_line file
            file="$DATA_DIR/manifest_full.csv"
            [ -f "$file" ] || file="$DATA_DIR/manifest.csv"
            if [ -f "$file" ]; then
                gsize_line=$(grep -F "$organism" "$file" 2>/dev/null | head -1)
                gsize_mb=$(echo "$gsize_line" | cut -d, -f2 | tr -d '" ' || echo "5.0")
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$condition_id" "$organism" "$coverage" "$contamination" \
                   "$reads_long" "" "$asm_subdir" \
                   "rm -rf \"$cond_dir/flye\" && $runner flye --threads $REALVAL_THREADS --out-dir \"$cond_dir/flye\" --genome-size \"${gsize_mb}m\" $( [ \"$long_read_type\" = \"nanopore\" ] && echo \"--nano-raw\" || echo \"--pacbio-raw\" ) \"$reads_long\" > \"$log_file\" 2> \"$err_file\" && ln -sf \"$cond_dir/flye/assembly.fasta\" \"$cond_dir/flye/contigs.fasta\"" \
                >> "$commands_file"
        elif [ "$actual_assembler" = "canu" ]; then
            asm_subdir="canu"
            local gsize_mb=5.0
            local gsize_line file
            file="$DATA_DIR/manifest_full.csv"
            [ -f "$file" ] || file="$DATA_DIR/manifest.csv"
            if [ -f "$file" ]; then
                gsize_line=$(grep -F "$organism" "$file" 2>/dev/null | head -1)
                gsize_mb=$(echo "$gsize_line" | cut -d, -f2 | tr -d '" ' || echo "5.0")
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$condition_id" "$organism" "$coverage" "$contamination" \
                   "$reads_long" "" "$asm_subdir" \
                   "rm -rf \"$cond_dir/canu\" && $runner canu -p asm -d \"$cond_dir/canu\" genomeSize=\"${gsize_mb}m\" $( [ \"$long_read_type\" = \"nanopore\" ] && echo \"-nanopore-raw\" || echo \"-pacbio-raw\" ) \"$reads_long\" useGrid=false > \"$log_file\" 2> \"$err_file\" && ln -sf \"$cond_dir/canu/asm.contigs.fasta\" \"$cond_dir/canu/contigs.fasta\"" \
                >> "$commands_file"
        elif [ "$actual_assembler" = "unicycler" ]; then
            asm_subdir="unicycler"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$condition_id" "$organism" "$coverage" "$contamination" \
                   "$reads_pe1" "$reads_pe2" "$asm_subdir" \
                   "rm -rf \"$cond_dir/unicycler\" && $runner unicycler -t $REALVAL_THREADS -o \"$cond_dir/unicycler\" -1 \"$reads_pe1\" -2 \"$reads_pe2\" -l \"$reads_long\" > \"$log_file\" 2> \"$err_file\" && ln -sf \"$cond_dir/unicycler/assembly.fasta\" \"$cond_dir/unicycler/contigs.fasta\"" \
                >> "$commands_file"
        else
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$condition_id" "$organism" "$coverage" "$contamination" \
                   "$reads_pe1" "$reads_pe2" "$asm_subdir" \
                   "rm -rf \"$cond_dir/spades\" && $runner spades.py --memory 16 --only-assembler -t $REALVAL_THREADS -o \"$cond_dir/spades\" --pe1-1 \"$reads_pe1\" --pe1-2 \"$reads_pe2\" -k \"$kmers\" --phred-offset 33 > \"$log_file\" 2> \"$err_file\"" \
                >> "$commands_file"
        fi
    done < "$conditions_csv"

    echo "[assemble] $total total, $pending pending, $skip already done"

    if [ "$pending" -eq 0 ]; then
        echo "[assemble] Nothing to do."
        mark_done "$stage"
        elapsed "$stage" "$S"
        return 0
    fi

    echo "[assemble] Launching $pending assemblies ($REALVAL_JOBS parallel)..."

    local launched=0 success=0 failed=0
    local fail_file="$STATE_DIR/assemble_failed.txt"
    > "$fail_file"

    while IFS=$'\t' read -r condition_id organism coverage contamination reads_pe1 reads_pe2 asm_subdir cmd; do
        while [ "$(jobs -r | wc -l)" -ge "$REALVAL_JOBS" ]; do
            sleep 10
        done
        launched=$((launched + 1))
        local log_file="$assembly_dir/${condition_id}/assembly.log"
        local err_file="$assembly_dir/${condition_id}/assembly.err.log"
        local contig_path="$assembly_dir/${condition_id}/${asm_subdir}/contigs.fasta"
        (
            if eval "$cmd"; then
                if [ -f "$contig_path" ] && [ -s "$contig_path" ]; then
                    echo "OK  $condition_id ($organism, ${coverage}x/${contamination})" >> "$STATE_DIR/assemble_results.txt"
                else
                    echo "WARN $condition_id ($organism, ${coverage}x/${contamination}): command exited 0 but no contigs" >> "$STATE_DIR/assemble_results.txt"
                    echo "WARN $condition_id ($organism, ${coverage}x/${contamination}): no contigs produced" >> "$error_log"
                fi
            else
                local rc=$?
                echo "FAIL $condition_id ($organism, ${coverage}x/${contamination}) exit=$rc" >> "$STATE_DIR/assemble_results.txt"
                echo "FAIL $condition_id ($organism, ${coverage}x/${contamination}) exit=$rc" >> "$error_log"
                echo "$condition_id" >> "$fail_file"
                if [ -f "$err_file" ] && [ -s "$err_file" ]; then
                    echo "--- stderr for $condition_id ---" >> "$error_log"
                    tail -20 "$err_file" >> "$error_log"
                    echo "" >> "$error_log"
                fi
            fi
        ) &
        echo "[assemble] [$launched/$pending] $condition_id ($organism, ${coverage}x/${contamination})"
    done < "$commands_file"

    wait

    if [ -f "$STATE_DIR/assemble_results.txt" ]; then
        success=$(grep -c '^OK ' "$STATE_DIR/assemble_results.txt" 2>/dev/null || echo 0)
        failed=$(cat "$fail_file" 2>/dev/null | wc -l)
        failed=${failed:-0}
    fi
    echo "[assemble] $success succeeded, $failed failed"

    if [ "$failed" -gt 0 ]; then
        echo "[assemble] Errors logged to $error_log"
        echo "[assemble] Failed conditions:"
        cat "$fail_file"
    fi

    mark_done "$stage"
    elapsed "$stage" "$S"
}

do_qc() {
    local S stage="qc"
    echo "━━━ Stage: qc ━━━"
    S=$(date +%s)

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

    local quast_commands="$STATE_DIR/quast_commands.txt"
    local busco_commands="$STATE_DIR/busco_commands.txt"
    > "$quast_commands"
    > "$busco_commands"

    local line_num=0
    while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reads_long long_read_type reference label_qc label_remediation label_tool assembler_config; do
        line_num=$((line_num + 1))
        [ $line_num -eq 1 ] && continue

        local assembly_fasta=""
        for try_subdir in spades megahit flye canu unicycler; do
            local try_path="$assembly_dir/${condition_id}/${try_subdir}/contigs.fasta"
            if [ -f "$try_path" ] && [ -s "$try_path" ]; then
                assembly_fasta="$try_path"
                break
            fi
        done
        local qc_cond_dir="$qc_dir/${condition_id}"
        mkdir -p "$qc_cond_dir"

        if [ ! -f "$assembly_fasta" ]; then
            echo "[qc] WARNING: Assembly not found at $assembly_fasta, skipping $condition_id"
            continue
        fi

        local quast_out="$qc_cond_dir/quast"
        if [ ! -f "$quast_out/report.tsv" ]; then
            echo "conda_run quast.py -o \"$quast_out\" -R \"$reference\" --gene-finding -t $REALVAL_THREADS \"$assembly_fasta\" 2>&1 | tail -3" >> "$quast_commands"
        fi

        local busco_out="$qc_cond_dir/busco"
        if [ ! -d "$busco_out" ]; then
            echo "conda_run busco -i \"$assembly_fasta\" -o \"$busco_out\" -m genome -l bacteria_odb10 -c $REALVAL_THREADS --offline 2>&1 | tail -5" >> "$busco_commands"
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

    echo "[qc] Attempting CheckM..."
    local checkm_data_dir="$DATA_DIR/checkm_data"
    if [ -f "$checkm_data_dir/.checkm_db_ready" ]; then
        local checkm_commands="$STATE_DIR/checkm_commands.txt"
        > "$checkm_commands"
        local line_num2=0
    while IFS=, read -r organism condition_id coverage contamination reads_pe1 reads_pe2 reads_long long_read_type reference label_qc label_remediation label_tool assembler_config; do
            line_num2=$((line_num2 + 1))
            [ $line_num2 -eq 1 ] && continue

            local assembly_fasta=""
            for try_subdir in spades megahit flye canu unicycler; do
                local try_path="$assembly_dir/${condition_id}/${try_subdir}/contigs.fasta"
                if [ -f "$try_path" ] && [ -s "$try_path" ]; then
                    assembly_fasta="$try_path"
                    break
                fi
            done
            local checkm_out="$qc_dir/${condition_id}/checkm"
            mkdir -p "$checkm_out"

            if [ -f "$assembly_fasta" ] && [ ! -f "$checkm_out/qa.txt" ]; then
                echo "mkdir -p \"$checkm_out/bins\" && ln -sf \"$assembly_fasta\" \"$checkm_out/bins/contigs.fasta\" && conda_run checkm lineage_wf -t $REALVAL_THREADS -x fasta \"$checkm_out/bins\" \"$checkm_out/wf\" > \"$checkm_out/lineage.log\" 2>&1 && conda_run checkm qa -t $REALVAL_THREADS --tab_table -f \"$checkm_out/qa.txt\" \"$checkm_out/wf/lineage.ms\" \"$checkm_out/wf\" >> \"$checkm_out/lineage.log\" 2>&1" >> "$checkm_commands"
            fi
        done < "$conditions_csv"

        if [ -s "$checkm_commands" ]; then
            local n_checkm
            n_checkm=$(wc -l < "$checkm_commands")
            local checkm_jobs=$(( REALVAL_JOBS / 2 ))
            [ "$checkm_jobs" -lt 1 ] && checkm_jobs=1
            echo "  $n_checkm CheckM jobs ($checkm_jobs concurrent)"
            while IFS= read -r cmd; do
                while [ "$(jobs -r | wc -l)" -ge "$checkm_jobs" ]; do sleep 10; done
                (eval "$cmd") &
            done < "$checkm_commands"
            wait
        fi
    else
        echo "  CheckM data not available. Skipping CheckM."
    fi

    mark_done "$stage"
    elapsed "$stage" "$S"
}

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

    if [ -f "$DATA_DIR/test.jsonl" ]; then
        local n_questions
        n_questions=$(wc -l < "$DATA_DIR/test.jsonl")
        echo "[render] Generated $n_questions GAQA-format questions"

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
        sub = rec.get('subject', 'unknown')
        subjects[sub] += 1
        out = rec.get('output', '')
        conds[sub] += 1
print('By subject:', dict(subjects))
" 2>&1

        echo "[render] Validating label consistency..."
        python3 -c "
import json
errors = 0
with open('$DATA_DIR/test.jsonl') as f:
    for line in f:
        rec = json.loads(line)
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

do_eval() {
    local S stage="eval"
    echo "━━━ Stage: eval ━━━"
    S=$(date +%s)

    if [ -z "$ADAPTER_DIR" ]; then
        echo "[eval] ADAPTER_DIR not set — skipping eval stage."
        echo "  To run later:"
        echo "    ADAPTER_DIR=models/your-adapter BITNET_LORA_R=8 BITNET_LORA_ALPHA=16 \\"
        echo "      bash scripts/realval_pipeline.sh --stage eval"
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

    export DATA_PROCESSED_DIR="$DATA_DIR"
    export BITNET_LORA_R="$BITNET_LORA_R"
    export BITNET_LORA_ALPHA="$BITNET_LORA_ALPHA"

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

    if [ -f "$DATA_DIR/test.jsonl" ]; then
        cp "$DATA_DIR/test.jsonl" "$RESULTS_DIR/eval_${adapter_name}_test.jsonl"
    fi

    echo "[eval] Results written to $eval_output"
    mark_done "$stage"
    elapsed "$stage" "$S"
}

MAIN_START=$(date +%s)
for stage in "${RUN_STAGES[@]}"; do
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
        *)
            echo "Unknown stage: $stage"
            echo "Valid stages: setup, download, subsample, assemble, qc, render, eval"
            exit 1
            ;;
    esac

    if [ $? -ne 0 ]; then
        echo "WARNING: Stage '$stage' encountered errors. Continuing with next stage."
        echo "  Check logs above for details."
    fi
done

elapsed "Total pipeline" "$MAIN_START"
echo "=== Pipeline complete ==="
