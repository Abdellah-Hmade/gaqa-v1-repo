BITNET_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
KNOWLEDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

KNOWLEDGE_DIR = "data/gaqa_knowledge"
KNOWLEDGE_SOURCES_DIR = "data/knowledge_sources"
CHUNK_WORDS = 380
CHUNK_OVERLAP = 60
CHUNK_MIN_WORDS = 80

SEED = 42
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 1024
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_BATCH_SIZE = 16

SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "heldout": "test.jsonl"}

DATA_DIR = "data/realval"
ENV_RUNNER = "micromamba run -n realval"
JOBS = 12
BUSCO_LINEAGE = "bacteria_odb10"
COVERAGES = [10, 30, 100]

BACTERIA = [
    {"key": "ecoli", "name": "Escherichia coli K-12 MG1655", "taxid": 562, "strain": r"MG1655|K-?12", "gcf": "GCF_000005845.2", "size_mb": 4.6, "gc": 50.8, "split": "train", "related": "styphi", "distant": "mtb"},
    {"key": "saureus", "name": "Staphylococcus aureus NCTC 8325", "taxid": 1280, "strain": r"NCTC ?8325|8325", "gcf": "GCF_000013425.1", "size_mb": 2.9, "gc": 32.9, "split": "train", "related": "bsub", "distant": "mtb"},
    {"key": "vcholerae", "name": "Vibrio cholerae N16961", "taxid": 666, "strain": r"N16961", "gcf": "GCF_000006745.1", "size_mb": 4.0, "gc": 47.5, "split": "train", "related": "paeru", "distant": "mtb"},
    {"key": "bsub", "name": "Bacillus subtilis 168", "taxid": 1423, "strain": r"168", "gcf": "GCF_000009045.1", "size_mb": 4.2, "gc": 43.5, "split": "heldout", "related": "saureus", "distant": "mtb"},
    {"key": "mtb", "name": "Mycobacterium tuberculosis H37Rv", "taxid": 1773, "strain": r"H37Rv", "gcf": "GCF_000195955.2", "size_mb": 4.4, "gc": 65.6, "split": "heldout", "related": "none", "distant": "ecoli"},
    {"key": "paeru", "name": "Pseudomonas aeruginosa PAO1", "taxid": 287, "strain": r"PAO1", "gcf": "GCF_000006765.1", "size_mb": 6.3, "gc": 66.6, "split": "train", "related": "vcholerae", "distant": "mtb"},
    {"key": "spneumo", "name": "Streptococcus pneumoniae R6", "taxid": 1313, "strain": r"R6", "gcf": "GCF_000007045.1", "size_mb": 2.0, "gc": 39.7, "split": "train", "related": "efaec", "distant": "mtb"},
    {"key": "lmono", "name": "Listeria monocytogenes EGD-e", "taxid": 1639, "strain": r"EGD-?e", "gcf": "GCF_000196035.1", "size_mb": 2.9, "gc": 38.0, "split": "train", "related": "efaec", "distant": "mtb"},
    {"key": "styphi", "name": "Salmonella Typhimurium LT2", "taxid": 28901, "strain": r"LT2", "gcf": "GCF_000006945.2", "size_mb": 4.9, "gc": 52.2, "split": "train", "related": "ecoli", "distant": "mtb"},
    {"key": "kpneumo", "name": "Klebsiella pneumoniae MGH 78578", "taxid": 573, "strain": r"MGH ?78578|78578", "gcf": "GCF_000016305.1", "size_mb": 5.3, "gc": 57.0, "split": "train", "related": "ecoli", "distant": "mtb"},
    {"key": "nmeningo", "name": "Neisseria meningitidis MC58", "taxid": 487, "strain": r"MC58", "gcf": "GCF_000008805.1", "size_mb": 2.3, "gc": 51.5, "split": "heldout", "related": "paeru", "distant": "mtb"},
    {"key": "efaec", "name": "Enterococcus faecalis V583", "taxid": 1351, "strain": r"V583", "gcf": "GCF_000007785.1", "size_mb": 3.2, "gc": 37.5, "split": "train", "related": "lmono", "distant": "mtb"},
]

MANUAL_RUNS = {"saureus": "DRR1000166", "kpneumo": "SRR15422670", "nmeningo": "SRR23299964"}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}<|eot_id|>\n"
    "{% endif %}"
    "{% endfor %}"
)


MODES = ["low_coverage", "contamination", "misassembly", "acceptable"]

CONDITIONS = [
    ("C01", 10, 0, None, "low_coverage", "default"),
    ("C02", 30, 0, None, "acceptable", "default"),
    ("C03", 100, 0, None, "acceptable", "default"),
    ("C04", 30, 5, "related", "contamination", "default"),
    ("C05", 100, 10, "related", "contamination", "default"),
    ("C06", 100, 10, "distant", "contamination", "default"),
    ("C07", 30, 10, "distant", "contamination", "default"),
    ("C08", 100, 5, "related", "contamination", "default"),
    ("C09", 30, 10, "related", "contamination", "default"),
    ("C10", 100, 0, None, "misassembly", "wrong"),
    ("C11", 100, 0, None, "misassembly", "megahit"),
]

FAILURE_MODES = {
    "low_coverage": {
        "description": "Low sequencing coverage / shallow depth",
        "remediation": "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
    },
    "contamination": {
        "description": "Contamination (foreign DNA in the assembly)",
        "remediation": "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
    },
    "misassembly": {
        "description": "Misassembly errors (structural errors in the assembly)",
        "remediation": "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
    },
    "acceptable": {
        "description": "Acceptable quality assembly (no major issues)",
        "remediation": "Proceed to downstream annotation",
    },
}

METRIC_LABELS = {
    "num_contigs": "Number of contigs",
    "n50": "N50 (bp)",
    "largest_contig": "Largest contig (bp)",
    "total_length": "Total assembly length (bp)",
    "gc_content_pct": "GC content (%)",
    "busco_complete_pct": "BUSCO complete (%)",
    "busco_duplicated_pct": "BUSCO duplicated (%)",
    "busco_single_copy_pct": "BUSCO single-copy (%)",
    "busco_fragmented_pct": "BUSCO fragmented (%)",
    "busco_missing_pct": "BUSCO missing (%)",
    "checkm_completeness_pct": "CheckM completeness (%)",
    "checkm_contamination_pct": "CheckM contamination (%)",
    "num_misassemblies": "Number of misassemblies",
    "mismatches_per_100kbp": "Mismatches per 100 kbp",
    "genome_fraction_pct": "Genome fraction (%)",
    "sequencing_depth": "Sequencing depth (x)",
}

ALTERNATIVE_LABELS = {
    "num_contigs": ["# contigs", "Contig count", "Total contigs", "Contigs (n)"],
    "n50": ["N50 (kb)", "N50 length (bp)", "N50 (kbp)", "Contig N50"],
    "largest_contig": ["Largest contig", "Max contig length (bp)", "Longest contig (bp)"],
    "total_length": ["Assembly size (bp)", "Genome size (bp)", "Total bases assembled"],
    "gc_content_pct": ["GC (%)", "GC ratio (%)"],
    "busco_complete_pct": ["BUSCO complete (%)", "BUSCO completeness (%)"],
    "busco_duplicated_pct": ["BUSCO duplicated (%)", "Duplicated BUSCOs (%)"],
    "checkm_completeness_pct": ["CheckM completeness (%)", "Completeness (CheckM)"],
    "checkm_contamination_pct": ["CheckM contamination (%)", "Contamination (CheckM %)"],
    "num_misassemblies": ["Misassembly count", "# misassemblies"],
    "mismatches_per_100kbp": ["Mismatches/100kbp", "Mismatch rate (per 100 kbp)"],
    "genome_fraction_pct": ["Genome fraction (%)", "Genome coverage (%)"],
    "sequencing_depth": ["Sequencing depth (x)", "Coverage depth (x)", "Read depth (x)"],
}

QC_DIAGNOSIS_TEMPLATES = [
    "Below is a genome assembly QC report:\n{metrics}\n\nWhat is the most likely problem with this assembly?",
    "A draft genome was assembled and the following quality metrics were obtained:\n{metrics}\n\nBased on these metrics, what is the primary issue with the assembly?",
    "Assembly evaluation report:\n{metrics}\n\nIdentify the most critical problem with this genome assembly.",
    "Review the following assembly statistics:\n{metrics}\n\nWhich of the following best describes the assembly quality?",
    "=== GENOME ASSEMBLY QC SUMMARY ===\n{metrics}\n\nDIAGNOSIS: What is the single most important issue affecting this assembly?",
    "Quality assessment for draft genome:\n{metrics}\n\nChoose the option that best captures the assembly's main shortcoming.",
    "Assembly Statistics Report\n-----------------------------\n{metrics}\n\nWhat is the predominant quality issue indicated by these metrics?",
    "[QC Report]\n{metrics}\n\nBased on the above report, which problem is most critical?",
    "Genome Assembly Evaluation\n{metrics}\n\nWhat is the best diagnosis of this assembly's quality?",
    "Below are quality metrics from a newly assembled genome:\n{metrics}\n\nWhat is the primary concern with this assembly?",
    "Draft genome QC results:\n{metrics}\n\nSelect the most accurate description of this assembly's quality.",
]

REMEDIATION_TEMPLATES = [
    "Given the following assembly QC report:\n{metrics}\n\nWhat is the best next action?",
    "A genome assembly project produced these results:\n{metrics}\n\nWhat step should be taken next?",
    "After reviewing the following assembly metrics:\n{metrics}\n\nWhich course of action is most appropriate?",
    "Based on the assembly evaluation below:\n{metrics}\n\nWhat should the researcher do next?",
    "=== REMEDIATION ANALYSIS ===\n{metrics}\n\nRECOMMENDATION: What is the most appropriate next step?",
    "Assembly project review:\n{metrics}\n\nWhat action should the bioinformatics team take?",
    "Post-assembly QC Metrics:\n{metrics}\n\nGiven these results, what is the single best course of action?",
    "[Assembly Assessment]\n{metrics}\n\nWhich remediation strategy do you recommend?",
    "Genome assembly quality report:\n{metrics}\n\nWhat is the recommended next step for this project?",
    "Below are the QC statistics for a completed genome assembly:\n{metrics}\n\nWhat should the researcher do next to improve this assembly?",
    "Evaluation of assembly completeness and accuracy:\n{metrics}\n\nWhich action is most warranted given these metrics?",
]

TOOL_SELECTION_TEMPLATES = [
    "A researcher has the following sequencing data:\n{stats}\n\nWhich assembler would be most appropriate?",
    "Given these sample and sequencing parameters:\n{stats}\n\nSelect the best assembly strategy.",
    "Based on the following sequencing project details:\n{stats}\n\nWhat is the optimal assembler choice?",
    "A genomics lab is planning an assembly with the following specifications:\n{stats}\n\nWhat tool or parameter should they use?",
    "=== ASSEMBLY STRATEGY SELECTION ===\n{stats}\n\nWhich assembly approach is most suitable for this dataset?",
    "Sequencing project summary:\n{stats}\n\nChoose the best assembler for this scenario.",
    "Experimental Design:\n{stats}\n\nWhat is the recommended assembly tool for this project?",
    "[Tool Selection]\n{stats}\n\nWhich assembler should be selected given these parameters?",
    "A bioinformatics core receives a new sequencing dataset:\n{stats}\n\nWhat assembly pipeline should they recommend?",
    "Given the following sequencing specifications:\n{stats}\n\nWhat is the most appropriate assembly method?",
    "Planning a genome assembly experiment:\n{stats}\n\nWhich tool or parameter set should be used?",
]

CONTINUOUS_KEYS = {
    "n50", "largest_contig", "total_length", "num_contigs", "gc_content_pct",
    "busco_complete_pct", "busco_duplicated_pct", "busco_fragmented_pct",
    "busco_missing_pct", "checkm_completeness_pct", "checkm_contamination_pct",
    "genome_fraction_pct", "num_misassemblies", "mismatches_per_100kbp",
    "sequencing_depth",
}

NOISE_METRICS = [
    ("Read N50 (bp)", lambda rng: f"{rng.randint(2000, 50000)}"),
    ("Mean PHRED score", lambda rng: f"{rng.uniform(30, 40):.1f}"),
    ("Duplication rate (%)", lambda rng: f"{rng.uniform(5, 60):.1f}"),
    ("Mean insert size (bp)", lambda rng: f"{rng.randint(200, 800)}"),
    ("GC dropout (%)", lambda rng: f"{rng.uniform(0, 15):.1f}"),
    ("Read mapping rate (%)", lambda rng: f"{rng.uniform(70, 99.5):.1f}"),
]
