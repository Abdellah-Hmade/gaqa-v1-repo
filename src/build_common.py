QC_DIAGNOSIS_TEMPLATES = ['Below is a genome assembly QC report:\n{metrics}\n\nWhat is the most likely problem with this assembly?', 'A draft genome was assembled and the following quality metrics were obtained:\n{metrics}\n\nBased on these metrics, what is the primary issue with the assembly?', 'Assembly evaluation report:\n{metrics}\n\nIdentify the most critical problem with this genome assembly.', 'Review the following assembly statistics:\n{metrics}\n\nWhich of the following best describes the assembly quality?', '=== GENOME ASSEMBLY QC SUMMARY ===\n{metrics}\n\nDIAGNOSIS: What is the single most important issue affecting this assembly?', "Quality assessment for draft genome:\n{metrics}\n\nChoose the option that best captures the assembly's main shortcoming.", 'Assembly Statistics Report\n-----------------------------\n{metrics}\n\nWhat is the predominant quality issue indicated by these metrics?', '[QC Report]\n{metrics}\n\nBased on the above report, which problem is most critical?', "Genome Assembly Evaluation\n{metrics}\n\nWhat is the best diagnosis of this assembly's quality?", 'Below are quality metrics from a newly assembled genome:\n{metrics}\n\nWhat is the primary concern with this assembly?', "Draft genome QC results:\n{metrics}\n\nSelect the most accurate description of this assembly's quality."]

REMEDIATION_TEMPLATES = ['Given the following assembly QC report:\n{metrics}\n\nWhat is the best next action?', 'A genome assembly project produced these results:\n{metrics}\n\nWhat step should be taken next?', 'After reviewing the following assembly metrics:\n{metrics}\n\nWhich course of action is most appropriate?', 'Based on the assembly evaluation below:\n{metrics}\n\nWhat should the researcher do next?', '=== REMEDIATION ANALYSIS ===\n{metrics}\n\nRECOMMENDATION: What is the most appropriate next step?', 'Assembly project review:\n{metrics}\n\nWhat action should the bioinformatics team take?', 'Post-assembly QC Metrics:\n{metrics}\n\nGiven these results, what is the single best course of action?', '[Assembly Assessment]\n{metrics}\n\nWhich remediation strategy do you recommend?', 'Genome assembly quality report:\n{metrics}\n\nWhat is the recommended next step for this project?', 'Below are the QC statistics for a completed genome assembly:\n{metrics}\n\nWhat should the researcher do next to improve this assembly?', 'Evaluation of assembly completeness and accuracy:\n{metrics}\n\nWhich action is most warranted given these metrics?']

TOOL_SELECTION_TEMPLATES = ['A researcher has the following sequencing data:\n{stats}\n\nWhich assembler would be most appropriate?', 'Given these sample and sequencing parameters:\n{stats}\n\nSelect the best assembly strategy.', 'Based on the following sequencing project details:\n{stats}\n\nWhat is the optimal assembler choice?', 'A genomics lab is planning an assembly with the following specifications:\n{stats}\n\nWhat tool or parameter should they use?', '=== ASSEMBLY STRATEGY SELECTION ===\n{stats}\n\nWhich assembly approach is most suitable for this dataset?', 'Sequencing project summary:\n{stats}\n\nChoose the best assembler for this scenario.', 'Experimental Design:\n{stats}\n\nWhat is the recommended assembly tool for this project?', '[Tool Selection]\n{stats}\n\nWhich assembler should be selected given these parameters?', 'A bioinformatics core receives a new sequencing dataset:\n{stats}\n\nWhat assembly pipeline should they recommend?', 'Given the following sequencing specifications:\n{stats}\n\nWhat is the most appropriate assembly method?', 'Planning a genome assembly experiment:\n{stats}\n\nWhich tool or parameter set should be used?']

FAILURE_MODES = {'low_coverage': {'description': 'Low sequencing coverage / shallow depth', 'remediation': 'Sequence deeper, e.g., add more sequencing or long reads to span repeats'}, 'contamination': {'description': 'Contamination (foreign DNA in the assembly)', 'remediation': 'Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble'}, 'misassembly': {'description': 'Misassembly errors (structural errors in the assembly)', 'remediation': 'Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality'}, 'acceptable': {'description': 'Acceptable quality assembly (no major issues)', 'remediation': 'Proceed to downstream annotation'}}

def generate_tool_scenarios(rng):
    scenarios = []
    for _ in range(20):
        genome_size = rng.uniform(0.5, 12.0)
        depth = rng.uniform(30, 200)
        read_len = rng.choice([150, 250, 300])
        tech = rng.choice(['Illumina NovaSeq', 'Illumina HiSeq', 'Illumina MiSeq'])
        stats_text = f'Genome size: {genome_size:.1f} Mb\nSequencing technology: {tech} (2×{read_len} bp, paired-end)\nSequencing depth: {depth:.0f}×\nSample type: pure isolate'
        correct = 'SPAdes (or Unicycler for bacterial isolates)'
        distractors = ['Flye (long-read assembler)', 'MEGAHIT (metagenome assembler)', 'Hifiasm (PacBio HiFi assembler)']
        scenarios.append((stats_text, correct, distractors))
    for _ in range(15):
        genome_size = rng.uniform(0.5, 100.0)
        depth = rng.uniform(15, 100)
        read_len = rng.uniform(5000, 30000)
        tech = rng.choice(['ONT MinION', 'ONT GridION', 'ONT PromethION'])
        stats_text = f'Genome size: {genome_size:.1f} Mb\nSequencing technology: {tech} (long reads, mean length {read_len:.0f} bp)\nSequencing depth: {depth:.0f}×\nSample type: pure isolate'
        correct = 'Flye (long-read assembler for ONT data)'
        distractors = ['SPAdes (short-read assembler)', 'MEGAHIT (metagenome assembler)', 'ABySS (k-mer based short-read assembler)']
        scenarios.append((stats_text, correct, distractors))
    for _ in range(10):
        genome_size = rng.uniform(0.5, 50.0)
        depth = rng.uniform(15, 60)
        read_len = rng.uniform(10000, 20000)
        stats_text = f'Genome size: {genome_size:.1f} Mb\nSequencing technology: PacBio HiFi (CCS reads, mean length {read_len:.0f} bp)\nSequencing depth: {depth:.0f}×\nSample type: pure isolate'
        correct = 'Hifiasm (HiFi assembler)'
        distractors = ['Flye (noisy long-read assembler)', 'SPAdes (short-read assembler)', 'Unicycler (hybrid assembler)']
        scenarios.append((stats_text, correct, distractors))
    for _ in range(10):
        genome_size = rng.uniform(0.5, 20.0)
        short_depth = rng.uniform(30, 150)
        long_depth = rng.uniform(10, 50)
        long_len = rng.uniform(5000, 20000)
        stats_text = f'Genome size: {genome_size:.1f} Mb\nShort reads: Illumina (2×150 bp, {short_depth:.0f}×)\nLong reads: ONT (mean {long_len:.0f} bp, {long_depth:.0f}×)\nSample type: pure isolate'
        correct = 'Unicycler (hybrid mode) or hybridSPAdes'
        distractors = ['Flye (long-read only)', 'SPAdes (short-read only)', 'MEGAHIT (metagenome short-read)']
        scenarios.append((stats_text, correct, distractors))
    for _ in range(10):
        depth = rng.uniform(30, 200)
        read_len = rng.choice([150, 250])
        num_samples_desc = rng.choice(['', ' (complex community)', ' (gut microbiome)', ' (soil sample)'])
        stats_text = f'Sequencing technology: Illumina (2×{read_len} bp, paired-end)\nSequencing depth: {depth:.0f}×\nSample type: metagenome{num_samples_desc}'
        correct = 'MEGAHIT or metaSPAdes (metagenome assembler)'
        distractors = ['SPAdes (isolate assembler)', 'Flye (long-read assembler)', 'Hifiasm (HiFi assembler)']
        scenarios.append((stats_text, correct, distractors))
    for _ in range(15):
        read_len = rng.choice([150, 250, 300])
        depth = rng.uniform(8, 80)
        genome_size = rng.uniform(0.5, 12.0)
        is_low_cov = depth < 25
        if is_low_cov:
            k_vals = range(21, min(read_len - 1, 41), 2)
            k_mer = rng.choice(list(k_vals)) if k_vals else 21
            correct = f'k = {k_mer} (shorter k-mer for low coverage)'
            distractors = [f'k = {k_mer + 20} (too long for low coverage)', f'k = {k_mer + 50} (too long, > read length)', 'Use a single k-mer of 99']
        else:
            correct = 'Multiple k-mer values (e.g., 21,33,55,77) with SPAdes'
            distractor_ks = [f'k = {rng.choice([21, 33, 55, 77])}', 'k = 21 only', 'k = 99 only']
            rng.shuffle(distractor_ks)
            distractors = distractor_ks[:3]
        stats_text = f'Sequencing technology: Illumina (2×{read_len} bp, paired-end)\nSequencing depth: {depth:.0f}×\nGenome size: {genome_size:.1f} Mb\nSample type: pure isolate'
        scenarios.append((stats_text, correct, distractors))
    return scenarios

def make_question_output(correct_letter, correct_answer_text):
    return f'The correct answer is {correct_letter}: {correct_answer_text}'

def assign_letters(shuffled_options, correct_text):
    letters = ['A', 'B', 'C', 'D']
    for (letter, opt) in zip(letters, shuffled_options):
        if opt == correct_text:
            return letter
    return 'A'

def make_input_text(subject, question_text, options):
    lines = [f'Subject: {subject}', f'Question: {question_text}', 'Options:']
    for (letter, opt) in zip(['A', 'B', 'C', 'D'], options):
        lines.append(f'{letter}) {opt}')
    return '\n'.join(lines)
