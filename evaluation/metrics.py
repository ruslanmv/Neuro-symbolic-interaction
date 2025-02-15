import csv
import matplotlib.pyplot as plt
from owlready2 import get_ontology
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------
# 1. BLEU: Perfect match => BLEU close to 1.0
# ---------------------------------------------------------------------
def calculate_bleu(prediction, target):
    reference = [target.split()]
    candidate = prediction.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)


# ---------------------------------------------------------------------
# 2. ROUGE: We compare ROUGE-1 (unigrams) and ROUGE-L (longest common subsequence).
# ---------------------------------------------------------------------
def calculate_rouge(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(target, prediction)


# ---------------------------------------------------------------------
# 3. Coherence: Simple TF-IDF cosine similarity between context and response.
# ---------------------------------------------------------------------
def calculate_coherence(context, response):
    vectorizer = TfidfVectorizer().fit([context, response])
    vectors = vectorizer.transform([context, response])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


# ---------------------------------------------------------------------
# 4. HRR (Hallucination Reduction Rate).
#    Not strictly tied to text overlap, so this part stays mostly as is.
# ---------------------------------------------------------------------
def calculate_hrr(baseline_outputs, validated_outputs):
    """
    Calculates Hallucination Reduction Rate (HRR).

    Parameters:
      - baseline_outputs: List of baseline model outputs with potential hallucinations.
      - validated_outputs: List of validated outputs indicating fixed hallucinations.

    Returns:
      - HRR percentage.
    """
    hallucinations_reduced = sum(
        1 for base, valid in zip(baseline_outputs, validated_outputs)
        if base.get("is_hallucination") and not valid.get("is_hallucination")
    )
    total_hallucinations = sum(1 for base in baseline_outputs if base.get("is_hallucination"))

    if total_hallucinations == 0:
        return 0  # Avoid division by zero

    hrr = (hallucinations_reduced / total_hallucinations) * 100
    return round(hrr, 2)


# ---------------------------------------------------------------------
# 5. LCS (Longest Common Subsequence) for two strings, normalized by
#    the length of the target (or you can pick another denominator).
# ---------------------------------------------------------------------
def calculate_lcs(prediction, target):
    """
    Calculates the normalized Longest Common Subsequence score
    between a prediction and a target text. Returns a value
    in [0.0, 1.0].

    If the two texts are identical (modulo whitespace/newlines),
    the LCS ratio should be close to 1.0.
    """
    # Basic normalization: remove extra newlines/spaces
    pred_tokens = prediction.replace("\n", " ").split()
    tgt_tokens = target.replace("\n", " ").split()

    # Edge case: if target is empty, define LCS as 0 or short-circuit
    if not tgt_tokens:
        return 0.0

    # Build DP table
    dp = [[0] * (len(tgt_tokens) + 1) for _ in range(len(pred_tokens) + 1)]

    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(tgt_tokens) + 1):
            if pred_tokens[i - 1] == tgt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    length_lcs = dp[len(pred_tokens)][len(tgt_tokens)]
    # Normalize by length of the target
    return length_lcs / len(tgt_tokens)


# ---------------------------------------------------------------------
# 6. RA (Response Accuracy) with simple normalization to handle
#    newlines, or consider partial matches if you prefer.
# ---------------------------------------------------------------------
def calculate_ra(gold_standard, model_outputs):
    """
    Calculates Response Accuracy (RA).

    Parameters:
      - gold_standard: List of dicts containing correct answers
                       with key "correct_answer".
      - model_outputs: List of dicts containing model's outputs
                       with key "output".

    Returns:
      - RA percentage (0 to 100).
    """
    if not gold_standard or len(gold_standard) != len(model_outputs):
        raise ValueError("Gold standard and model outputs must have the same length.")

    correct_responses = 0

    for gold, output in zip(gold_standard, model_outputs):
        # Normalize newlines/spaces
        gold_text = gold.get("correct_answer", "").replace("\n", " ").strip()
        model_text = output.get("output", "").replace("\n", " ").strip()

        # Exact match after normalization
        if gold_text == model_text:
            correct_responses += 1

    ra = (correct_responses / len(gold_standard)) * 100
    return round(ra, 2)


# ---------------------------------------------------------------------
# 7. Knowledge Retention: similarly, allow minor normalization
#    if exact string match is too brittle.
# ---------------------------------------------------------------------
def evaluate_knowledge_retention(questions, correct_answers, model_outputs):
    """
    Evaluates knowledge retention by checking how many outputs
    match the known correct answers (after simple newline normalization).

    Returns a percentage from 0 to 100.
    """
    if len(questions) != len(correct_answers) or len(correct_answers) != len(model_outputs):
        raise ValueError("Questions, correct_answers, and model_outputs must have the same length.")

    total = len(questions)
    retained_count = 0

    for q, a, o in zip(questions, correct_answers, model_outputs):
        # Normalize newlines/spaces
        a_text = a.replace("\n", " ").strip()
        o_text = o.replace("\n", " ").strip()

        if a_text == o_text:
            retained_count += 1

    return (retained_count / total) * 100

def generate_latex_table(df, filename="results.tex"):
    latex_table = """
\\documentclass{article}
\\usepackage{graphicx}
\\usepackage{longtable}
\\usepackage{geometry}
\\geometry{a4paper, margin=1in}
\\title{Experimental Results}
\\author{}
\\date{}

\\begin{document}
\\maketitle
\\section*{Evaluation Metrics Across Examples}
\\begin{longtable}{|c|l|l|l|l|l|l|l|l|}
\\hline
\\textbf{Example ID} & \\textbf{BLEU} & \\textbf{ROUGE-1 F1} & \\textbf{ROUGE-L F1} & \\textbf{Coherence} & \\textbf{HRR} & \\textbf{LCS} & \\textbf{RA} & \\textbf{Knowledge Retention} \\\\ \\hline
"""
    for _, row in df.iterrows():
        latex_table += f"{int(row['Example ID'])} & {row['BLEU Score']:.2f} & {row['ROUGE-1 F1']:.2f} & {row['ROUGE-L F1']:.2f} & {row['Coherence Score']:.2f} & {row['HRR']:.2f} & {row['LCS']:.2f} & {row['RA']:.2f} & {row['Knowledge Retention']:.2f} \\\\ \\hline\n"

    latex_table += """
\\end{longtable}

\\section*{Plots of Evaluation Metrics}

\\begin{figure}[h!]
\\centering
\\includegraphics[width=0.8\\textwidth]{BLEU_Score.pdf}
\\caption{BLEU Score Across Examples}
\\end{figure}

\\begin{figure}[h!]
\\centering
\\includegraphics[width=0.8\\textwidth]{ROUGE-1_F1.pdf}
\\caption{ROUGE-1 F1 Score Across Examples}
\\end{figure}

\\begin{figure}[h!]
\\centering
\\includegraphics[width=0.8\\textwidth]{ROUGE-L_F1.pdf}
\\caption{ROUGE-L F1 Score Across Examples}
\\end{figure}

\\begin{figure}[h!]
\\centering
\\includegraphics[width=0.8\\textwidth]{Coherence_Score.pdf}
\\caption{Coherence Score Across Examples}
\\end{figure}

\\end{document}
"""

    with open(filename, "w") as file:
        file.write(latex_table)
    print(f"LaTeX file saved as {filename}")


def generate_plots(df):
    example_ids = df['Example ID']
    metrics = ["BLEU Score", "ROUGE-1 F1", "ROUGE-L F1", "Coherence Score", "HRR", "LCS", "RA", "Knowledge Retention"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(example_ids, df[metric], marker='o', linestyle='-', label=metric)
        plt.title(f"{metric} Across Examples", fontsize=14)
        plt.xlabel("Example ID", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric.replace(' ', '_')}.pdf")
        print(f"Plot saved as {metric.replace(' ', '_')}.pdf")
