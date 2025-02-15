import csv
import matplotlib.pyplot as plt
from owlready2 import get_ontology
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# Metrics for evaluation
def calculate_bleu(prediction, target):
    reference = [target.split()]
    candidate = prediction.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)


def calculate_rouge(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(target, prediction)


def calculate_coherence(context, response):
    vectorizer = TfidfVectorizer().fit([context, response])
    vectors = vectorizer.transform([context, response])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


def calculate_hrr_old(baseline_outputs, validated_outputs):
    hallucinations_reduced = sum(
        1 for base, valid in zip(baseline_outputs, validated_outputs)
        if base.get("is_hallucination") and not valid.get("is_hallucination")
    )
    total_hallucinations = sum(1 for base in baseline_outputs if base.get("is_hallucination"))
    return (hallucinations_reduced / total_hallucinations) * 100 if total_hallucinations > 0 else 0
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
    return round(hrr, 2)  # Ensure consistent rounding for better readability



def calculate_lcs_old(responses):
    consistent_responses = sum(1 for response in responses if response.get("is_consistent"))
    return (consistent_responses / len(responses)) * 100
def calculate_lcs(responses):
    """
    Calculates Logical Consistency Score (LCS).

    Parameters:
    - responses: List of model outputs with a flag for logical consistency.

    Returns:
    - LCS percentage.
    """
    if not responses:
        return 0  # Avoid division by zero

    consistent_responses = sum(1 for response in responses if response.get("is_consistent", False))
    lcs = (consistent_responses / len(responses)) * 100
    return round(lcs, 2)



def calculate_ra_old(gold_standard, model_outputs):
    correct_responses = sum(
        1 for gold, output in zip(gold_standard, model_outputs)
        if gold["correct_answer"] == output["output"]
    )
    return (correct_responses / len(gold_standard)) * 100
def calculate_ra(gold_standard, model_outputs):
    """
    Calculates Response Accuracy (RA).

    Parameters:
    - gold_standard: List of dictionaries containing correct answers.
    - model_outputs: List of dictionaries containing model's outputs.

    Returns:
    - RA percentage.
    """
    if not gold_standard or len(gold_standard) != len(model_outputs):
        raise ValueError("Gold standard and model outputs must be of equal length.")

    correct_responses = sum(
        1 for gold, output in zip(gold_standard, model_outputs)
        if gold.get("correct_answer") == output.get("output")
    )
    ra = (correct_responses / len(gold_standard)) * 100
    return round(ra, 2)


def evaluate_knowledge_retention(questions, correct_answers, model_outputs):
    retained = sum(1 for q, a, o in zip(questions, correct_answers, model_outputs) if a == o)
    return retained / len(questions) * 100


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
