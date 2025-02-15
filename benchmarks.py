import pandas as pd
import matplotlib.pyplot as plt
from evaluation.metrics import (
    calculate_bleu, calculate_rouge, calculate_coherence, calculate_hrr,
    calculate_lcs, calculate_ra, evaluate_knowledge_retention
)
from backend import handle_evaluation, main
from evaluation.data import evaluation_data

# Evaluate the model for a given ontology usage flag
def evaluate_model_old(ontology_path, statements, use_ontology):
    onto, model, vectorizer = main(ontology_path)

    results = []
    baseline_outputs = [{"query": s, "is_hallucination": True} for s in statements]
    validated_outputs = [{"query": s, "is_hallucination": False} for s in statements]

    for i, statement in enumerate(statements):
        print(f"Evaluating statement {i+1}: {statement} (Ontology: {'Enabled' if use_ontology else 'Disabled'})")
        response = handle_evaluation(statement, use_ontology, ontology_path)
        response_dict = eval(response)  # Convert JSON string to dictionary

        query = statement
        output = response_dict.get('statements', '') or "No response generated."
        gold_answer = "Correct answer based on ontology"  # Replace with expected answer logic

        # Debug: Print evaluation inputs
        print(f"Query: {query}")
        print(f"Model Output: {output}")
        print(f"Gold Answer: {gold_answer}\n")

        # Calculate metrics with debugging prints
        bleu_score = calculate_bleu(output, gold_answer)
        print(f"BLEU Score: {bleu_score}")

        rouge_scores = calculate_rouge(output, gold_answer)
        print(f"ROUGE Scores: {rouge_scores}")

        coherence_score = calculate_coherence(query, output)
        print(f"Coherence Score: {coherence_score}")

        hrr_score = calculate_hrr(baseline_outputs, validated_outputs)
        print(f"HRR Score: {hrr_score}")

        lcs_score = calculate_lcs(validated_outputs)
        print(f"LCS Score: {lcs_score}")

        ra_score = calculate_ra([{"correct_answer": gold_answer}], [{"output": output}])
        print(f"RA Score: {ra_score}")

        knowledge_retention_score = evaluate_knowledge_retention(
            [query], [gold_answer], [output]
        )
        print(f"Knowledge Retention Score: {knowledge_retention_score}\n")

        results.append({
            "Example ID": f"{use_ontology}-{i + 1}",  # Add unique prefix for ontology usage
            "Query": query,
            "Gold Answer": gold_answer,
            "Model Output": output,
            "Use Ontology": use_ontology,
            "BLEU Score": bleu_score,
            "ROUGE-1 F1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-L F1": rouge_scores['rougeL'].fmeasure,
            "Coherence Score": coherence_score,
            "HRR": hrr_score,
            "LCS": lcs_score,
            "RA": ra_score,
            "Knowledge Retention": knowledge_retention_score,
        })

    return results
import csv
import json
import os
def log_evaluation_results_old(logs_path, question, statement, knowledge_base, label, ontology):
    """
    Logs evaluation results into a CSV file.

    Parameters:
    - logs_path: Path to the CSV file for logging.
    - question: The original user question.
    - statement: The generated statement or response.
    - knowledge_base: The knowledge base or logical form evaluated.
    - label: Whether the statement is true or false.
    - ontology: Whether ontology was used ('True' or 'False').
    """
    # Define the header for the CSV file
    fieldnames = ['Question', 'Statement', 'Knowledge Base', 'Label', 'Ontology']
    
    # Check if the file exists to avoid overwriting headers
    write_header = not os.path.exists(logs_path)
    
    # Write the results to the CSV file
    with open(logs_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        if write_header:
            writer.writeheader()
        writer.writerow({
            'Question': question,
            'Statement': statement,
            'Knowledge Base': knowledge_base,
            'Label': label,
            'Ontology': ontology
        })

import os
import pandas as pd

def log_evaluation_results(logs_path, question, statement, knowledge_base, label, ontology):
    """
    Logs evaluation results into a pickle file.

    Parameters:
    - logs_path: Path to the pickle file for logging.
    - question: The original user question.
    - statement: The generated statement or response.
    - knowledge_base: The knowledge base or logical form evaluated.
    - label: Whether the statement is true or false.
    - ontology: Whether ontology was used ('True' or 'False').
    """

    # Create a dictionary with the data
    data = {
        'Question': question,
        'Statement': statement,
        'Knowledge Base': knowledge_base,
        'Label': label,
        'Ontology': ontology
    }

    # Load existing DataFrame if the file exists, otherwise create a new one
    if os.path.exists(logs_path):
        df = pd.read_pickle(logs_path)
    else:
        df = pd.DataFrame(columns=data.keys())

    # Append the new data to the DataFrame
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    # Save the updated DataFrame to the pickle file
    df.to_pickle(logs_path)

def evaluate_model_new1(ontology_path, statements, use_ontology):
    """
    Evaluates the model performance based on the ontology usage.

    Parameters:
    - ontology_path: Path to the ontology file.
    - statements: List of input queries/statements.
    - use_ontology: Boolean flag to toggle ontology usage.

    Returns:
    - List of results with evaluation metrics.
    """
    onto, model, vectorizer = main(ontology_path)

    results = []
    baseline_outputs = [{"query": s, "is_hallucination": True} for s in statements]
    validated_outputs = [{"query": s, "is_hallucination": False} for s in statements]

    for i, statement in enumerate(statements):
        response = handle_evaluation(statement, use_ontology, ontology_path)
        response_dict = eval(response)  # Convert JSON string to dictionary

        query = statement
        output = response_dict.get('statements', '') or "No response generated."
        gold_answer = "Correct answer based on ontology"  # Replace with actual ontology-based logic

        # Metrics evaluation
        bleu_score = calculate_bleu(output, gold_answer)
        rouge_scores = calculate_rouge(output, gold_answer)
        coherence_score = calculate_coherence(query, output)
        hrr_score = calculate_hrr(baseline_outputs, validated_outputs)
        lcs_score = calculate_lcs(validated_outputs)
        ra_score = calculate_ra([{"correct_answer": gold_answer}], [{"output": output}])
        knowledge_retention_score = evaluate_knowledge_retention([query], [gold_answer], [output])

        # Append results
        results.append({
            "Example ID": f"{use_ontology}-{i + 1}",
            "Query": query,
            "Gold Answer": gold_answer,
            "Model Output": output,
            "Use Ontology": use_ontology,
            "BLEU Score": round(bleu_score, 2),
            "ROUGE-1 F1": round(rouge_scores['rouge1'].fmeasure, 2),
            "ROUGE-L F1": round(rouge_scores['rougeL'].fmeasure, 2),
            "Coherence Score": round(coherence_score, 2),
            "HRR": hrr_score,
            "LCS": lcs_score,
            "RA": ra_score,
            "Knowledge Retention": round(knowledge_retention_score, 2),
        })

    return results

def evaluate_model(ontology_path, statements, use_ontology, logs_path="logs.pkl"):
    """
    Evaluates the model performance and logs the results.

    Parameters:
    - ontology_path: Path to the ontology file.
    - statements: List of input queries/statements.
    - use_ontology: Boolean flag to toggle ontology usage.
    - logs_path: Path to the logs file (default: "logs.csv").
    """
    onto, model, vectorizer = main(ontology_path)

    results = []
    baseline_outputs = [{"query": s, "is_hallucination": True} for s in statements]
    validated_outputs = [{"query": s, "is_hallucination": False} for s in statements]

    for i, statement in enumerate(statements):
        print(f"Evaluating statement {i+1}: {statement} (Ontology: {'Enabled' if use_ontology else 'Disabled'})")
        response = handle_evaluation(statement, use_ontology, ontology_path)
        
        #print("response:",response)
        response_dict = json.loads(response)  # Parse JSON string into a dictionary

        query = statement
        output = response_dict.get('statements', '') or "No response generated."
        gold_answer = "Correct answer based on ontology"  # Replace with actual expected answer logic

        # Calculate label (true/false) based on logical form evaluation
        label = 'True' if response_dict.get('statements') else 'False'

        # Log the evaluation details
        log_evaluation_results(
            logs_path=logs_path,
            question=query,
            statement=output,
            knowledge_base=response_dict.get('details', 'N/A'),
            label=label,
            ontology=str(use_ontology)
        )

        # Metrics evaluation
        bleu_score = calculate_bleu(output, gold_answer)
        rouge_scores = calculate_rouge(output, gold_answer)
        coherence_score = calculate_coherence(query, output)
        hrr_score = calculate_hrr(baseline_outputs, validated_outputs)
        lcs_score = calculate_lcs(validated_outputs)
        ra_score = calculate_ra([{"correct_answer": gold_answer}], [{"output": output}])
        knowledge_retention_score = evaluate_knowledge_retention([query], [gold_answer], [output])

        # Append results
        results.append({
            "Example ID": f"{use_ontology}-{i + 1}",
            "Query": query,
            "Gold Answer": gold_answer,
            "Model Output": output,
            "Use Ontology": use_ontology,
            "BLEU Score": round(bleu_score, 2),
            "ROUGE-1 F1": round(rouge_scores['rouge1'].fmeasure, 2),
            "ROUGE-L F1": round(rouge_scores['rougeL'].fmeasure, 2),
            "Coherence Score": round(coherence_score, 2),
            "HRR": hrr_score,
            "LCS": lcs_score,
            "RA": ra_score,
            "Knowledge Retention": round(knowledge_retention_score, 2),
        })

    return results

# Save results and generate reports
def save_results(results, output_csv="evaluation_results.csv", latex_file="results.tex"):
    df = pd.DataFrame(results)
    # Debug: Print DataFrame before saving
    print("Results DataFrame:\n", df.head(), "\n")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

import csv

# Save the training data to a CSV file
with open('testing.csv', 'w', newline='') as csvfile:
    fieldnames = ['Question', 'Statement', 'Knowledge Base', 'Label']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    writer.writerows(evaluation_data)
if __name__ == "__main__":
    ontology_path = "engine_ontology.owl"

    # Read questions from CSV
    with open("testing.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        statements = [row[0] for row in reader]

    # Evaluate with and without ontology
    results_with_ontology = evaluate_model(ontology_path, statements, use_ontology=True)
    results_without_ontology = evaluate_model(ontology_path, statements, use_ontology=False)

    # Merge and save results
    all_results = results_with_ontology + results_without_ontology
    #save_results(all_results)