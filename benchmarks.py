import pandas as pd
import matplotlib.pyplot as plt
from evaluation.metrics import (
    calculate_bleu, calculate_rouge, calculate_coherence, calculate_hrr,
    calculate_lcs, calculate_ra, evaluate_knowledge_retention
)
from backend import handle_evaluation, main


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

evaluation_data = [
    # Custom statements
    ("Does battery_1 cause failure in electric_engine_1?", "battery_1.CausesFailure(electric_engine_1)", True),
    # True statements with variations
    ("What causes piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How does a piston lead to oil engine failure?", "The piston leads to the oil engine failing.", "piston_1.CausesFailure(oil_engine_1)", True),
    ("Why does a piston cause oil engine failure?", "Failure of the oil engine is caused by the piston.", "piston_1.CausesFailure(oil_engine_1)", True),
    ("In what specific ways can a piston cause failure in an oil engine?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What are the signs of a failing piston in an oil engine?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How can the condition of a piston in an oil engine be monitored and assessed?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What factors contribute to the increased risk of piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How does piston design and material choice impact the likelihood of failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What maintenance practices can help prolong the life of pistons in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),

    ("What causes oil pump failure in oil engines?", "Oil pump causes failure of oil engine", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("How does an oil pump lead to oil engine failure?", "The oil pump leads to the oil engine failing.", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("Why does an oil pump cause oil engine failure?", "Failure of the oil engine is caused by the oil pump.", "oil_pump_1.CausesFailure(oil_engine_1)", True),

    ("What causes battery failure in electric engines?", "Battery causes failure of electric engine", "battery_1.CausesFailure(electric_engine_1)", True),
    ("How does a battery lead to electric engine failure?", "The battery leads to the electric engine failing.", "battery_1.CausesFailure(electric_engine_1)", True),
    ("Why does a battery cause electric engine failure?", "Failure of the electric engine is caused by the battery.", "battery_1.CausesFailure(electric_engine_1)", True),

    ("What causes motor failure in electric engines?", "Motor causes failure of electric engine", "motor_1.CausesFailure(electric_engine_1)", True),
    ("How does a motor lead to electric engine failure?", "The motor leads to the electric engine failing.", "motor_1.CausesFailure(electric_engine_1)", True),
    ("Why does a motor cause electric engine failure?", "Failure of the electric engine is caused by the motor.", "motor_1.CausesFailure(electric_engine_1)", True),

    # Additional true statements (repeats for more examples)
    ("What causes piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What causes oil pump failure in oil engines?", "Oil pump causes failure of oil engine", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("What causes battery failure in electric engines?", "Battery causes failure of electric engine", "battery_1.CausesFailure(electric_engine_1)", True),
    ("What causes motor failure in electric engines?", "Motor causes failure of electric engine", "motor_1.CausesFailure(electric_engine_1)", True),

    # False statements
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),
]
evaluation_data_test = [
    # Custom statements
    ("Does battery_1 cause failure in electric_engine_1?", "battery_1.CausesFailure(electric_engine_1)", True),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

 ]
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