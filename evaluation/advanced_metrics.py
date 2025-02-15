"""
advanced_metrics.py

Provides advanced evaluation metrics for neurosymbolic or ontology-based systems:
1) Factual Accuracy
2) Consistency Score
3) Ontology Alignment Score
4) Error Detection and Correction Rate
5) Human Evaluation Metrics (Qualitative Ratings + Inter-Rater Agreement)
6) Efficiency and Latency

Author: Your Name
Date: Your-MM-DD
"""

import time
import statistics
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------------
# 5.1 Factual Accuracy
# ------------------------------------------------------------------------
def calculate_factual_accuracy(model_output: str, gold_answer: str) -> float:
    """
    Calculates the percentage (0-100) of correct factual statements
    within a model's output by comparing against a known or "gold" answer.

    Parameters:
    - model_output: The predicted text from the model (string).
    - gold_answer:  The ground-truth reference text or facts (string).

    Returns:
    - factual_accuracy (float): A score in [0, 100], representing the
      percentage of factually correct statements.

    Note:
    - The simplest approach might be to do an exact or partial string match
      on specific "fact" segments. A more advanced approach would parse or
      extract factual triples from the model output and compare to a knowledge base.
    """
    # --- Placeholder / illustrative approach ---
    # Split each into tokens or lines representing "facts"
    gold_facts = [fact.strip() for fact in str(gold_answer).split('\n') if fact.strip()]
    output_facts = [fact.strip() for fact in str(model_output).split('\n') if fact.strip()]

    # Count how many facts in the model output also appear in the gold facts
    correct_count = sum(1 for fact in output_facts if fact in gold_facts)

    # Avoid division by zero if output_facts is empty
    if not output_facts:
        return 0.0

    factual_accuracy = (correct_count / len(output_facts)) * 100
    return factual_accuracy


# ------------------------------------------------------------------------
# 5.2 Consistency Score
# ------------------------------------------------------------------------
def calculate_consistency_score(model_output: str) -> float:
    """
    Evaluates the internal logical consistency of the model's output.
    Returns a score in [0.0, 1.0] or [0,100] depending on preference.

    Parameters:
    - model_output: The predicted text from the model (string).

    Returns:
    - consistency_score (float): A numeric measure of consistency.

    Note:
    - This function is a placeholder. A real implementation might
      parse the text for contradictory statements or rely on a separate
      contradiction-detection model to produce a numeric score.
    """
    # --- Placeholder logic ---
    # For demonstration, if the output includes "not" and "is" about the same subject,
    # we interpret that as potential contradiction. Real logic would be more advanced.
    # We'll just return a random, constant, or heuristic-based value for now.
    # For a short text, assume "fully consistent" unless flagged.
    # A better approach might do rule-based or ML-based contradiction checks.
    if "contradiction" in str(model_output).lower():
        return 0.5  # penalize presence of "contradiction" keyword
    return 1.0


# ------------------------------------------------------------------------
# 5.3 Ontology Alignment Score
# ------------------------------------------------------------------------
def calculate_ontology_alignment_score(model_output: str, ontology_concepts: List[str]) -> float:
    """
    Evaluates how well the entities and relationships in the model's output
    align with a known set of ontology concepts.

    Parameters:
    - model_output: The predicted text from the model (string).
    - ontology_concepts: A list of strings representing valid entities,
                          classes, or relationships from the domain ontology.

    Returns:
    - alignment_score (float): A value in [0.0, 1.0] or [0,100] depending on preference.

    Note:
    - In a real scenario, you would parse the output into structured form (triples, etc.)
      and check for coverage or correct usage of the ontology’s classes and properties.
    """
    # --- Example approach: ratio of recognized domain concepts in output vs. total domain concepts used. ---
    # Tokenize or split model output to identify which ontology concepts are mentioned.
    output_lower = str(model_output).lower()
    recognized_count = 0
    used_concepts = 0

    for concept in ontology_concepts:
        concept_lower = concept.lower()
        if concept_lower in output_lower:
            recognized_count += 1
            used_concepts += 1
        else:
            # If you want to measure partial coverage, or track how many were not used, etc.
            # For now, we only count those actually used in the text.
            pass

    if used_concepts == 0:
        # If the model output doesn't use any ontology concept from the list, score could be 0
        # Or you could define a different logic (like if no usage was expected).
        return 0.0

    # Otherwise, the alignment score is fraction recognized among those actually used in text.
    # In a more advanced approach, you'd parse relationships, not just concepts.
    alignment_score = recognized_count / used_concepts
    return alignment_score


# ------------------------------------------------------------------------
# 5.4 Error Detection and Correction Rate
# ------------------------------------------------------------------------
def calculate_error_detection_and_correction(
    baseline_outputs: List[Dict[str, Any]],
    validated_outputs: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Calculates:
    1) The Error Detection Rate (ratio of detected inconsistencies to total real inconsistencies).
    2) The Error Correction Rate (ratio of successful corrections among detected errors).

    For demonstration, assume each dict has:
      - 'is_consistent' (bool) or 'is_hallucination' (bool),
      - 'was_corrected' (bool) indicating if a correction was applied.

    Parameters:
    - baseline_outputs: List of dictionaries describing the baseline system outputs.
    - validated_outputs: List of dictionaries describing the validated / corrected system outputs.

    Returns:
    - (detection_rate, correction_rate): Both as percentages in [0, 100].
    """
    # Example definitions:
    # "inconsistency" = baseline_outputs with is_consistent=False or is_hallucination=True
    # "detected" = validated_outputs with was_corrected=True
    # "corrected" = validated_outputs with was_corrected=True and new output is consistent

    # 1) Count total real inconsistencies in the baseline
    total_inconsistencies = sum(
        1 for b in baseline_outputs
        if not b.get("is_consistent", True) or b.get("is_hallucination", False)
    )

    # 2) Count how many were detected (i.e., a correction was attempted)
    #    Typically you'd compare the same index or ID in baseline vs. validated to see if an error was "caught."
    detected_inconsistencies = 0
    corrected_inconsistencies = 0
    for b, v in zip(baseline_outputs, validated_outputs):
        was_inconsistent = (not b.get("is_consistent", True)) or b.get("is_hallucination", False)
        was_corrected = v.get("was_corrected", False)
        is_consistent_now = v.get("is_consistent", True) and not v.get("is_hallucination", False)

        if was_inconsistent and was_corrected:
            detected_inconsistencies += 1
            if is_consistent_now:
                corrected_inconsistencies += 1

    if total_inconsistencies == 0:
        detection_rate = 0.0
        correction_rate = 0.0
    else:
        detection_rate = (detected_inconsistencies / total_inconsistencies) * 100
        # Among the ones we detected, how many are truly corrected?
        if detected_inconsistencies == 0:
            correction_rate = 0.0
        else:
            correction_rate = (corrected_inconsistencies / detected_inconsistencies) * 100

    return (round(detection_rate, 2), round(correction_rate, 2))


# ------------------------------------------------------------------------
# 5.5 Human Evaluation Metrics
# ------------------------------------------------------------------------
def calculate_human_evaluation_metrics(
    expert_ratings: List[Dict[str, float]],
    rating_keys: List[str] = None
) -> Dict[str, float]:
    """
    Aggregates human-expert ratings on aspects like clarity, reliability,
    and overall quality from multiple domain experts.

    Parameters:
    - expert_ratings: A list of dictionaries. Each dict might look like:
          {
             'expert_id': 1,
             'clarity': 4.5,
             'reliability': 4.0,
             'overall': 4.3
          }
    - rating_keys: The keys we want to aggregate (default: all numeric keys except 'expert_id').

    Returns:
    - average_scores: A dict of the form {metric_name: avg_value, ...}
    """
    if not expert_ratings:
        return {}

    if rating_keys is None:
        # By default, take all numeric keys except 'expert_id'
        rating_keys = [k for k in expert_ratings[0].keys()
                       if k != 'expert_id' and isinstance(expert_ratings[0][k], (int, float))]

    # Compute average for each rating key
    result = {}
    for key in rating_keys:
        all_values = [r[key] for r in expert_ratings if key in r]
        if all_values:
            avg_value = statistics.mean(all_values)
            result[key] = round(avg_value, 2)
        else:
            result[key] = None  # or 0, or skip

    return result


def calculate_inter_rater_agreement(raters_data: List[List[int]]) -> float:
    """
    Computes an inter-rater agreement statistic (e.g., Cohen's Kappa)
    for binary or categorical ratings across multiple raters.

    Parameters:
    - raters_data: A list of lists. Each sub-list is the set of
      ratings from one rater. E.g.,
      [
        [1, 1, 0, 1],  # Rater A's ratings on 4 items
        [1, 0, 0, 1],  # Rater B's ratings on 4 items
      ]

    Returns:
    - kappa (float): A placeholder for inter-rater agreement measure in [0, 1].

    Notes:
    - For a real implementation, you would import or implement Cohen's Kappa
      or Fleiss’ Kappa from a stats library. Here we provide a stub or demonstration.
    """
    # --- Placeholder demonstration only ---
    # Example: A trivial approach that checks how often raters match
    # (not a real Kappa calculation).
    # For a real Kappa, you can do:
    #    from statsmodels.stats.inter_rater import cohens_kappa
    # or implement your own.
    if not raters_data or len(raters_data) < 2:
        return 1.0  # if there's only one rater or none, we trivially have no disagreement

    # Transpose so we can look at each "item" across raters
    num_items = len(raters_data[0])
    # Make sure all raters have the same number of items
    for rd in raters_data:
        if len(rd) != num_items:
            raise ValueError("All raters must have the same number of items to compare.")

    # Count matches vs. total possible
    total_comparisons = 0
    matches = 0
    # Compare each item across all pairs of raters
    for item_index in range(num_items):
        # For each item, collect all rater responses
        item_ratings = [rater[item_index] for rater in raters_data]
        # Compare each pair
        for i in range(len(item_ratings)):
            for j in range(i + 1, len(item_ratings)):
                total_comparisons += 1
                if item_ratings[i] == item_ratings[j]:
                    matches += 1

    if total_comparisons == 0:
        return 1.0

    # A naive measure: fraction of pairwise matches
    raw_agreement = matches / total_comparisons
    # This is not a real Kappa, but let's call it "kappa" as a placeholder
    kappa = raw_agreement
    return round(kappa, 2)


# ------------------------------------------------------------------------
# 5.6 Efficiency and Latency
# ------------------------------------------------------------------------
def measure_efficiency_and_latency(
    function_to_run,
    *args,
    **kwargs
) -> float:
    """
    Measures the execution time (latency) of a given function call, in seconds.

    Parameters:
    - function_to_run: The function or callable to be timed.
    - *args, **kwargs: Arguments passed to `function_to_run`.

    Returns:
    - elapsed_time (float): Time in seconds taken for the function to complete.
    """
    start_time = time.time()
    function_to_run(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time