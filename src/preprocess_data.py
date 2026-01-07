"""Data preprocessing script for LLM uncertainty quantification.

This script processes the raw JSONL dataset and converts it into a format
compatible with the feature engineering pipeline. Each content_id with multiple
questions is split into separate samples, one per concept.
"""

import json
import pandas as pd
import structlog

from pathlib import Path
from typing import Dict, List, Tuple

logger = structlog.getLogger(__name__)


# Concept to question mapping
CONCEPT_QUESTION_MAP = {
    "ADULT": [
        "contain list of questions involve or intended foradult content",
        "contain question involve or intended for adult content",
    ],
    "KIDS": [
        "contain list of questions involve or intended for content aimed at kids",
        "contain question involve or intended for content aimed at kids",
    ],
    "DIMC": ["contain list of questions involve or intended for dimc"],
    "DAT": ["contain list of questions involve or intended for dat"],
}


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries, one per line
    """
    logger.info("Loading JSONL file", path=str(file_path))
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip().rstrip(","))
                data.append(record)
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse JSON line", line_num=line_num, error=str(e)
                )

    logger.info("Loaded records", count=len(data))
    return data


def find_question_in_logprobs(
    logprobs_data: List[Dict], target_questions: List[str]
) -> Tuple[Dict, str]:
    """Find the logprobs entry matching one of the target questions.

    Args:
        logprobs_data: List of question/answer/logprobs dictionaries
        target_questions: List of question texts to find (tries each in order)

    Returns:
        Tuple of (matching entry dict, matched question text), or (empty dict, empty string) if not found
    """
    # Normalize all target questions for comparison
    target_normalized = [q.strip().lower() for q in target_questions]

    for entry in logprobs_data:
        if not isinstance(entry, dict):
            continue

        question = entry.get("question", "")
        if isinstance(question, str):
            question_normalized = question.strip().lower()
            if question_normalized in target_normalized:
                # Return both the entry and which question matched
                return entry, question

    return {}, ""


def split_content_into_samples(
    record: Dict,
) -> List[Tuple[str, str, Dict, int, int, str, str]]:
    """Split a content record into separate samples for each concept.

    Args:
        record: Raw record from JSONL with all concepts and questions

    Returns:
        List of tuples: (content_id, concept, sample_data, is_correct, ground_truth, llm_prediction, matched_question)
        - is_correct: 1 if LLM prediction matches ground truth, 0 otherwise (the label)
        - ground_truth: Original label from dataset (1 or 0)
        - llm_prediction: LLM's answer (YES or NO)
        - matched_question: The actual question that was found and matched
    """
    content_id = record.get("content_id")
    logprobs_data = record.get("logprobs_data", [])

    # Handle case where logprobs_data is a JSON string instead of a list
    if isinstance(logprobs_data, str):
        try:
            logprobs_data = json.loads(logprobs_data)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse logprobs_data string",
                content_id=content_id,
                error=str(e),
            )
            logprobs_data = []

    if not content_id:
        logger.warning("Record missing content_id", record=record)
        return []

    samples = []

    for concept, question_variants in CONCEPT_QUESTION_MAP.items():
        # Ensure question_variants is always a list
        if isinstance(question_variants, str):
            question_variants = [question_variants]

        # Get the ground truth label for this concept
        label_column = f"{concept}_thumbnail+text"
        ground_truth = record.get(label_column)

        # Skip if ground truth is missing
        if ground_truth is None:
            logger.warning(
                "Missing ground truth for concept",
                content_id=content_id,
                concept=concept,
                column=label_column,
            )
            continue

        # Find the matching question in logprobs_data (try all variants)
        question_entry, matched_question = find_question_in_logprobs(
            logprobs_data, question_variants
        )

        if not question_entry:
            logger.warning(
                "No question variant found in logprobs_data",
                content_id=content_id,
                concept=concept,
                num_variants=len(question_variants),
            )
            continue

        # Get LLM's prediction
        llm_answer = question_entry.get("answer", "").strip().upper()

        if llm_answer not in ["YES", "NO"]:
            logger.warning(
                "Invalid LLM answer",
                content_id=content_id,
                concept=concept,
                answer=llm_answer,
            )
            continue

        # Calculate if LLM is correct
        ground_truth_int = int(ground_truth)
        llm_predicts_positive = llm_answer == "YES"
        is_correct = int(
            (ground_truth_int == 1 and llm_predicts_positive)
            or (ground_truth_int == 0 and not llm_predicts_positive)
        )

        # Create sample in the format expected by feature engineering pipeline
        sample_data = {
            "content_id": content_id,
            "concept": concept,
            "title": record.get("title", ""),
            "transcript": record.get("transcript", ""),
            "logprobs_data": [question_entry],
            "error": None,
        }

        samples.append(
            (
                content_id,
                concept,
                sample_data,
                is_correct,
                ground_truth_int,
                llm_answer,
                matched_question,
            )
        )

    return samples


def _process_raw_data(
    raw_data: List[Dict],
) -> Tuple[List[Dict], List[int], List[int], List[str], List[str], List[str]]:
    """Process raw data into samples and metadata lists.

    Returns:
        Tuple of (samples, labels, ground_truths, llm_predictions, matched_questions, content_concept_ids)
    """
    all_samples = []
    all_labels = []
    all_ground_truths = []
    all_llm_predictions = []
    all_matched_questions = []
    content_concept_ids = []

    for record in raw_data:
        samples = split_content_into_samples(record)
        for (
            content_id,
            concept,
            sample_data,
            is_correct,
            ground_truth,
            llm_prediction,
            matched_question,
        ) in samples:
            all_samples.append(sample_data)
            all_labels.append(is_correct)
            all_ground_truths.append(ground_truth)
            all_llm_predictions.append(llm_prediction)
            all_matched_questions.append(matched_question)
            content_concept_ids.append(f"{content_id}_{concept}")

    return (
        all_samples,
        all_labels,
        all_ground_truths,
        all_llm_predictions,
        all_matched_questions,
        content_concept_ids,
    )


def _log_preprocessing_statistics(
    raw_data: List[Dict],
    all_labels: List[int],
    all_ground_truths: List[int],
    all_llm_predictions: List[str],
    metadata_df: pd.DataFrame,
) -> None:
    """Log comprehensive preprocessing statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input records: {len(raw_data)}")
    logger.info(f"Output samples: {len(all_labels)}")

    if len(all_labels) == 0:
        logger.info("\nNo samples generated (empty dataset)")
        logger.info("=" * 80)
        return

    _log_label_distribution(all_labels)
    _log_concept_statistics(metadata_df)
    _log_question_variant_usage(metadata_df)
    _log_prediction_statistics(all_ground_truths, all_llm_predictions)
    logger.info("=" * 80)


def _log_label_distribution(all_labels: List[int]) -> None:
    """Log the distribution of correct vs incorrect labels."""
    logger.info("\nLabel Distribution (LLM Correctness):")
    correct_count = sum(all_labels)
    total_count = len(all_labels)
    logger.info(
        f"  Correct (1): {correct_count} ({correct_count/total_count*100:.1f}%)"
    )
    logger.info(
        f"  Incorrect (0): {total_count - correct_count} ({(total_count-correct_count)/total_count*100:.1f}%)"
    )


def _log_concept_statistics(metadata_df: pd.DataFrame) -> None:
    """Log statistics by concept."""
    logger.info("\nSamples by Concept:")
    concept_counts = metadata_df["concept"].value_counts()
    for concept, count in concept_counts.items():
        concept_data = metadata_df[metadata_df["concept"] == concept]
        correct_count = concept_data["is_correct"].sum()
        ground_truth_positive = (concept_data["ground_truth"] == 1).sum()
        logger.info(
            f"  {concept}: {count} samples ({correct_count} correct predictions, {ground_truth_positive} positive ground truth)"
        )


def _log_question_variant_usage(metadata_df: pd.DataFrame) -> None:
    """Log which question variants were used for each concept."""
    logger.info("\nQuestion Variant Usage by Concept:")
    for concept in CONCEPT_QUESTION_MAP.keys():
        concept_data = metadata_df[metadata_df["concept"] == concept]
        if len(concept_data) > 0:
            question_counts = concept_data["question_used"].value_counts()
            logger.info(f"  {concept}:")
            for question, count in question_counts.items():
                question_preview = (
                    question[:80] + "..." if len(question) > 80 else question
                )
                logger.info(f"    {count} samples: {question_preview}")


def _log_prediction_statistics(
    all_ground_truths: List[int], all_llm_predictions: List[str]
) -> None:
    """Log ground truth vs prediction statistics."""
    logger.info("\nGround Truth vs LLM Prediction:")
    gt_positive_count = sum(all_ground_truths)
    total_count = len(all_ground_truths)
    llm_yes_count = sum(1 for p in all_llm_predictions if p == "YES")

    logger.info(
        f"  Ground Truth Positive: {gt_positive_count} ({gt_positive_count/total_count*100:.1f}%)"
    )
    logger.info(
        f"  LLM Predicted Positive (YES): {llm_yes_count} ({llm_yes_count/total_count*100:.1f}%)"
    )


def preprocess_dataset(
    input_file: Path, output_samples_file: Path, output_labels_file: Path
) -> Tuple[List[Dict], pd.Series]:
    """Preprocess the JSONL dataset into samples and labels.

    Args:
        input_file: Path to input JSONL file
        output_samples_file: Path to save processed samples (JSON)
        output_labels_file: Path to save labels (CSV)

    Returns:
        Tuple of (samples_list, labels_series)
    """
    # Load raw data
    raw_data = load_jsonl(input_file)
    logger.info("Processing records into samples", num_records=len(raw_data))

    # Process data into samples and metadata
    (
        all_samples,
        all_labels,
        all_ground_truths,
        all_llm_predictions,
        all_matched_questions,
        content_concept_ids,
    ) = _process_raw_data(raw_data)

    logger.info(
        "Created samples",
        num_samples=len(all_samples),
        num_correct=sum(all_labels),
        num_incorrect=len(all_labels) - sum(all_labels),
    )

    # Convert labels to pandas Series
    labels_series = pd.Series(all_labels, name="is_correct")

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(
        {
            "content_concept_id": content_concept_ids,
            "content_id": [s["content_id"] for s in all_samples],
            "concept": [s["concept"] for s in all_samples],
            "ground_truth": all_ground_truths,
            "llm_prediction": all_llm_predictions,
            "is_correct": all_labels,
            "question_used": all_matched_questions,
        }
    )

    # Save files
    logger.info("Saving samples", path=str(output_samples_file))
    with open(output_samples_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    logger.info("Saving labels and metadata", path=str(output_labels_file))
    metadata_df.to_csv(output_labels_file, index=False)

    # Log comprehensive statistics
    _log_preprocessing_statistics(
        raw_data, all_labels, all_ground_truths, all_llm_predictions, metadata_df
    )

    return all_samples, labels_series


def main():
    """Run the main preprocessing function."""
    # File paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "test_results_gemini.jsonl"
    output_samples_file = base_dir / "preprocessed_samples.json"
    output_labels_file = base_dir / "preprocessed_labels.csv"

    logger.info("Starting preprocessing", input_file=str(input_file))

    # Preprocess
    samples, labels = preprocess_dataset(
        input_file, output_samples_file, output_labels_file
    )

    logger.info("\n✅ Preprocessing complete!")
    logger.info(f"   Samples saved to: {output_samples_file}")
    logger.info(f"   Labels saved to: {output_labels_file}")

    # Show first sample
    logger.info("\nFirst sample structure:")
    logger.info(json.dumps(samples[0], indent=2)[:1000])

    return samples, labels


if __name__ == "__main__":
    main()
