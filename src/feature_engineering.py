"""Feature Engineering module for LLM Uncertainty Quantification.

This module provides classes and functions for extracting uncertainty-related features
from LLM outputs, particularly focusing on model-centric features derived from logprobs.
"""

import numpy as np
import pandas as pd
import structlog
import warnings

from scipy.stats import entropy
from typing import Any, Dict, List, Optional, Protocol

logger = structlog.getLogger(__name__)


# ==================== Utility Functions ====================


def parse_logprobs_array(logprobs: Any) -> np.ndarray:
    """Parse logprobs from various formats into a numpy array.

    Args:
        logprobs: Can be a string, list, or numpy array of logprob values

    Returns:
        np.ndarray: Array of logprob values, empty array if parsing fails
    """
    # Check type first before using 'in' operator to avoid numpy ambiguity
    if logprobs is None or (isinstance(logprobs, str) and logprobs in ["None", ""]):
        return np.array([])

    # Handle pandas NA values
    try:
        if pd.isna(logprobs):
            return np.array([])
    except (ValueError, TypeError):
        # pd.isna can fail on some types, continue processing
        pass

    try:
        # If already numpy array
        if isinstance(logprobs, np.ndarray):
            return logprobs

        # If list
        if isinstance(logprobs, list):
            return np.array([float(x) for x in logprobs])

        # If string (comma-separated)
        if isinstance(logprobs, str):
            return np.array([float(x) for x in logprobs.split(",")])

        return np.array([])
    except (ValueError, TypeError) as e:
        warnings.warn(f"Failed to parse logprobs: {logprobs}, error: {e}")
        return np.array([])


def parse_top_logprobs(top_logprobs: Any) -> List[Dict[str, float]]:
    """Parse top_logprobs from various formats into a list of dictionaries.

    Args:
        top_logprobs: Can be a list of dicts or a string representation

    Returns:
        List[Dict]: List of dictionaries with 'token', 'logprob', 'linear_prob' keys
    """
    # Check type first before using 'in' operator to avoid numpy/list ambiguity
    if top_logprobs is None or (
        isinstance(top_logprobs, str) and top_logprobs in ["None", ""]
    ):
        return []

    # Handle pandas NA values
    try:
        if pd.isna(top_logprobs):
            return []
    except (ValueError, TypeError):
        # pd.isna can fail on some types, continue processing
        pass

    try:
        if isinstance(top_logprobs, list):
            # Empty list is valid
            if len(top_logprobs) == 0:
                return []
            # Validate structure
            if all(isinstance(item, dict) for item in top_logprobs):
                return top_logprobs
        return []
    except (ValueError, TypeError) as e:
        warnings.warn(f"Failed to parse top logprobs: {top_logprobs}, error: {e}")
        return []


def get_stable_probabilities(top_logprobs: List[Dict[str, float]]) -> np.ndarray:
    """Calculate stable probabilities from top logprobs using log-sum-exp trick.

    Args:
        top_logprobs: List of dicts with 'logprob' key

    Returns:
        np.ndarray: Normalized probability distribution
    """
    if not top_logprobs:
        return np.array([1.0])

    logprobs = np.array(
        [d.get("logprob", -np.inf) for d in top_logprobs if isinstance(d, dict)]
    )

    if logprobs.size == 0:
        return np.array([1.0])

    # Log-sum-exp trick for numerical stability
    max_logprob = np.max(logprobs)
    exp_logprobs = np.exp(logprobs - max_logprob)
    return exp_logprobs / np.sum(exp_logprobs)


def filter_answer_tokens(
    top_logprobs: List[Dict[str, float]], valid_tokens: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """Filter top_logprobs to only include specific answer tokens.

    Args:
        top_logprobs: List of token logprob dictionaries
        valid_tokens: List of valid tokens (default: ["YES", "NO"])

    Returns:
        List[Dict]: Filtered list containing only valid tokens
    """
    if valid_tokens is None:
        valid_tokens = ["YES", "NO"]

    valid_tokens_upper = [t.upper() for t in valid_tokens]

    filtered = []
    for item in top_logprobs:
        if isinstance(item, dict):
            token_value = item.get("token", "")
            if isinstance(token_value, str):
                token = token_value.strip().upper()
                if token in valid_tokens_upper:
                    filtered.append(item)

    return filtered


# ==================== Feature Extractor Protocol ====================


class FeatureExtractor(Protocol):
    """Protocol defining the interface for feature extractors."""

    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from input data.

        Args:
            data: Dictionary containing logprobs data

        Returns:
            Dict[str, float]: Dictionary of feature name to value mappings
        """
        ...


# ==================== Concrete Feature Extractors ====================


class EntropyFeatureExtractor:
    """Extracts entropy-based uncertainty features from probability distributions."""

    def __init__(self, use_filtered: bool = False):
        """Initialize entropy feature extractor.

        Args:
            use_filtered: Whether to use filtered logprobs (YES/NO only)
        """
        self.use_filtered = use_filtered
        self.suffix = "_filtered" if use_filtered else "_top_5"

    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract entropy features from top logprobs.

        Args:
            data: Dictionary with 'top_logprobs' key

        Returns:
            Dict with entropy, normalized_entropy, effective_choices, confidence_score
        """
        top_logprobs = data.get("top_logprobs", [])

        if self.use_filtered:
            top_logprobs = filter_answer_tokens(top_logprobs)

        probs = get_stable_probabilities(top_logprobs)

        if len(probs) < 2:
            return {
                f"model_outcome_entropy{self.suffix}": 0.0,
                f"model_outcome_normalized_entropy{self.suffix}": 0.0,
                f"model_effective_choices{self.suffix}": 0.0,
                f"model_confidence_score{self.suffix}": 1.0,
            }

        # Calculate entropy (base 2)
        ent = entropy(probs, base=2)

        # Normalized entropy
        max_entropy = np.log2(len(probs))
        norm_ent = ent / max_entropy if max_entropy > 0 else 0.0

        # Effective number of choices
        eff_choices = 2**ent

        # Confidence score (inverse of normalized entropy)
        confidence = 1.0 - norm_ent

        return {
            f"model_outcome_entropy{self.suffix}": float(ent),
            f"model_outcome_normalized_entropy{self.suffix}": float(norm_ent),
            f"model_effective_choices{self.suffix}": float(eff_choices),
            f"model_confidence_score{self.suffix}": float(confidence),
        }


class ConfidenceFeatureExtractor:
    """Extracts confidence-based features from probability distributions."""

    def __init__(self, use_filtered: bool = False):
        """Initialize confidence feature extractor.

        Args:
            use_filtered: Whether to use filtered logprobs (YES/NO only)
        """
        self.use_filtered = use_filtered
        self.suffix = "_filtered" if use_filtered else ""

    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence metrics: MSP, P1-P2 gap, ratio.

        Args:
            data: Dictionary with 'top_logprobs' key

        Returns:
            Dict with max_softmax_prob, p1_p2_gap, p1_p2_gap_normalized, confidence_ratio
        """
        top_logprobs = data.get("top_logprobs", [])

        if self.use_filtered:
            top_logprobs = filter_answer_tokens(top_logprobs)

        probs = get_stable_probabilities(top_logprobs)

        if len(probs) < 2:
            msp = probs[0] if len(probs) == 1 else 1.0
            return {
                f"model_max_softmax_prob{self.suffix}": float(msp),
                f"model_p1_p2_gap{self.suffix}": 0.0,
                f"model_p1_p2_gap_normalized{self.suffix}": 0.0,
                f"model_confidence_ratio{self.suffix}": 0.0,
            }

        msp = probs[0]
        p1_p2_gap = float(probs[0] - probs[1])
        p1_p2_gap_normalized = p1_p2_gap / (probs[0] + 1e-12)
        ratio = probs[0] / (probs[1] + 1e-12)

        features = {
            f"model_p1_p2_gap{self.suffix}": p1_p2_gap,
            f"model_p1_p2_gap_normalized{self.suffix}": p1_p2_gap_normalized,
            f"model_confidence_ratio{self.suffix}": float(ratio),
        }

        if not self.use_filtered:
            features["model_max_softmax_prob"] = float(msp)

        return features


class LogprobGapFeatureExtractor:
    """Extracts features based on logprob differences between top tokens."""

    def __init__(self, use_filtered: bool = False):
        """Initialize logprob gap feature extractor.

        Args:
            use_filtered: Whether to use filtered logprobs (YES/NO only)
        """
        self.use_filtered = use_filtered
        self.suffix = "_filtered" if use_filtered else ""

    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract logprob gap between top two predictions.

        Args:
            data: Dictionary with 'top_logprobs' key

        Returns:
            Dict with logprob_gap and logprob_gap_normalized
        """
        top_logprobs = data.get("top_logprobs", [])

        if self.use_filtered:
            top_logprobs = filter_answer_tokens(top_logprobs)

        if not top_logprobs or len(top_logprobs) < 2:
            return {
                f"model_logprob_gap{self.suffix}": 0.0,
                f"model_logprob_gap_normalized{self.suffix}": 1.0,
            }

        logprob1 = top_logprobs[0].get("logprob", np.nan)
        logprob2 = top_logprobs[1].get("logprob", np.nan)

        if pd.isna(logprob1) or pd.isna(logprob2):
            return {
                f"model_logprob_gap{self.suffix}": 0.0,
                f"model_logprob_gap_normalized{self.suffix}": 1.0,
            }

        # Ensure we have valid float values
        logprob1 = float(logprob1)
        logprob2 = float(logprob2)

        gap = logprob2 - logprob1
        gap_normalized = gap / logprob2 if logprob2 != 0 else 1.0

        return {
            f"model_logprob_gap{self.suffix}": float(gap),
            f"model_logprob_gap_normalized{self.suffix}": float(gap_normalized),
        }


# ==================== Main Feature Engineering Pipeline ====================


class FeatureEngineeringPipeline:
    """Production-grade feature engineering pipeline for LLM uncertainty quantification.

    This pipeline processes raw LLM output (including logprobs) and generates
    model-centric features for uncertainty estimation following SOLID principles.
    """

    def __init__(
        self,
        extractors: Optional[List[FeatureExtractor]] = None,
        include_filtered_features: bool = True,
    ):
        """Initialize the feature engineering pipeline.

        Args:
            extractors: Custom list of feature extractors (uses default if None)
            include_filtered_features: Whether to include YES/NO filtered features
        """
        if extractors is None:
            extractors = self._create_default_extractors(include_filtered_features)

        self.extractors = extractors
        self.feature_names: List[str] = []
        logger.info(
            "Initialized feature engineering pipeline",
            num_extractors=len(extractors),
        )

    def _create_default_extractors(
        self, include_filtered: bool
    ) -> List[FeatureExtractor]:
        """Create the default set of feature extractors.

        Args:
            include_filtered: Whether to include filtered (YES/NO) variants

        Returns:
            List of feature extractor instances
        """
        extractors: List[FeatureExtractor] = [
            EntropyFeatureExtractor(use_filtered=False),
            ConfidenceFeatureExtractor(use_filtered=False),
            LogprobGapFeatureExtractor(use_filtered=False),
        ]

        if include_filtered:
            extractors.extend(
                [
                    EntropyFeatureExtractor(use_filtered=True),
                    ConfidenceFeatureExtractor(use_filtered=True),
                    LogprobGapFeatureExtractor(use_filtered=True),
                ]
            )

        return extractors

    def transform_single(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Transform a single record by extracting all features.

        Args:
            record: Dictionary containing logprobs_data (list of question records)

        Returns:
            Dict of feature name to value mappings
        """
        features: Dict[str, float] = {}

        # Extract logprobs_data - it's a list of questions/answers
        logprobs_data = record.get("logprobs_data", [])

        if not logprobs_data:
            logger.warning("No logprobs_data found in record")
            return features

        # Process the first question/answer (can be extended for multiple)
        first_qa = logprobs_data[0] if isinstance(logprobs_data, list) else {}

        # Extract top_logprobs from the first question
        top_logprobs = first_qa.get("top_logprobs", [])

        # Prepare data for extractors
        extractor_data = {"top_logprobs": top_logprobs}

        # Run all extractors
        for extractor in self.extractors:
            try:
                extracted = extractor.extract(extractor_data)
                features.update(extracted)
            except Exception as e:
                logger.error(
                    "Feature extraction failed",
                    extractor=type(extractor).__name__,
                    error=str(e),
                )

        return features

    def transform(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform a list of records into a feature DataFrame.

        Args:
            data: List of dictionaries with logprobs_data

        Returns:
            pd.DataFrame: Features for all records
        """
        logger.info("Starting feature engineering", num_records=len(data))

        features_list = []
        for idx, record in enumerate(data):
            try:
                features = self.transform_single(record)
                features_list.append(features)
            except Exception as e:
                logger.error("Failed to process record", index=idx, error=str(e))
                features_list.append({})

        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()

        logger.info(
            "Feature engineering complete",
            num_features=len(self.feature_names),
            features=self.feature_names,
        )

        return df

    def fit_transform(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fit the pipeline and transform data (compatibility method).

        Args:
            data: List of dictionaries with logprobs_data

        Returns:
            pd.DataFrame: Extracted features
        """
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names generated by this pipeline.

        Returns:
            List of feature names
        """
        return self.feature_names.copy()


# ==================== Convenience Functions ====================


def create_feature_pipeline(
    include_filtered: bool = True,
) -> FeatureEngineeringPipeline:
    """Create a standard feature engineering pipeline.

    Args:
        include_filtered: Whether to include YES/NO filtered features

    Returns:
        FeatureEngineeringPipeline: Configured pipeline instance
    """
    return FeatureEngineeringPipeline(include_filtered_features=include_filtered)
