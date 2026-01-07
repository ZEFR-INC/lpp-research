"""Downsampling module for handling imbalanced datasets.

This module provides strategies for undersampling the majority class in binary
classification problems, following best practices from imbalanced learning literature.

"""

import numpy as np
import pandas as pd
import structlog

from dataclasses import dataclass
from enum import Enum
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

logger = structlog.getLogger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies for balancing datasets."""

    RANDOM = "random"
    TOMEK = "tomek"
    ENN = "enn"
    HYBRID = "hybrid"


@dataclass
class DownsamplingConfig:
    """Configuration for downsampling pipeline.

    In LLM uncertainty quantification, the typical scenario is:
    - Positive class (1): LLM is correct (majority class, 75-90% of cases)
    - Negative class (0): LLM is incorrect (minority class, 10-25% of cases)

    The pipeline automatically detects the majority class and undersamples it
    to achieve the desired positive:negative ratio.

    Attributes:
        train_ratio: Target positive:negative ratio in training (e.g., 6 means 6 pos : 1 neg)
        test_ratio: Target positive:negative ratio in test (e.g., 3 means 3 pos : 1 neg)
        sampling_strategy: Strategy to use for undersampling the majority class
        random_state: Random seed for reproducibility
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        min_samples_per_class: Minimum number of samples required per class
        verbose: Whether to print detailed logging
    """

    train_ratio: float = 6.0  # 6:1 ratio (6 positives : 1 negative)
    test_ratio: float = 3.0  # 3:1 ratio (3 positives : 1 negative)
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    random_state: int = 42
    test_size: float = 0.2
    min_samples_per_class: int = 10
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.train_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("Ratios must be positive")

        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be at least 1")


class DownsamplingPipeline:
    """Production-grade pipeline for balanced undersampling.

    This pipeline handles imbalanced binary classification data by:
    1. Splitting data into train/test with stratification
    2. Undersampling the majority class to achieve target ratios
    3. Optionally applying cleaning techniques (Tomek Links, ENN)
    4. Ensuring reproducibility and data quality

    Example:
        ```python
        from downsampling import DownsamplingPipeline, DownsamplingConfig

        # Configure pipeline
        config = DownsamplingConfig(
            train_ratio=1/6,
            test_ratio=1/3,
            sampling_strategy=SamplingStrategy.RANDOM
        )
        pipeline = DownsamplingPipeline(config)

        # Prepare balanced data
        X_train, X_test, y_train, y_test, report = pipeline.fit_resample(
            X, y, feature_columns=feature_cols
        )
        ```
    """

    def __init__(self, config: Optional[DownsamplingConfig] = None):
        """Initialize downsampling pipeline.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DownsamplingConfig()
        self.stats: List[Dict] = []

        logger.info(
            "Initialized downsampling pipeline",
            train_ratio=self.config.train_ratio,
            test_ratio=self.config.test_ratio,
            strategy=self.config.sampling_strategy.value,
        )

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict, np.ndarray, np.ndarray
    ]:
        """Perform balanced undersampling on the dataset.

        Args:
            X: Feature DataFrame
            y: Target Series (binary: 0=negative, 1=positive)
            feature_columns: List of feature columns to use (uses all if None)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, report_dict, train_indices, test_indices)
            The indices correspond to the original positions in the input X/y

        Raises:
            ValueError: If data validation fails or insufficient samples
        """
        # Validate inputs
        X_clean, y_clean, feature_columns = self._validate_and_clean(
            X, y, feature_columns
        )

        # Check class distribution
        class_counts = y_clean.value_counts()
        if len(class_counts) != 2:
            raise ValueError(
                f"Expected binary classification, got {len(class_counts)} classes"
            )

        pos_count = int(class_counts.get(1, 0))
        neg_count = int(class_counts.get(0, 0))

        logger.info(
            "Original class distribution",
            positives=pos_count,
            negatives=neg_count,
            ratio=pos_count / neg_count if neg_count > 0 else 0,
        )

        # Split into train/test with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean[feature_columns],
            y_clean,
            test_size=self.config.test_size,
            stratify=y_clean,
            random_state=self.config.random_state,
        )

        # Apply downsampling to training set and track removed samples
        X_train_balanced, y_train_balanced, train_removed_indices = (
            self._downsample_split(
                X_train,
                y_train,
                target_ratio=self.config.train_ratio,
                split_name="train",
            )
        )

        # For TOMEK/HYBRID strategies, transfer unused positive samples to test set
        # to help achieve the desired test ratio
        if self.config.sampling_strategy in [
            SamplingStrategy.TOMEK,
            SamplingStrategy.HYBRID,
        ]:
            # Get positive samples that were removed from training
            removed_pos_indices = [
                idx for idx in train_removed_indices if y_train.loc[idx] == 1
            ]

            if removed_pos_indices:
                logger.info(
                    "Transferring removed positive samples from train to test",
                    n_samples=len(removed_pos_indices),
                    strategy=self.config.sampling_strategy.value,
                )

                # Add these samples to the test set
                X_test_augmented = pd.concat(
                    [X_test, X_train.loc[removed_pos_indices]], axis=0
                )
                y_test_augmented = pd.concat(
                    [y_test, y_train.loc[removed_pos_indices]], axis=0
                )
                # Ensure y_test_augmented is a Series
                if isinstance(y_test_augmented, pd.DataFrame):
                    y_test_augmented = y_test_augmented.iloc[:, 0]
            else:
                X_test_augmented = X_test
                y_test_augmented = y_test
        else:
            X_test_augmented = X_test
            y_test_augmented = y_test

        # Apply downsampling to test set (with potential augmentation)
        X_test_balanced, y_test_balanced, _ = self._downsample_split(
            X_test_augmented,
            y_test_augmented,
            target_ratio=self.config.test_ratio,
            split_name="test",
        )

        # Get the indices that were kept after downsampling
        train_indices = X_train_balanced.index.to_numpy()
        test_indices = X_test_balanced.index.to_numpy()

        # Generate report
        report = self._generate_report(
            y_clean, y_train, y_test, y_train_balanced, y_test_balanced, feature_columns
        )

        logger.info(
            "Downsampling complete",
            train_size=len(y_train_balanced),
            test_size=len(y_test_balanced),
        )

        return (
            X_train_balanced,
            X_test_balanced,
            y_train_balanced,
            y_test_balanced,
            report,
            train_indices,
            test_indices,
        )

    def _validate_and_clean(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Validate and clean input data.

        Args:
            X: Feature DataFrame
            y: Target Series
            feature_columns: List of feature columns

        Returns:
            Tuple of (cleaned_X, cleaned_y, valid_feature_columns)
        """
        # Validate basic structure
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if len(X) < self.config.min_samples_per_class * 2:
            raise ValueError(
                f"Insufficient samples: {len(X)} < {self.config.min_samples_per_class * 2}"
            )

        # Handle feature columns
        if feature_columns is None:
            feature_columns = X.columns.tolist()
        else:
            missing = set(feature_columns) - set(X.columns)
            if missing:
                logger.warning("Missing feature columns", missing=list(missing))
                feature_columns = [c for c in feature_columns if c in X.columns]

        # Remove rows with null targets
        valid_mask = y.notna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        if len(X_clean) != len(X):
            logger.warning(
                "Removed rows with null targets", removed=len(X) - len(X_clean)
            )

        # Remove constant features
        constant_features = []
        for col in feature_columns:
            if X_clean[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning("Removing constant features", features=constant_features)
            feature_columns = [c for c in feature_columns if c not in constant_features]
            X_clean = X_clean.drop(columns=constant_features)

        # Replace infinities with NaN
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check for high missing rates
        missing_rates = X_clean[feature_columns].isnull().mean()
        high_missing = missing_rates[missing_rates > 0.5].index.tolist()

        if high_missing:
            logger.warning(
                "Features with >50% missing values",
                features=high_missing,
                counts=missing_rates[high_missing].to_dict(),
            )

        return X_clean, y_clean, feature_columns

    def _downsample_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float,
        split_name: str,
    ) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """Downsample a single split to achieve target ratio with tracking of removed samples.

        This is an enhanced version of _downsample_split that also returns the indices
        of samples that were removed during downsampling. This is useful for transferring
        samples between splits when using advanced strategies like TOMEK or HYBRID.

        Args:
            X: Features for this split
            y: Labels for this split (1=correct/positive, 0=incorrect/negative)
            target_ratio: Target positive:negative ratio (e.g., 6 means 6 pos : 1 neg)
            split_name: Name of split for logging

        Returns:
            Tuple of (downsampled_X, downsampled_y, removed_indices_list)
        """
        # Get current class counts
        pos_indices = y[y == 1].index.tolist()
        neg_indices = y[y == 0].index.tolist()

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Store original indices for tracking
        all_original_indices = set(y.index.tolist())

        # Determine majority and minority classes
        if n_pos > n_neg:
            majority_class = 1
            majority_indices = pos_indices
            minority_indices = neg_indices
            n_majority = n_pos
            n_minority = n_neg
            logger.debug(
                f"{split_name}: Positive is majority class",
                positives=n_pos,
                negatives=n_neg,
            )
        else:
            majority_class = 0
            majority_indices = neg_indices
            minority_indices = pos_indices
            n_majority = n_neg
            n_minority = n_pos
            logger.debug(
                f"{split_name}: Negative is majority class",
                positives=n_pos,
                negatives=n_neg,
            )

        # Calculate target majority samples based on desired ratio
        if majority_class == 1:
            # Positive is majority, target_ratio is pos:neg (e.g., 6 means 6 pos : 1 neg)
            target_majority = int(target_ratio * n_minority)
        else:
            # Negative is majority, need to invert ratio (neg:pos)
            target_majority = int(n_minority * target_ratio)

        # Ensure we don't try to upsample
        if target_majority > n_majority:
            logger.warning(
                f"{split_name}: Not enough majority samples for exact ratio",
                requested=target_majority,
                available=n_majority,
                using_all=True,
            )
            target_majority = n_majority

        # Ensure minimum samples
        if target_majority < self.config.min_samples_per_class:
            logger.warning(
                f"{split_name}: Target majority samples below minimum",
                target=target_majority,
                minimum=self.config.min_samples_per_class,
            )
            target_majority = min(self.config.min_samples_per_class, n_majority)

        # Select majority class samples based on strategy
        if self.config.sampling_strategy == SamplingStrategy.RANDOM:
            selected_majority_indices = self._random_undersample(
                majority_indices, target_majority
            )
        elif self.config.sampling_strategy == SamplingStrategy.TOMEK:
            selected_majority_indices = self._tomek_undersample(
                X, y, majority_class, majority_indices, target_majority
            )
        elif self.config.sampling_strategy == SamplingStrategy.ENN:
            selected_majority_indices = self._enn_undersample(
                X, y, majority_class, majority_indices, target_majority
            )
        elif self.config.sampling_strategy == SamplingStrategy.HYBRID:
            selected_majority_indices = self._hybrid_undersample(
                X, y, majority_class, majority_indices, target_majority
            )
        else:
            selected_majority_indices = self._random_undersample(
                majority_indices, target_majority
            )

        # Combine minority (keep all) and selected majority samples
        balanced_indices = minority_indices + selected_majority_indices

        # Calculate removed indices
        kept_indices = set(balanced_indices)
        removed_indices = list(all_original_indices - kept_indices)

        # Shuffle for randomization
        np.random.seed(self.config.random_state)
        np.random.shuffle(balanced_indices)

        X_balanced = X.loc[balanced_indices].copy()
        y_balanced = y.loc[balanced_indices].copy()

        actual_pos = (y_balanced == 1).sum()
        actual_neg = (y_balanced == 0).sum()
        actual_ratio = actual_pos / actual_neg if actual_neg > 0 else float("inf")

        logger.info(
            f"Downsampled {split_name} split",
            original_size=len(y),
            balanced_size=len(y_balanced),
            removed=len(removed_indices),
            target_ratio=target_ratio,
            actual_ratio=round(actual_ratio, 3),
            positives=actual_pos,
            negatives=actual_neg,
        )

        return X_balanced, y_balanced, removed_indices

    def _random_undersample(
        self, neg_indices: List[int], target_count: int
    ) -> List[int]:
        """Randomly undersample negative class.

        Args:
            neg_indices: Indices of negative class samples
            target_count: Number of samples to select

        Returns:
            List of selected indices
        """
        np.random.seed(self.config.random_state)
        return list(np.random.choice(neg_indices, size=target_count, replace=False))

    def _tomek_undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        majority_class: int,
        majority_indices: List[int],
        target_count: int,
    ) -> List[int]:
        """Undersample using Tomek Links to remove boundary samples.

        Note: This method applies Tomek Links for boundary cleaning, then
        randomly samples from the remaining majority class samples.

        Args:
            X: Feature DataFrame
            y: Target Series
            majority_class: The class label of majority (0 or 1)
            majority_indices: Indices of majority class samples
            target_count: Target number of majority samples

        Returns:
            List of selected indices from the original DataFrame
        """
        try:
            # Apply Tomek Links directly on original indices
            # This will remove Tomek Links (boundary samples between classes)
            sampler = TomekLinks(sampling_strategy="majority")
            X_cleaned, y_cleaned = sampler.fit_resample(X, y)

            # Get the indices that survived Tomek Links cleaning
            # These are from the original DataFrame index
            cleaned_indices = X_cleaned.index.tolist()

            # Filter to only majority class indices
            remaining_majority = [
                idx for idx in cleaned_indices if idx in majority_indices
            ]

            # If we removed enough, we're done
            if len(remaining_majority) <= target_count:
                return remaining_majority

            # Otherwise, randomly sample from remaining
            return self._random_undersample(remaining_majority, target_count)

        except Exception as e:
            logger.warning("Tomek Links failed, falling back to random", error=str(e))
            return self._random_undersample(majority_indices, target_count)

    def _enn_undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        majority_class: int,
        majority_indices: List[int],
        target_count: int,
    ) -> List[int]:
        """Undersample using Edited Nearest Neighbors to remove noisy samples.

        Note: This method applies ENN for noise removal, then randomly samples
        from the remaining majority class samples.

        Args:
            X: Feature DataFrame
            y: Target Series
            majority_class: The class label of majority (0 or 1)
            majority_indices: Indices of majority class samples
            target_count: Target number of majority samples

        Returns:
            List of selected indices from the original DataFrame
        """
        try:
            # Apply ENN directly on original indices
            # This will remove noisy samples (misclassified by k-NN)
            sampler = EditedNearestNeighbours(
                sampling_strategy="majority", n_neighbors=3
            )
            X_cleaned, y_cleaned = sampler.fit_resample(X, y)

            # Get the indices that survived ENN cleaning
            # These are from the original DataFrame index
            cleaned_indices = X_cleaned.index.tolist()

            # Filter to only majority class indices
            remaining_majority = [
                idx for idx in cleaned_indices if idx in majority_indices
            ]

            # If we removed enough, we're done
            if len(remaining_majority) <= target_count:
                return remaining_majority

            # Otherwise, randomly sample from remaining
            return self._random_undersample(remaining_majority, target_count)

        except Exception as e:
            logger.warning("ENN failed, falling back to random", error=str(e))
            return self._random_undersample(majority_indices, target_count)

    def _hybrid_undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        majority_class: int,
        majority_indices: List[int],
        target_count: int,
    ) -> List[int]:
        """Hybrid approach: Apply Tomek Links then ENN for boundary cleaning and noise removal.

        This method chains two undersampling techniques:
        1. Tomek Links - Remove boundary samples between classes
        2. ENN - Remove noisy samples misclassified by k-NN
        Then randomly samples from the remaining majority class samples.

        Args:
            X: Feature DataFrame
            y: Target Series
            majority_class: The class label of majority (0 or 1)
            majority_indices: Indices of majority class samples
            target_count: Target number of majority samples

        Returns:
            List of selected indices from the original DataFrame
        """
        try:
            # Step 1: Apply Tomek Links
            tomek = TomekLinks(sampling_strategy="majority")
            X_tomek, y_tomek = tomek.fit_resample(X, y)

            # Get indices that survived Tomek Links
            tomek_indices = X_tomek.index.tolist()

            # Filter to majority class only from Tomek results
            tomek_majority = [idx for idx in tomek_indices if idx in majority_indices]

            # Check if we have enough samples to proceed with ENN
            if len(X_tomek) < self.config.min_samples_per_class * 2:
                logger.info(
                    "Not enough samples after Tomek Links for ENN, using Tomek result",
                    remaining=len(X_tomek),
                )
                if len(tomek_majority) <= target_count:
                    return tomek_majority
                return self._random_undersample(tomek_majority, target_count)

            # Step 2: Apply ENN on Tomek-cleaned data
            enn = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3)
            X_enn, y_enn = enn.fit_resample(X_tomek, y_tomek)

            # Get indices that survived both Tomek and ENN
            enn_indices = X_enn.index.tolist()

            # Filter to majority class only from final results
            final_majority = [idx for idx in enn_indices if idx in majority_indices]

            # If we removed enough, we're done
            if len(final_majority) <= target_count:
                return final_majority

            # Otherwise, randomly sample from remaining
            return self._random_undersample(final_majority, target_count)

        except Exception as e:
            logger.warning(
                "Hybrid sampling failed, falling back to random", error=str(e)
            )
            return self._random_undersample(majority_indices, target_count)

    def _generate_report(
        self,
        y_original: pd.Series,
        y_train_original: pd.Series,
        y_test_original: pd.Series,
        y_train_balanced: pd.Series,
        y_test_balanced: pd.Series,
        feature_columns: List[str],
    ) -> Dict:
        """Generate comprehensive downsampling report.

        Args:
            y_original: Original target before any processing
            y_train_original: Original training targets before balancing
            y_test_original: Original test targets before balancing
            y_train_balanced: Balanced training targets
            y_test_balanced: Balanced test targets
            feature_columns: List of feature column names

        Returns:
            Dictionary with detailed statistics
        """

        def get_distribution(y: pd.Series) -> Dict:
            counts = y.value_counts()
            pos = int(counts.get(1, 0))
            neg = int(counts.get(0, 0))
            return {
                "positive": pos,
                "negative": neg,
                "total": pos + neg,
                "ratio_pos_neg": pos / neg if neg > 0 else 0,
            }

        report = {
            "config": {
                "train_target_ratio": self.config.train_ratio,
                "test_target_ratio": self.config.test_ratio,
                "sampling_strategy": self.config.sampling_strategy.value,
                "random_state": self.config.random_state,
                "test_size": self.config.test_size,
            },
            "distribution": {
                "original": get_distribution(y_original),
                "train_before": get_distribution(y_train_original),
                "train_after": get_distribution(y_train_balanced),
                "test_before": get_distribution(y_test_original),
                "test_after": get_distribution(y_test_balanced),
            },
            "reduction": {
                "train_samples_removed": len(y_train_original) - len(y_train_balanced),
                "test_samples_removed": len(y_test_original) - len(y_test_balanced),
                "train_reduction_pct": (
                    1 - len(y_train_balanced) / len(y_train_original)
                )
                * 100,
                "test_reduction_pct": (1 - len(y_test_balanced) / len(y_test_original))
                * 100,
            },
            "features": {
                "count": len(feature_columns),
                "names": feature_columns,
            },
        }

        # Store for analysis
        self.stats.append(report)

        return report

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary of all downsampling runs as a DataFrame.

        Returns:
            DataFrame with summary statistics for each run
        """
        if not self.stats:
            return pd.DataFrame()

        summary_data = []
        for idx, report in enumerate(self.stats):
            summary_data.append(
                {
                    "run_id": idx,
                    "strategy": report["config"]["sampling_strategy"],
                    "original_samples": report["distribution"]["original"]["total"],
                    "train_before": report["distribution"]["train_before"]["total"],
                    "train_after": report["distribution"]["train_after"]["total"],
                    "test_before": report["distribution"]["test_before"]["total"],
                    "test_after": report["distribution"]["test_after"]["total"],
                    "train_ratio_achieved": report["distribution"]["train_after"][
                        "ratio_pos_neg"
                    ],
                    "test_ratio_achieved": report["distribution"]["test_after"][
                        "ratio_pos_neg"
                    ],
                }
            )

        return pd.DataFrame(summary_data)


def create_downsampling_pipeline(
    train_ratio: float = 6.0,
    test_ratio: float = 3.0,
    strategy: SamplingStrategy = SamplingStrategy.RANDOM,
    random_state: int = 42,
) -> DownsamplingPipeline:
    """Create a downsampling pipeline with common settings.

    In LLM scenarios, positive (LLM correct) is typically the majority class.
    The pipeline automatically undersamples the majority class to achieve
    the specified positive:negative ratio.

    Args:
        train_ratio: Target positive:negative ratio for training (e.g., 6 means 6:1)
        test_ratio: Target positive:negative ratio for test (e.g., 3 means 3:1)
        strategy: Undersampling strategy to use
        random_state: Random seed for reproducibility

    Returns:
        Configured DownsamplingPipeline instance

    Example:
        ```python
        # Create pipeline with 6:1 train ratio and 3:1 test ratio
        pipeline = create_downsampling_pipeline(
            train_ratio=6,  # 6 positives : 1 negative in training
            test_ratio=3,   # 3 positives : 1 negative in test
            strategy=SamplingStrategy.HYBRID
        )
        ```
    """
    config = DownsamplingConfig(
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        sampling_strategy=strategy,
        random_state=random_state,
    )
    return DownsamplingPipeline(config)
