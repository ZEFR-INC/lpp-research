"""Feature analysis and selection for LLM uncertainty quantification.

This module provides comprehensive exploratory data analysis (EDA) and feature selection
capabilities with emphasis on handling multicollinearity and identifying the most predictive
features for meta-model training.

Key Features:
- Multicollinearity detection using correlation analysis and VIF
- Comprehensive EDA with automated visualizations
- Multiple feature selection algorithms with consensus-based selection
- Statistical significance testing and effect size calculations

References:
    - Kuhn & Johnson (2013): Applied Predictive Modeling
    - Guyon & Elisseeff (2003): An Introduction to Variable and Feature Selection
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
import warnings

from dataclasses import dataclass
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logger = structlog.getLogger(__name__)


# ==================== Configuration ====================


@dataclass
class FeatureAnalysisConfig:
    """Configuration for feature analysis pipeline.

    Attributes:
        correlation_threshold: Threshold for correlation-based multicollinearity (default: 0.9)
        vif_threshold: Threshold for VIF-based multicollinearity detection (default: 15.0)
        min_samples: Minimum samples required for reliable analysis (default: 30)
        min_consensus_votes: Minimum votes needed for consensus selection (default: 2)
        top_k_features: Number of features to select in SelectKBest (default: 8)
        min_correlation: Minimum absolute correlation for statistical significance (default: 0.1)
        alpha: Significance level for statistical tests (default: 0.05)
        random_state: Random seed for reproducibility (default: 42)
        create_visualizations: Whether to generate plots (default: True)
        save_plots: Whether to save plots to disk (default: True)
        plot_dir: Directory to save plots (default: "results/feature_analysis")
        figsize_correlation: Figure size for correlation heatmap (default: (12, 10))
        figsize_importance: Figure size for importance plots (default: (12, 6))
        figsize_distributions: Figure size for distribution plots (default: (15, 10))
    """

    correlation_threshold: float = 0.9
    vif_threshold: float = 15.0
    min_samples: int = 11
    min_consensus_votes: int = 2
    top_k_features: int = 8
    min_correlation: float = 0.1
    alpha: float = 0.05
    random_state: int = 42
    create_visualizations: bool = True
    save_plots: bool = True
    plot_dir: str = "results/feature_analysis"
    figsize_correlation: Tuple[int, int] = (12, 10)
    figsize_importance: Tuple[int, int] = (12, 6)
    figsize_distributions: Tuple[int, int] = (15, 10)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.vif_threshold <= 1:
            raise ValueError("vif_threshold must be > 1")
        if self.min_samples < 10:
            raise ValueError("min_samples must be >= 10")
        if self.min_consensus_votes < 1:
            raise ValueError("min_consensus_votes must be >= 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


@dataclass
class FeatureAnalysisResult:
    """Results from feature analysis pipeline.

    Attributes:
        original_features: List of original feature names
        features_after_multicollinearity: List of features after multicollinearity removal
        selected_features: Final consensus selected features
        removed_by_correlation: Features removed due to high correlation
        removed_by_vif: Features removed due to high VIF
        feature_statistics: Descriptive statistics for each feature
        target_correlations: Correlation of each feature with target
        selection_summary: DataFrame showing feature selection across methods
        multicollinearity_pairs: List of highly correlated feature pairs removed
        eda_visualizations_path: Path where visualizations were saved (if applicable)
    """

    original_features: List[str]
    features_after_multicollinearity: List[str]
    selected_features: List[str]
    removed_by_correlation: List[str]
    removed_by_vif: List[str]
    feature_statistics: Dict[str, Dict[str, float]]
    target_correlations: Dict[str, Dict[str, float]]
    selection_summary: pd.DataFrame
    multicollinearity_pairs: List[Tuple[str, str, float]]
    eda_visualizations_path: Optional[str] = None


# ==================== Main Pipeline ====================


class FeatureAnalysisPipeline:
    """Production-grade pipeline for feature analysis and selection.

    This pipeline implements comprehensive exploratory data analysis and feature selection
    with emphasis on multicollinearity handling and statistical rigor.

    Example:
        ```python
        from feature_analysis import (
            FeatureAnalysisPipeline,
            FeatureAnalysisConfig
        )

        # Configure pipeline
        config = FeatureAnalysisConfig(
            correlation_threshold=0.9,
            vif_threshold=15.0,
            save_plots=True
        )
        pipeline = FeatureAnalysisPipeline(config)

        # Run analysis
        result = pipeline.analyze(X_train, y_train)

        # Get selected features
        selected_features = result.selected_features
        X_train_selected = X_train[selected_features]
        ```
    """

    def __init__(self, config: Optional[FeatureAnalysisConfig] = None):
        """Initialize feature analysis pipeline.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or FeatureAnalysisConfig()
        logger.info(
            "Initialized feature analysis pipeline",
            correlation_threshold=self.config.correlation_threshold,
            vif_threshold=self.config.vif_threshold,
            min_samples=self.config.min_samples,
        )

    def analyze(
        self, X: pd.DataFrame, y: pd.Series, analysis_name: str = "dataset"
    ) -> FeatureAnalysisResult:
        """Run complete feature analysis pipeline.

        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target variable (pandas Series with binary labels 0/1)
            analysis_name: Name for this analysis (used in plot titles and filenames)

        Returns:
            FeatureAnalysisResult containing all analysis outputs

        Raises:
            ValueError: If insufficient data or no valid features
        """
        # Validate inputs first
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")

        logger.info(
            "Starting feature analysis",
            samples=len(X),
            features=len(X.columns),
            analysis_name=analysis_name,
        )

        # Validate sample size
        if len(X) < self.config.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(X)} < {self.config.min_samples}"
            )

        # Step 1: Identify numerical features
        numerical_features = self._get_numerical_features(X)

        if len(numerical_features) == 0:
            raise ValueError("No valid numerical features found")

        # Step 2: Perform EDA
        logger.info("Performing exploratory data analysis")
        feature_stats, target_correlations = self._perform_eda(
            X, y, numerical_features, analysis_name
        )

        # Step 3: Handle multicollinearity
        logger.info("Analyzing multicollinearity")
        (
            clean_features,
            removed_corr,
            removed_vif,
            high_corr_pairs,
        ) = self._handle_multicollinearity(X, y, numerical_features)

        # Step 4: Feature selection
        logger.info("Performing feature selection")
        selected_features, selection_summary = self._feature_selection(
            X[clean_features], y
        )

        # Create result object
        result = FeatureAnalysisResult(
            original_features=numerical_features,
            features_after_multicollinearity=clean_features,
            selected_features=selected_features,
            removed_by_correlation=removed_corr,
            removed_by_vif=removed_vif,
            feature_statistics=feature_stats,
            target_correlations=target_correlations,
            selection_summary=selection_summary,
            multicollinearity_pairs=high_corr_pairs,
            eda_visualizations_path=(
                str(Path(self.config.plot_dir) / analysis_name)
                if self.config.save_plots and analysis_name
                else str(Path(self.config.plot_dir)) if self.config.save_plots else None
            ),
        )

        logger.info(
            "Feature analysis complete",
            original_features=len(numerical_features),
            after_multicollinearity=len(clean_features),
            selected_features=len(selected_features),
        )

        return result

    def _get_numerical_features(self, X: pd.DataFrame) -> List[str]:
        """Identify valid numerical features.

        Args:
            X: Feature matrix

        Returns:
            List of numerical feature column names
        """
        numerical_features = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                non_null_count = X[col].notna().sum()
                if non_null_count >= 10:
                    numerical_features.append(col)
                else:
                    logger.debug(
                        "Excluding feature due to insufficient data",
                        feature=col,
                        non_null_count=non_null_count,
                    )

        logger.info("Identified numerical features", count=len(numerical_features))
        return numerical_features

    def _perform_eda(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        analysis_name: str,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Perform comprehensive exploratory data analysis.

        Args:
            X: Feature matrix
            y: Target variable
            features: List of features to analyze
            analysis_name: Name for analysis (used in plots)

        Returns:
            Tuple of (feature_statistics, target_correlations)
        """
        feature_stats = {}
        target_correlations = {}

        # Calculate statistics for each feature
        for feature in features:
            data = X[feature].dropna()
            if len(data) == 0:
                continue

            # Descriptive statistics
            stats_dict = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "cv": float(data.std() / data.mean() if data.mean() != 0 else np.inf),
                "non_null_pct": float((len(data) / len(X)) * 100),
            }
            feature_stats[feature] = stats_dict

            # Correlation with target
            aligned_y = y.loc[data.index]
            try:
                corr, p_val = stats.pearsonr(data, aligned_y)
                target_correlations[feature] = {
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "significant": bool(p_val < self.config.alpha),
                }
            except Exception as e:
                logger.warning(
                    "Failed to calculate correlation",
                    feature=feature,
                    error=str(e),
                )
                target_correlations[feature] = {
                    "correlation": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                }

        # Log summary
        sig_count = sum(1 for v in target_correlations.values() if v["significant"])
        logger.info(
            "EDA complete",
            features_analyzed=len(features),
            significant_correlations=sig_count,
        )

        # Create visualizations
        if self.config.create_visualizations:
            self._create_visualizations(
                X, y, features, target_correlations, analysis_name
            )

        return feature_stats, target_correlations

    def _create_visualizations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        target_correlations: Dict,
        analysis_name: str,
    ):
        """Create and save comprehensive EDA visualizations.

        Args:
            X: Feature matrix
            y: Target variable
            features: List of features
            target_correlations: Target correlation results
            analysis_name: Name for analysis
        """
        # Create output directory
        if self.config.save_plots:
            plot_dir = Path(self.config.plot_dir) / analysis_name
            plot_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created plot directory", path=str(plot_dir))

        # Set plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Correlation heatmap
        if len(features) > 1:
            self._plot_correlation_heatmap(X, features, analysis_name)

        # 2. Target correlation bar plot
        self._plot_target_correlations(target_correlations, analysis_name)

        # 3. Feature distributions
        self._plot_feature_distributions(X, y, target_correlations, analysis_name)

    def _plot_correlation_heatmap(
        self, X: pd.DataFrame, features: List[str], analysis_name: str
    ):
        """Create correlation heatmap."""
        fig, ax = plt.subplots(figsize=self.config.figsize_correlation)

        corr_matrix = X[features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=len(features) <= 20,  # Only annotate if not too many features
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(
            f"Feature Correlation Matrix\n{analysis_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if self.config.save_plots:
            plot_path = (
                Path(self.config.plot_dir) / analysis_name / "correlation_heatmap.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.debug("Saved correlation heatmap", path=str(plot_path))

        plt.close()

    def _plot_target_correlations(self, target_correlations: Dict, analysis_name: str):
        """Create target correlation bar plot."""
        # Filter significant correlations
        sig_correlations = {
            k: v
            for k, v in target_correlations.items()
            if v["significant"] and not np.isnan(v["correlation"])
        }

        if not sig_correlations:
            logger.debug("No significant correlations to plot")
            return

        fig, ax = plt.subplots(figsize=self.config.figsize_importance)

        features_sorted = sorted(
            sig_correlations.keys(),
            key=lambda x: abs(sig_correlations[x]["correlation"]),
            reverse=True,
        )
        correlations = [sig_correlations[f]["correlation"] for f in features_sorted]

        bars = ax.barh(features_sorted, correlations)

        # Color bars based on direction
        for i, bar in enumerate(bars):
            if correlations[i] > 0:
                bar.set_color("green")
            else:
                bar.set_color("red")

        ax.set_xlabel("Correlation with Target", fontsize=12)
        ax.set_title(
            f"Significant Feature-Target Correlations\n{analysis_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if self.config.save_plots:
            plot_path = (
                Path(self.config.plot_dir) / analysis_name / "target_correlations.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.debug("Saved target correlations plot", path=str(plot_path))

        plt.close()

    def _plot_feature_distributions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_correlations: Dict,
        analysis_name: str,
    ):
        """Create distribution plots for top features."""
        # Get top features by absolute correlation
        top_features = sorted(
            target_correlations.items(),
            key=lambda x: (
                abs(x[1]["correlation"]) if not np.isnan(x[1]["correlation"]) else 0
            ),
            reverse=True,
        )[:18]

        if not top_features:
            logger.debug("No features to plot distributions")
            return

        fig, axes = plt.subplots(6, 3, figsize=self.config.figsize_distributions)
        axes = axes.flatten()

        for i, (feature, corr_info) in enumerate(top_features):
            if i >= 18:
                break

            # Plot distributions split by target
            for target_val in [0, 1]:
                data = X.loc[y == target_val, feature].dropna()
                if len(data) > 0:
                    axes[i].hist(
                        data,
                        alpha=0.6,
                        bins=20,
                        label=f"y={target_val}",
                        density=True,
                        edgecolor="black",
                        linewidth=0.5,
                    )

            axes[i].set_title(
                f'{feature}\nr={corr_info["correlation"]:.3f}', fontsize=9
            )
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=8)

        # Hide unused subplots
        for i in range(len(top_features), 18):
            axes[i].set_visible(False)

        plt.suptitle(
            f"Feature Distributions by Target\n{analysis_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if self.config.save_plots:
            plot_path = (
                Path(self.config.plot_dir) / analysis_name / "feature_distributions.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.debug("Saved feature distributions plot", path=str(plot_path))

        plt.close()

    def _handle_multicollinearity(
        self, X: pd.DataFrame, y: pd.Series, features: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, float]]]:
        """Handle multicollinearity using correlation and VIF analysis.

        Args:
            X: Feature matrix
            y: Target variable
            features: List of features to analyze

        Returns:
            Tuple of (clean_features, removed_by_correlation, removed_by_vif, high_corr_pairs)
        """
        if len(features) <= 1:
            logger.debug("Insufficient features for multicollinearity analysis")
            return features, [], [], []

        # Method 1: Correlation-based removal
        correlation_matrix = X[features].corr().abs()
        removed_by_correlation = []
        high_corr_pairs = []

        # Find highly correlated pairs
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if corr_val > self.config.correlation_threshold:
                    feat1 = correlation_matrix.columns[i]
                    feat2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((feat1, feat2, float(corr_val)))

        # Remove one feature from each pair (keep one with higher target correlation)
        features_to_keep = set(features)
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 in features_to_keep and feat2 in features_to_keep:
                target_corr1 = abs(X[feat1].corr(y))
                target_corr2 = abs(X[feat2].corr(y))

                if target_corr1 > target_corr2:
                    features_to_keep.discard(feat2)
                    removed_by_correlation.append(feat2)
                    logger.debug(
                        "Removed feature due to correlation",
                        feature=feat2,
                        correlated_with=feat1,
                        correlation=corr_val,
                    )
                else:
                    features_to_keep.discard(feat1)
                    removed_by_correlation.append(feat1)
                    logger.debug(
                        "Removed feature due to correlation",
                        feature=feat1,
                        correlated_with=feat2,
                        correlation=corr_val,
                    )

        features_after_correlation = list(features_to_keep)

        # Method 2: VIF-based removal
        removed_by_vif = []
        if len(features_after_correlation) > 1:
            X_clean = X[features_after_correlation].copy()
            max_iterations = len(features_after_correlation)

            for _ in range(max_iterations):
                try:
                    # Calculate VIF for all features
                    vif_data = []
                    for i, col in enumerate(X_clean.columns):
                        vif = variance_inflation_factor(X_clean.values, i)
                        vif_data.append({"feature": col, "vif": vif})

                    vif_df = pd.DataFrame(vif_data)

                    # Find highest VIF above threshold
                    high_vif = vif_df[vif_df["vif"] > self.config.vif_threshold]

                    if len(high_vif) == 0:
                        break

                    # Remove feature with highest VIF
                    worst_idx = high_vif["vif"].idxmax()
                    worst_feature = high_vif.loc[worst_idx, "feature"]
                    worst_vif = high_vif.loc[worst_idx, "vif"]

                    X_clean = X_clean.drop(columns=[worst_feature])
                    removed_by_vif.append(worst_feature)
                    logger.debug(
                        "Removed feature due to VIF",
                        feature=worst_feature,
                        vif=worst_vif,
                    )

                except Exception as e:
                    logger.warning("VIF calculation failed", error=str(e))
                    break

            final_features = list(X_clean.columns)
        else:
            final_features = features_after_correlation

        logger.info(
            "Multicollinearity handling complete",
            original=len(features),
            removed_correlation=len(removed_by_correlation),
            removed_vif=len(removed_by_vif),
            final=len(final_features),
        )

        return final_features, removed_by_correlation, removed_by_vif, high_corr_pairs

    def _feature_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], pd.DataFrame]:
        """Execute comprehensive feature selection using multiple methods.

        Args:
            X: Feature matrix (after multicollinearity removal)
            y: Target variable

        Returns:
            Tuple of (consensus_features, selection_summary_df)
        """
        if len(X.columns) == 0:
            return [], pd.DataFrame()

        features = list(X.columns)
        selection_results = {}

        # Method 1: Statistical significance
        stat_significant = []
        for feature in features:
            try:
                corr, p_val = stats.pearsonr(X[feature], y)
                if (
                    abs(corr) > self.config.min_correlation
                    and p_val < self.config.alpha
                ):
                    stat_significant.append(feature)
            except Exception:
                continue

        selection_results["statistical"] = stat_significant
        logger.debug("Statistical selection", selected=len(stat_significant))

        # Method 2: Mutual information
        try:
            k = min(self.config.top_k_features, len(features))
            selector_mi = SelectKBest(mutual_info_classif, k=k)
            selector_mi.fit(X, y)
            mi_selected = X.columns[selector_mi.get_support()].tolist()
            selection_results["mutual_info"] = mi_selected
            logger.debug("Mutual information selection", selected=len(mi_selected))
        except Exception as e:
            logger.warning("Mutual information selection failed", error=str(e))
            selection_results["mutual_info"] = []

        # Method 3: Random Forest importance
        try:
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state, n_jobs=-1
            )
            rf.fit(X, y)

            importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": rf.feature_importances_}
            )
            mean_importance = importance_df["importance"].mean()
            rf_selected = importance_df[importance_df["importance"] > mean_importance][
                "feature"
            ].tolist()

            selection_results["random_forest"] = rf_selected
            logger.debug("Random forest selection", selected=len(rf_selected))
        except Exception as e:
            logger.warning("Random forest selection failed", error=str(e))
            selection_results["random_forest"] = []

        # Create consensus
        consensus_features = self._create_consensus(selection_results, features)

        # Create summary DataFrame
        summary_df = self._create_selection_summary(selection_results, features)

        logger.info(
            "Feature selection complete", consensus_features=len(consensus_features)
        )

        return consensus_features, summary_df

    def _create_consensus(
        self, selection_results: Dict[str, List[str]], all_features: List[str]
    ) -> List[str]:
        """Create consensus feature selection based on voting.

        Args:
            selection_results: Results from different selection methods
            all_features: All available features

        Returns:
            List of consensus features
        """
        # Count votes for each feature
        feature_votes = {feature: 0 for feature in all_features}

        for method, selected_features in selection_results.items():
            for feature in selected_features:
                if feature in feature_votes:
                    feature_votes[feature] += 1

        # Select features with minimum votes
        min_votes = min(
            self.config.min_consensus_votes, max(1, len(selection_results) // 2)
        )
        consensus_features = [
            feature for feature, votes in feature_votes.items() if votes >= min_votes
        ]

        # If no consensus, take top voted features (at least top 5)
        if len(consensus_features) == 0:
            sorted_features = sorted(
                feature_votes.items(), key=lambda x: x[1], reverse=True
            )
            consensus_features = [f for f, v in sorted_features[:5] if v > 0]

        logger.debug(
            "Consensus selection",
            min_votes=min_votes,
            selected=len(consensus_features),
        )

        return consensus_features

    def _create_selection_summary(
        self, selection_results: Dict[str, List[str]], all_features: List[str]
    ) -> pd.DataFrame:
        """Create summary DataFrame of feature selection.

        Args:
            selection_results: Results from different selection methods
            all_features: All available features

        Returns:
            DataFrame summarizing feature selection across methods
        """
        summary_data = []

        for feature in all_features:
            row: Dict[str, Any] = {"feature": feature}
            vote_count = 0

            for method, selected_features in selection_results.items():
                is_selected = feature in selected_features
                row[f"{method}_selected"] = is_selected
                if is_selected:
                    vote_count += 1

            row["total_votes"] = vote_count
            row["vote_percentage"] = (
                vote_count / len(selection_results) * 100 if selection_results else 0
            )
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("total_votes", ascending=False)

        return summary_df


# ==================== Convenience Functions ====================


def create_feature_analysis_pipeline(
    correlation_threshold: float = 0.9,
    vif_threshold: float = 15.0,
    save_plots: bool = True,
    plot_dir: str = "results/feature_analysis",
    random_state: int = 42,
) -> FeatureAnalysisPipeline:
    """Create feature analysis pipeline with common defaults.

    Args:
        correlation_threshold: Threshold for correlation-based multicollinearity
        vif_threshold: Threshold for VIF-based multicollinearity
        save_plots: Whether to save plots to disk
        plot_dir: Directory to save plots
        random_state: Random seed for reproducibility

    Returns:
        Configured FeatureAnalysisPipeline instance
    """
    config = FeatureAnalysisConfig(
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold,
        save_plots=save_plots,
        plot_dir=plot_dir,
        random_state=random_state,
    )
    return FeatureAnalysisPipeline(config)
