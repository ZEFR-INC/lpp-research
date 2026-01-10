"""Meta-model training and evaluation for LLM uncertainty quantification.

This module provides a complete ML pipeline for training and evaluating meta-models
that predict LLM prediction correctness based on uncertainty features.

References:
    - Platt (1999): Probabilistic Outputs for SVMs
    - Niculescu-Mizil & Caruana (2005): Predicting Good Probabilities with Supervised Learning
"""

import numpy as np
import pandas as pd
import pickle
import structlog
import warnings
import xgboost as xgb

from dataclasses import dataclass
from pathlib import Path
from scipy.stats import loguniform, randint, uniform
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import Any, Dict, List, Optional, Tuple

logger = structlog.getLogger(__name__)


# ==================== Utility Functions ====================


def _handle_extreme_values(X: np.ndarray) -> np.ndarray:
    """Replace extreme values with clipped versions."""
    X_copy = X.copy()
    # Clip to reasonable range
    X_copy = np.clip(X_copy, -1e10, 1e10)
    # Replace any remaining inf with large values
    X_copy[np.isinf(X_copy)] = np.nan
    return X_copy


def _has_predict_proba(estimator: Any) -> bool:
    """Check if estimator has predict_proba method."""
    return hasattr(estimator, "predict_proba") and callable(
        getattr(estimator, "predict_proba")
    )


def _get_continuous_scores(estimator: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Get continuous scores from estimator (proba or decision function).

    Args:
        estimator: Fitted estimator
        X: Feature matrix

    Returns:
        Continuous scores for positive class, or None if unavailable
    """
    if _has_predict_proba(estimator):
        return estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


# ==================== Configuration ====================


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    experiment_name: str
    feature_selector: str
    best_params: Dict[str, Any]
    best_cv_score: float
    cv_scores_std: float
    cv_refit_metric: str
    test_cost: float
    cv_f1: float
    cv_f1_neg_class: float
    cv_f1_macro: float
    cv_balanced_accuracy: float
    cv_roc_auc: float
    cv_average_precision_neg_class: float
    selected_features_count: int
    selected_features_list: List[str]
    test_f1: float
    test_f1_macro: float
    test_f1_neg_class: float
    test_balanced_accuracy: float
    test_roc_auc: float
    test_average_precision: float
    test_log_loss: float
    test_confusion_matrix: np.ndarray
    full_classification_report: str
    optimal_threshold: float
    validation_cost: float
    test_probabilities: Optional[np.ndarray] = None
    test_ground_truth: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    fitted_model: Optional[Pipeline] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    tn: Optional[int] = None
    fn: Optional[int] = None


@dataclass
class MetaModelConfig:
    """Configuration for meta-model training pipeline.

    Attributes:
        cost_misclassification: Cost of false positive (predicting correct when wrong)
        cost_human_review: Cost of false negative (predicting wrong when correct)
        n_cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level for training
    """

    cost_misclassification: float = 1
    cost_human_review: float = 0.64
    n_cv_folds: int = 3
    random_state: int = 42
    n_jobs: int = 1
    verbose: int = 0


# ==================== Main Pipeline ====================


class MetaModelPipeline:
    """Production-grade pipeline for training and evaluating meta-models.

    This pipeline implements the complete workflow from the notebook:
    1. Define experiments (Ridge, XGBoost with hyperparameter grids)
    2. Train models with cross-validation
    3. Optimize decision thresholds for business cost
    4. Evaluate on test set with comprehensive metrics
    5. Save/load trained models

    Example:
        ```python
        from meta_model import MetaModelPipeline, MetaModelConfig

        # Configure pipeline
        config = MetaModelConfig(
            cost_misclassification=0.94,
            cost_human_review=0.64,
            n_cv_folds=5
        )
        pipeline = MetaModelPipeline(config)

        # Train models
        results_df = pipeline.train_all(X_train, y_train, X_test, y_test)

        # Get best model
        best_result = pipeline.get_best_result(metric="test_f1")
        predictions = best_result.fitted_model.predict_proba(X_new)[:, 1]
        ```
    """

    def __init__(self, config: Optional[MetaModelConfig] = None):
        """Initialize meta-model pipeline.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or MetaModelConfig()
        self.results: List[ExperimentResult] = []

        # Calculate cost ratio
        self.fp_to_fn_cost_ratio = (
            self.config.cost_misclassification / self.config.cost_human_review
        )

        logger.info(
            "Initialized meta-model pipeline",
            cost_ratio=self.fp_to_fn_cost_ratio,
            cv_folds=self.config.n_cv_folds,
        )

    def _business_cost_scorer(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate business cost for predictions.

        Args:
            y_true: True labels (0=incorrect, 1=correct)
            y_pred: Predicted labels

        Returns:
            Total business cost (lower is better)
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Costs:
        # TP (predict correct, is correct): 0 cost
        # FP (predict correct, is wrong): misclassification cost
        # TN (predict wrong, is wrong): human review cost - misclassification cost
        # FN (predict wrong, is correct): human review cost

        tp_cost = 0.0
        fp_cost = fp * self.config.cost_misclassification
        tn_cost = tn * (
            self.config.cost_human_review - self.config.cost_misclassification
        )
        fn_cost = fn * self.config.cost_human_review

        total_cost = tp_cost + fp_cost + tn_cost + fn_cost
        return total_cost

    def _get_scorers(self) -> Dict[str, Any]:
        """Get dictionary of scoring functions for cross-validation.

        Returns:
            Dictionary mapping metric names to scorer objects
        """
        from sklearn.metrics import make_scorer

        return {
            "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "cost": make_scorer(self._business_cost_scorer, greater_is_better=False),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "average_precision_neg_class": make_scorer(
                average_precision_score, pos_label=0, needs_proba=True
            ),
            "f1_neg_class": make_scorer(f1_score, pos_label=0, zero_division=0),
        }

    def _create_pipeline(
        self, model: ClassifierMixin, feature_selector: Any = "passthrough"
    ) -> Pipeline:
        """Create sklearn Pipeline with preprocessing and model.

        Args:
            model: Classifier instance
            feature_selector: Feature selection transformer or "passthrough"

        Returns:
            Configured Pipeline
        """
        steps = [
            (
                "extreme_value_handler",
                FunctionTransformer(_handle_extreme_values, validate=False),
            ),
            ("scaler", StandardScaler()),
        ]

        if feature_selector != "passthrough":
            steps.append(("feature_selector", feature_selector))

        steps.append(("model", model))

        return Pipeline(steps)

    def define_experiments(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[Dict[str, Any]]:
        """Define experiment configurations for different models.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            List of experiment configurations
        """
        experiments = []

        # Calculate class imbalance for weight tuning
        # Base scale weight values around the cost ratio
        base_scale_weight = (
            self.config.cost_human_review / self.config.cost_misclassification
        )
        scale_weight_values = [
            base_scale_weight * 0.5,
            base_scale_weight * 0.8,
            base_scale_weight * 1.0,
            base_scale_weight * 1.2,
            base_scale_weight * 1.5,
            base_scale_weight * 2.0,
        ]
        scale_weight_values = [max(0.1, w) for w in scale_weight_values]

        # ===== Ridge Classifier with Calibration =====
        experiments.append(
            {
                "name": "Ridge_f1",
                "feature_selector": "passthrough",
                "model": CalibratedClassifierCV(
                    estimator=RidgeClassifier(random_state=self.config.random_state),
                    method="sigmoid",
                    cv=self.config.n_cv_folds,
                ),
                "search_type": "grid",
                "param_grid": {
                    "model__estimator__alpha": [0.1, 1.0, 10.0, 100.0],
                    "model__estimator__solver": ["auto", "lsqr"],
                    "model__estimator__tol": [1e-6, 1e-5, 1e-4, 1e-3],
                    "model__estimator__max_iter": [200, 500, 1000, 1500],
                    "model__estimator__class_weight": [
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 0.8},
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 1.0},
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 1.2},
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 1.5},
                        "balanced",
                        None,
                    ],
                    "model__method": ["sigmoid", "isotonic"],
                },
                "refit_metric": "f1",
            }
        )

        # ===== XGBoost Classifier =====
        experiments.append(
            {
                "name": "XGBoost_f1",
                "feature_selector": "passthrough",
                "model": xgb.XGBClassifier(
                    random_state=self.config.random_state,
                    objective="binary:logistic",
                    eval_metric="auc",
                    verbosity=0,
                    n_jobs=self.config.n_jobs,
                ),
                "search_type": "random",
                "param_distributions": {
                    "model__n_estimators": randint(50, 200),
                    "model__max_depth": randint(3, 6),
                    "model__learning_rate": loguniform(0.05, 0.3),
                    "model__subsample": uniform(0.7, 0.3),
                    "model__colsample_bytree": uniform(0.7, 0.3),
                    "model__scale_pos_weight": scale_weight_values,
                    "model__reg_alpha": loguniform(0.1, 5.0),
                    "model__reg_lambda": loguniform(0.5, 20.0),
                    "model__gamma": loguniform(1e-3, 0.1),
                    "model__min_child_weight": randint(3, 8),
                },
                "n_iter": 150,
                "refit_metric": "f1",
            }
        )

        # Add additional experiment variations
        experiments.append(
            {
                "name": "Ridge_cost_optimized",
                "feature_selector": "passthrough",
                "model": CalibratedClassifierCV(
                    estimator=RidgeClassifier(random_state=self.config.random_state),
                    method="sigmoid",
                    cv=self.config.n_cv_folds,
                ),
                "search_type": "grid",
                "param_grid": {
                    "model__estimator__alpha": [0.1, 1.0, 10.0],
                    "model__estimator__solver": ["auto"],
                    "model__estimator__tol": [1e-5, 1e-4],
                    "model__estimator__max_iter": [1000],
                    "model__estimator__class_weight": [
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 1.0},
                        {1: 1.0, 0: self.fp_to_fn_cost_ratio * 1.5},
                        "balanced",
                    ],
                    "model__method": ["sigmoid"],
                },
                "refit_metric": "cost",
            }
        )

        logger.info("Defined experiments", count=len(experiments))
        return experiments

    def _find_optimal_threshold(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[float, float]:
        """Find optimal probability threshold to minimize business cost.

        Args:
            model: Fitted model pipeline
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (best_threshold, minimum_cost)
        """
        if not _has_predict_proba(model):
            logger.warning("Model lacks predict_proba, using default threshold")
            cost_at_default = self._business_cost_scorer(y_val, model.predict(X_val))
            return 0.5, cost_at_default

        # Get predicted probabilities
        y_val_probs = model.predict_proba(X_val)[:, 1]

        # Generate candidate thresholds between 0.35 to 0.7
        thresholds = np.linspace(0.35, 0.7, num=100)

        min_cost = float("inf")
        best_threshold = 0.5

        for t in thresholds:
            y_val_pred = (y_val_probs >= t).astype(int)
            cost = self._business_cost_scorer(y_val, y_val_pred)

            if cost < min_cost:
                min_cost = cost
                best_threshold = t

        logger.info(
            "Optimal threshold found",
            threshold=round(float(best_threshold), 4),
            cost=round(float(min_cost), 4),
        )

        return float(best_threshold), float(min_cost)

    def _extract_selected_features(
        self, fitted_pipeline: Pipeline, all_features: pd.Index
    ) -> List[str]:
        """Extract list of selected features from fitted pipeline.

        Args:
            fitted_pipeline: Fitted Pipeline object
            all_features: Original feature names

        Returns:
            List of selected feature names
        """
        try:
            fs = fitted_pipeline.named_steps.get("feature_selector")
            if fs == "passthrough" or fs is None:
                return list(all_features)

            if hasattr(fs, "get_support"):
                mask = fs.get_support()
                if isinstance(mask, (list, np.ndarray)) and len(mask) == len(
                    all_features
                ):
                    return all_features[mask].tolist()

            return []
        except Exception as e:
            logger.warning("Could not extract selected features", error=str(e))
            return []

    def train_single_experiment(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[ExperimentResult, ExperimentResult]:
        """Train a single experiment configuration.

        Args:
            config: Experiment configuration dictionary
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (default_threshold_result, tuned_threshold_result)
        """
        logger.info("Starting experiment", name=config["name"])
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=self.config.random_state,
        )
        # Create pipeline
        pipeline = self._create_pipeline(
            clone(config["model"]), config["feature_selector"]
        )

        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Setup search
        search_params = {
            "estimator": pipeline,
            "scoring": self._get_scorers(),
            "refit": config["refit_metric"],
            "cv": cv,
            "n_jobs": self.config.n_jobs,
            "verbose": self.config.verbose,
        }

        if config.get("search_type") == "random":
            search_params["param_distributions"] = config.get("param_distributions", {})
            search_params["n_iter"] = config.get("n_iter", 100)
            search_params["random_state"] = self.config.random_state
            SearchClass = RandomizedSearchCV
        else:
            search_params["param_grid"] = config.get("param_grid", {})
            SearchClass = GridSearchCV

        # Fit model on inner train split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            search = SearchClass(**search_params)
            search.fit(X_tr, y_tr)

        if not hasattr(search, "best_estimator_") or search.best_estimator_ is None:
            raise ValueError("Search did not complete; no best estimator found")

        best_pipe = search.best_estimator_

        # -------- Variant A: Default threshold (0.5)
        if _has_predict_proba(best_pipe):
            default_val_pred = (best_pipe.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
            default_validation_cost = self._business_cost_scorer(
                y_val, default_val_pred
            )
            default_threshold = 0.5
        else:
            default_val_pred = best_pipe.predict(X_val)
            default_validation_cost = self._business_cost_scorer(
                y_val, default_val_pred
            )
            default_threshold = 0.5

        cvres = search.cv_results_
        idx = search.best_index_

        res_default = self._make_result(
            name=f"{config['name']}__thresh=default",
            config=config,
            best_pipe=best_pipe,
            refit_metric=config["refit_metric"],
            cvres=cvres,
            idx=idx,
            X_test=X_test,
            y_test=y_test,
            threshold=default_threshold,
            validation_cost=float(default_validation_cost),
            X_train_cols=X_train.columns if hasattr(X_train, "columns") else None,
        )

        # -------- Variant B: Tuned threshold
        opt_threshold, validation_cost = self._find_optimal_threshold(
            best_pipe, X_val, y_val
        )

        res_tuned = self._make_result(
            name=f"{config['name']}__thresh=tuned",
            config=config,
            best_pipe=best_pipe,
            refit_metric=config["refit_metric"],
            cvres=cvres,
            idx=idx,
            X_test=X_test,
            y_test=y_test,
            threshold=float(opt_threshold),
            validation_cost=float(validation_cost),
            X_train_cols=X_train.columns if hasattr(X_train, "columns") else None,
        )

        logger.info(
            "Experiment complete",
            name=config["name"],
            default_val_cost=round(float(default_validation_cost), 4),
            tuned_val_cost=round(float(validation_cost), 4),
            tuned_threshold=round(float(opt_threshold), 4),
        )

        return res_default, res_tuned

    def _make_result(
        self,
        name: str,
        config: Dict[str, Any],
        best_pipe: Pipeline,
        refit_metric: str,
        cvres: Dict,
        idx: int,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float,
        validation_cost: float,
        X_train_cols: Optional[pd.Index],
    ) -> ExperimentResult:
        """Create ExperimentResult from training outputs.

        Args:
            name: Experiment name
            config: Experiment configuration
            best_pipe: Fitted pipeline
            refit_metric: Metric used for refitting
            cvres: Cross-validation results
            idx: Index of best result
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold
            validation_cost: Validation cost at threshold
            X_train_cols: Training column names

        Returns:
            ExperimentResult object
        """
        # Get predictions
        test_proba = None
        if _has_predict_proba(best_pipe):
            test_proba = best_pipe.predict_proba(X_test)[:, 1]
            y_pred = (test_proba >= threshold).astype(int)
            test_logloss = float(log_loss(y_test, test_proba))
        else:
            y_pred = best_pipe.predict(X_test)
            test_logloss = float("nan")

        # Get continuous scores for AUC/AP
        scores = _get_continuous_scores(best_pipe, X_test)

        # Calculate metrics (with zero_division=0 to suppress sklearn warnings)
        test_cost = float(self._business_cost_scorer(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
        test_f1_macro = float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        test_f1_neg_class = float(
            f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        )
        test_bal_acc = float(balanced_accuracy_score(y_test, y_pred))

        if scores is not None:
            try:
                test_roc_auc = float(roc_auc_score(y_test, scores))
            except Exception:
                test_roc_auc = float("nan")
            try:
                test_ap = float(average_precision_score(y_test, scores))
            except Exception:
                test_ap = float("nan")
        else:
            test_roc_auc = float("nan")
            test_ap = float("nan")

        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(y_test, y_pred, zero_division=0)

        # Extract CV metrics
        def _get_cv(key: str, default=np.nan):
            return float(cvres.get(key, [default])[idx])

        cv_best = _get_cv(f"mean_test_{refit_metric}")
        cv_std = _get_cv(f"std_test_{refit_metric}")
        cv_f1 = _get_cv("mean_test_f1")
        cv_f1_neg_class = _get_cv("mean_test_f1_neg_class")
        cv_f1_macro = _get_cv("mean_test_f1_macro")
        cv_bal = _get_cv("mean_test_balanced_accuracy")
        cv_roc = _get_cv("mean_test_roc_auc")
        cv_ap_neg_class = _get_cv("mean_test_average_precision_neg_class")

        # Extract selected features
        selected_features = self._extract_selected_features(
            best_pipe, X_test.columns if hasattr(X_test, "columns") else X_train_cols
        )
        if cm.shape == (2, 2):
            # Standard 2x2 confusion matrix: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where only one class is present
            tn, fp, fn, tp = 0, 0, 0, 0
        return ExperimentResult(
            experiment_name=name,
            feature_selector=(
                type(config["feature_selector"]).__name__
                if config["feature_selector"] != "passthrough"
                else "passthrough"
            ),
            best_params=cvres["params"][idx] if "params" in cvres else {},
            best_cv_score=cv_best,
            cv_scores_std=cv_std,
            cv_refit_metric=refit_metric,
            test_cost=test_cost,
            cv_f1=cv_f1,
            cv_f1_neg_class=cv_f1_neg_class,
            cv_f1_macro=cv_f1_macro,
            cv_balanced_accuracy=cv_bal,
            cv_roc_auc=cv_roc,
            cv_average_precision_neg_class=cv_ap_neg_class,
            selected_features_count=len(selected_features),
            selected_features_list=selected_features,
            test_f1=test_f1,
            test_f1_neg_class=test_f1_neg_class,
            test_f1_macro=test_f1_macro,
            test_balanced_accuracy=test_bal_acc,
            test_roc_auc=test_roc_auc,
            test_average_precision=test_ap,
            test_log_loss=test_logloss,
            test_confusion_matrix=cm,
            full_classification_report=report,
            optimal_threshold=threshold,
            validation_cost=validation_cost,
            test_probabilities=test_proba,
            test_predictions=y_pred,
            test_ground_truth=y_test.values,
            test_indices=X_test.index.values if hasattr(X_test, "index") else None,
            fitted_model=best_pipe,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train all experiment configurations.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with results from all experiments
        """
        logger.info(
            "Starting training pipeline",
            train_samples=len(y_train),
            test_samples=len(y_test),
            features=X_train.shape[1],
        )

        experiments = self.define_experiments(X_train, y_train)

        for config in experiments:
            try:
                res_default, res_tuned = self.train_single_experiment(
                    config, X_train, y_train, X_test, y_test
                )
                self.results.extend([res_tuned])
            except Exception as e:
                logger.error(
                    "Experiment failed",
                    name=config["name"],
                    error=str(e),
                )

        logger.info("Training complete", total_results=len(self.results))

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis.

        Returns:
            DataFrame with all experiment results
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append(
                {
                    "experiment_name": r.experiment_name,
                    "feature_selector": r.feature_selector,
                    "selected_features_count": r.selected_features_count,
                    "best_cv_score": r.best_cv_score,
                    "cv_scores_std": r.cv_scores_std,
                    "cv_refit_metric": r.cv_refit_metric,
                    "test_cost": r.test_cost,
                    "cv_f1": r.cv_f1,
                    "cv_f1_neg_class": r.cv_f1_neg_class,
                    "cv_f1_macro": r.cv_f1_macro,
                    "cv_roc_auc": r.cv_roc_auc,
                    "cv_balanced_accuracy": r.cv_balanced_accuracy,
                    "test_f1": r.test_f1,
                    "test_f1_neg_class": r.test_f1_neg_class,
                    "test_f1_macro": r.test_f1_macro,
                    "test_roc_auc": r.test_roc_auc,
                    "test_average_precision": r.test_average_precision,
                    "test_balanced_accuracy": r.test_balanced_accuracy,
                    "test_log_loss": r.test_log_loss,
                    "optimal_threshold": r.optimal_threshold,
                    "validation_cost": r.validation_cost,
                    "tp": r.tp,
                    "fp": r.fp,
                    "tn": r.tn,
                    "fn": r.fn,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("test_f1", ascending=False).reset_index(drop=True)

    def get_best_result(
        self, metric: str = "test_f1", ascending: bool = False
    ) -> Optional[ExperimentResult]:
        """Get best result by specified metric.

        Args:
            metric: Metric to optimize (e.g., 'test_f1', 'test_cost')
            ascending: Whether lower is better

        Returns:
            Best ExperimentResult or None if no results
        """
        if not self.results:
            return None

        sorted_results = sorted(
            self.results, key=lambda r: getattr(r, metric), reverse=not ascending
        )
        return sorted_results[0]

    def save_model(
        self, result: ExperimentResult, filepath: Path, save_metadata: bool = True
    ) -> None:
        """Save trained model to disk.

        Args:
            result: ExperimentResult containing fitted model
            filepath: Path to save model
            save_metadata: Whether to save metadata alongside model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(filepath, "wb") as f:
            pickle.dump(result.fitted_model, f)

        logger.info("Model saved", path=str(filepath))

        # Save metadata
        if save_metadata:
            metadata_path = filepath.with_suffix(".json")
            metadata = {
                "experiment_name": result.experiment_name,
                "optimal_threshold": result.optimal_threshold,
                "test_f1": result.test_f1,
                "test_cost": result.test_cost,
                "selected_features": result.selected_features_list,
            }

            import json

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Metadata saved", path=str(metadata_path))

    def load_model(self, filepath: Path) -> Pipeline:
        """Load trained model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded Pipeline object
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded", path=str(filepath))
        return model


# ==================== Convenience Functions ====================


def create_meta_model_pipeline(
    cost_misclassification: float = 0.94,
    cost_human_review: float = 0.64,
    n_cv_folds: int = 3,
    random_state: int = 42,
) -> MetaModelPipeline:
    """Create meta-model pipeline with custom configuration.

    Args:
        cost_misclassification: Cost of false positive
        cost_human_review: Cost of false negative
        n_cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Configured MetaModelPipeline
    """
    config = MetaModelConfig(
        cost_misclassification=cost_misclassification,
        cost_human_review=cost_human_review,
        n_cv_folds=n_cv_folds,
        random_state=random_state,
    )
    return MetaModelPipeline(config)
