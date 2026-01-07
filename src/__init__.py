"""LLM Uncertainty Quantification Package.

This package provides production-ready tools for quantifying uncertainty in LLM outputs
through feature engineering, balanced sampling, and meta-model training.
"""

from downsampling import (
    DownsamplingConfig,
    DownsamplingPipeline,
    SamplingStrategy,
    create_downsampling_pipeline,
)
from feature_analysis import (
    FeatureAnalysisConfig,
    FeatureAnalysisPipeline,
    FeatureAnalysisResult,
    create_feature_analysis_pipeline,
)
from feature_engineering import (
    ConfidenceFeatureExtractor,
    EntropyFeatureExtractor,
    FeatureEngineeringPipeline,
    LogprobGapFeatureExtractor,
    create_feature_pipeline,
)
from meta_model import (
    ExperimentResult,
    MetaModelConfig,
    MetaModelPipeline,
    create_meta_model_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    # Feature Engineering
    "FeatureEngineeringPipeline",
    "EntropyFeatureExtractor",
    "ConfidenceFeatureExtractor",
    "LogprobGapFeatureExtractor",
    "create_feature_pipeline",
    # Feature Analysis
    "FeatureAnalysisPipeline",
    "FeatureAnalysisConfig",
    "FeatureAnalysisResult",
    "create_feature_analysis_pipeline",
    # Downsampling
    "DownsamplingPipeline",
    "DownsamplingConfig",
    "SamplingStrategy",
    "create_downsampling_pipeline",
    # Meta Model
    "MetaModelPipeline",
    "MetaModelConfig",
    "ExperimentResult",
    "create_meta_model_pipeline",
]
