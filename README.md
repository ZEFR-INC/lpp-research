## 🎯 Overview

This library provides a complete, scientifically-grounded pipeline for uncertainty quantification in LLM predictions:

1. **Feature Engineering**: Extract 18+ uncertainty signals from LLM logprobs (entropy, confidence, prediction gaps)
2. **Feature Analysis & Selection**: Perform exploratory data analysis, detect multicollinearity, and select optimal features using statistical methods
3. **Data Balancing**: Handle imbalanced datasets with multiple undersampling strategies
4. **Meta-Model Training**: Train Ridge and XGBoost classifiers with hyperparameter optimization and threshold tuning

### Basic Usage

```python
from src import (
    create_feature_pipeline,
    create_feature_analysis_pipeline,
    create_downsampling_pipeline,
    create_meta_model_pipeline,
)
import pandas as pd

# 1. Extract uncertainty features from LLM outputs
llm_outputs = [
    {
        "logprobs_data": [{
            "top_logprobs": [
                {"token": "NO", "logprob": -0.1, "linear_prob": 0.9},
                {"token": "YES", "logprob": -2.3, "linear_prob": 0.1}
            ]
        }]
    },
    # ... more records
]

feature_pipeline = create_feature_pipeline()
features = feature_pipeline.transform(llm_outputs)
print(f"Extracted {len(features.columns)} features")

# 2. Analyze features and select optimal subset
labels = pd.Series([0, 1, 0, 1, ...])  # 0=incorrect, 1=correct

analysis_pipeline = create_feature_analysis_pipeline(
    correlation_threshold=0.95,
    vif_threshold=30.0,
    save_plots=True
)

analysis_result = analysis_pipeline.analyze(X=features, y=labels)
print(f"Selected {len(analysis_result.selected_features)} features")

# Use recommended features
features = features[analysis_result.selected_features]

# 3. Balance your dataset
downsample_pipeline = create_downsampling_pipeline(
    train_ratio=1/6,  # 1:6 ratio in training
    test_ratio=1/3,   # 1:3 ratio in test
)

X_train, X_test, y_train, y_test, report = downsample_pipeline.fit_resample(
    features, labels
)

# 4. Train meta-models
meta_pipeline = create_meta_model_pipeline(
    cost_misclassification=0.94,
    cost_human_review=0.64,
    n_cv_folds=5
)

results_df = meta_pipeline.train_all(X_train, y_train, X_test, y_test)

# 5. Get best model and make predictions
best_model = meta_pipeline.get_best_result(metric="test_f1")
print(f"Best F1: {best_model.test_f1:.3f}")

# Save model for deployment
meta_pipeline.save_model(best_model, "best_model.pkl")

# Later: load and use
loaded_model = meta_pipeline.load_model("best_model.pkl")
predictions = loaded_model.predict_proba(X_new)[:, 1]
```

## 📊 Features

### Feature Engineering (18+ Features)

**Entropy-based features** quantify prediction uncertainty:
- Shannon entropy and normalized entropy
- Effective number of choices
- Model confidence scores

**Confidence features** measure prediction strength:
- Maximum softmax probability
- P1-P2 gap (difference between top two predictions)
- Confidence ratios

**Logprob gap features** analyze prediction margins:
- Raw and normalized logprob differences

All features available with YES/NO token filtering.

### Feature Analysis & Selection

**Exploratory Data Analysis**:
- Descriptive statistics for all features
- Target correlation analysis with statistical significance testing
- Comprehensive visualizations (heatmaps, distributions, correlation plots)

**Multicollinearity Detection**:
- Correlation-based removal (removes one feature from highly correlated pairs)
- VIF (Variance Inflation Factor) analysis for detecting multicollinearity
- Configurable thresholds for both methods

**Feature Selection Methods**:
- Statistical significance testing (Pearson correlation with p-values)
- Mutual information scoring
- Recursive Feature Elimination (RFE)
- Random Forest feature importance
- Consensus voting across all methods for robust selection

**Output**: Recommended feature subset, removal reports, and visualizations saved to `results/feature_analysis/`

### Balanced Sampling Strategies

- **RANDOM**: Fast random undersampling
- **TOMEK**: Removes Tomek Links (boundary cleaning)
- **ENN**: Edited Nearest Neighbors (noise removal)
- **HYBRID**: Combined Tomek + ENN (most thorough)

### Meta-Model Training

**Supported Models**:
- Ridge Classifier with CalibratedClassifierCV
- XGBoost with optimized hyperparameters

**Training Features**:
- GridSearchCV and RandomizedSearchCV
- Business cost function optimization
- Threshold tuning on validation set
- Cross-validation with stratified folds
- Comprehensive evaluation metrics:
  - F1 Score (overall, macro, per-class)
  - ROC-AUC and Average Precision
  - Balanced Accuracy
  - Log Loss
  - Confusion Matrix
  - Custom cost metrics

## 🏗️ Architecture

### Project Structure

```
lpp-research/
├── src/
│   ├── __init__.py              # Public API exports
│   ├── downsampling.py          # Balanced undersampling
│   ├── evaluation_plots.pt      # Visualization utilities for model evaluation metrics.
│   ├── feature_analysis.py      # EDA, multicollinearity, feature selection
│   ├── feature_engineering.py   # Logprob feature extraction
│   ├── inference.py   # Inference script for LLM uncertainty meta-model.
│   ├── meta_model.py            # Model training & evaluation
│   ├── preprocess_data.py       # preprocess data for the mock_data=
│   └── pipeline.py              # End-to-end pipeline orchestration
├── README.md
└── requirements.txt

## 📖 Documentation

### Input Data Format

The library expects LLM outputs in this format:

```json
{
  "title": "Sample Video",
  "logprobs_data": [
    {
      "question": "Does the content involve adult themes?",
      "answer": "NO",
      "top_logprobs": [
        {"token": "NO", "logprob": -0.105, "linear_prob": 0.9},
        {"token": "YES", "logprob": -2.303, "linear_prob": 0.1},
        {"token": "Maybe", "logprob": -4.605, "linear_prob": 0.01}
      ],
      "error": null
    }
  ],
  "error": null
}
```

**Example Data**: A complete working dataset is provided in `data/mock_data.jsonl` with real LLM predictions that you can use to test the pipeline.

### Running the Pipeline

To run the complete end-to-end pipeline on the example data:

```bash
# Make sure you're in the project directory
cd src

# Run the pipeline
python -m src/pipeline.py
```

The pipeline will:
1. Load and preprocess the example data from `data/`
2. Extract uncertainty features
3. Perform feature analysis and select optimal features (saves plots to `results/feature_analysis/`)
4. Balance the dataset using downsampling
5. Train and evaluate Ridge and XGBoost meta-models
6. Save the best model to `models/best_meta_model.pkl`
7. Generate comprehensive results in `results/`

### Examples

The `src/pipeline.py` file serves as a complete working example demonstrating the entire workflow from data loading to model deployment.

## Scientific Background

### Uncertainty Quantification
- **Entropy-based methods**: Quantify model uncertainty using Shannon entropy
- **Confidence calibration**: Platt scaling and isotonic regression for probability calibration
- **Threshold optimization**: Business cost-aware decision boundary tuning

### Feature Analysis & Selection
- **Multicollinearity detection**: VIF (Variance Inflation Factor) analysis to identify redundant features
- **Statistical testing**: Pearson correlation with significance testing (p-values)
- **Ensemble selection**: Consensus voting across multiple feature selection algorithms (mutual information, RFE, Random Forest)
- **Visualization**: Heatmaps, distribution plots, and correlation analysis for interpretability

### Imbalanced Learning
- **Undersampling strategies**
- **Stratified sampling**: Preserves class distribution in train/test splits
- **Cleaning methods**: Tomek Links and ENN for boundary refinement
