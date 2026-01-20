# UIDAI Hackathon - Aadhaar Analysis Project

## Project Overview

This project analyzes Aadhaar enrolment and update patterns to assess citizen experience and identity stability. It introduces three novel and interpretable indicators to move beyond raw count metrics and provide insights into system friction and identity volatility.

## Key Metrics

1. **Identity Stability Index (ISI)**: Measures the stability of Aadhaar records within a region.
2. **Aadhaar Friction Index (AFI)**: Quantifies the relative effort citizens expend in maintaining Aadhaar details.
3. **Update Dependency**: Identifies regions with a persistent reliance on repeated Aadhaar updates.

## Policy-Aware Analysis

The framework distinguishes between policy-mandated updates (age 15+) and discretionary updates:
- **Policy Update Ratio (PUR)**: 25% of biometric updates identified as mandatory re-enrollment
- **Normalized AFI**: Excludes policy-driven updates for true friction measurement
- **State Compliance Analysis**: Ranks states by policy update adherence

## Features

- Advanced machine learning models (Random Forest, Gradient Boosting) with R² scores up to 0.993
- 16 interactive visualizations including heatmaps, scatter plots, and time-series analysis
- Policy-aware metrics that separate legitimate lifecycle updates from system friction
- Age-cohort analysis to justify natural update patterns
- Economic mobility impact assessment

## Code Structure

```
src/enhanced_phase1_exploration.py
Handles data ingestion, cleaning, preprocessing, normalization, and exploratory analysis of Aadhaar enrolment and update datasets.

src/enhanced_phase2_ml.py
Contains machine learning and analytical models used for advanced insights and predictive extensions of the Aadhaar Friction Index (AFI) and Identity Stability Index (ISI).

src/enhanced_phase3_dashboard.py
Implements interactive visualizations and dashboard components for intuitive interpretation of key metrics and regional comparisons.
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main dashboard script to generate all visualizations:

```bash
python src/enhanced_phase3_dashboard.py
```

For policy-aware analysis:

```bash
python src/policy_aware_analysis.py
```

To run the complete analysis pipeline:

```bash
python main.py
```

## Output Files

All generated visualizations are saved to `outputs/` as interactive HTML files.

## Repository Organization

```
src/enhanced_phase1_exploration.py
Handles data ingestion, cleaning, preprocessing, normalization, and exploratory analysis of Aadhaar enrolment and update datasets.

src/enhanced_phase2_ml.py
Contains machine learning and analytical models used for advanced insights and predictive extensions of the Aadhaar Friction Index (AFI) and Identity Stability Index (ISI).

src/enhanced_phase3_dashboard.py
Implements interactive visualizations and dashboard components for intuitive interpretation of key metrics and regional comparisons.

src/final_enhancement_dashboard.py
Creates age-cohort and economic mobility analysis visualizations.

src/policy_aware_analysis.py
Implements policy-mandated update separation analysis.

outputs/
Stores all generated plots, visualizations, intermediate results, and final analytical outputs used in this report.

main.py
Main execution script to run the complete analysis pipeline.
```

All figures and metrics presented in this submission were generated programmatically using this codebase.

## GitHub Repository

The full implementation is available at:

GitHub Repository: https://github.com/Yashvi2874/uidai