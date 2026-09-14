"""
Adaptation Guide - How to quickly adapt the pipeline for new data (within 5 hours).

This module provides utilities and templates for fast adaptation.
"""

import pandas as pd
import numpy as np
from pipeline import Config, run_pipeline


# ============================================================================
# TEMPLATE 1: Electricity Price Prediction
# ============================================================================

class ElectricityPriceConfig(Config):
    DATA_PATH = "data/electricity_prices.csv"
    TARGET = "price"
    TIME_COL = "datetime"
    PROBLEM_TYPE = "regression"
    PRIMARY_MODEL = "ridge"
    MODELS = ["linear", "ridge", "lasso", "elastic_net"]
    NORMALIZATION = "standard"
    OUTLIER_METHOD = "clip"


# ============================================================================
# TEMPLATE 2: Vessel/Ship Price Prediction
# ============================================================================

class VesselPriceConfig(Config):
    DATA_PATH = "data/vessel_prices.csv"  # replace with actual data
    TARGET = "price"  # adjust to actual target column name
    TIME_COL = None  # likely cross-sectional, not time-series
    PROBLEM_TYPE = "regression"
    PRIMARY_MODEL = "ridge"
    MODELS = ["linear", "ridge", "lasso", "elastic_net"]
    NORMALIZATION = "standard"
    OUTLIER_METHOD = "clip"
    CATEGORICAL_COLS = ["vessel_type", "material", "condition", "location"]
    ROLLING_WINDOWS = []  # no rolling features for cross-sectional data


# ============================================================================
# TEMPLATE 3: Collateral/Mortgage Valuation
# ============================================================================

class CollateralConfig(Config):
    DATA_PATH = "data/collateral.csv"  # replace with actual data
    TARGET = "property_value"  # adjust to actual target column name
    TIME_COL = None
    PROBLEM_TYPE = "regression"
    PRIMARY_MODEL = "ridge"
    MODELS = ["linear", "ridge", "lasso", "elastic_net"]
    NORMALIZATION = "standard"
    OUTLIER_METHOD = "clip"
    CATEGORICAL_COLS = ["property_type", "occupancy_type", "region", "loan_type"]
    ROLLING_WINDOWS = []


# ============================================================================
# AUTO-DETECT CONFIG
# ============================================================================

def auto_detect_config(data_path, target_col=None, time_col=None, logger=None):
    """
    Automatically detect configuration from a new dataset.
    This is the fastest way to adapt the pipeline.
    """
    # Load a sample
    if data_path.endswith(".csv"):
        df_sample = pd.read_csv(data_path, nrows=1000)
    elif data_path.endswith((".xlsx", ".xls")):
        df_sample = pd.read_excel(data_path, nrows=1000)
    else:
        raise ValueError(f"Unsupported format: {data_path}")

    config = Config()
    config.DATA_PATH = data_path

    # Auto-detect target (look for common target names)
    if target_col is None:
        target_candidates = ["price", "target", "value", "y", "label",
                             "sale_price", "property_value", "amount"]
        for candidate in target_candidates:
            if candidate in df_sample.columns:
                target_col = candidate
                break
        if target_col is None:
            # Use the last numeric column
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[-1] if len(numeric_cols) > 0 else df_sample.columns[-1]

    config.TARGET = target_col

    # Auto-detect time column
    if time_col is None:
        time_candidates = ["datetime", "date", "time", "timestamp", "datetime"]
        for candidate in time_candidates:
            if candidate in df_sample.columns:
                time_col = candidate
                break
    config.TIME_COL = time_col

    # Auto-detect categorical
    config.CATEGORICAL_COLS = [
        col for col in df_sample.columns
        if df_sample[col].dtype == "object" and col != config.TARGET
    ]

    # Auto-detect numeric
    config.NUMERIC_COLS = [
        col for col in df_sample.select_dtypes(include=[np.number]).columns
        if col != config.TARGET
    ]

    # Auto-detect ID column
    id_candidates = ["id", "ID", "Id", "index"]
    for candidate in id_candidates:
        if candidate in df_sample.columns:
            config.ID_COL = candidate
            break

    if logger:
        logger.info(f"Auto-detected config: target={config.TARGET}, time_col={config.TIME_COL}")
        logger.info(f"  Categorical: {config.CATEGORICAL_COLS}")
        logger.info(f"  Numeric count: {len(config.NUMERIC_COLS)}")

    return config


# ============================================================================
# QUICK RUN
# ============================================================================

def quick_run(data_path, target_col=None, time_col=None, output_dir="output"):
    """Quick run with auto-detected configuration."""
    config = auto_detect_config(data_path, target_col, time_col)
    config.OUTPUT_DIR = output_dir
    results = run_pipeline(config)
    return results


if __name__ == "__main__":
    # Example: adapt to new data
    # results = quick_run("data/new_data.csv", target_col="price")
    # print("Done!")

    # Or use a specific template:
    config = ElectricityPriceConfig()
    results = run_pipeline(config)
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    for name, metrics in results["training_results"].items():
        print(f"  {name}: Val RMSE={metrics['val_rmse']:.4f}, Val R2={metrics['val_r2']:.4f}")
    print(f"\n  CV RMSE: {results['cv_results']['mean_rmse']:.4f} +/- {results['cv_results']['std_rmse']:.4f}")
    print(f"  Test R2: {results['sanity_checks']['r2_sanity']['r2']:.4f}")
    print(f"\n  Top 5 features:")
    for _, row in results["feature_importance"].head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.6f}")
