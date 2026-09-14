"""
Quant Research Pipeline - Core Module
A modular, interpretable pipeline for Kaggle-style prediction tasks.
Designed for electricity price / vessel price / collateral valuation problems.

Key design principles:
1. Interpretability first (linear models, transparent features)
2. Modularity (each stage is independent and swappable)
3. Reproducibility (seeded, logged, checkpointed)
4. Fast adaptation (change config, not code)

Author: Quant Research Team
"""

import os
import sys
import json
import time
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration. Modify this to adapt the pipeline to new data."""

    # Data path (use relative path)
    DATA_PATH = "data/electricity_prices.csv"

    # Target variable
    TARGET = "price"

    # Problem type: "regression" or "classification"
    PROBLEM_TYPE = "regression"

    # Time column (if time-series data). Set to None if not applicable.
    TIME_COL = "datetime"

    # ID column (if exists). Set to None if not applicable.
    ID_COL = None

    # Columns to drop (non-informative or leakage)
    DROP_COLS = []

    # Categorical columns to encode
    CATEGORICAL_COLS = []  # auto-detected if empty

    # Numeric columns (auto-detected if empty)
    NUMERIC_COLS = []

    # Test set size (fraction)
    TEST_SIZE = 0.2

    # Validation set size (fraction of train)
    VAL_SIZE = 0.2

    # Cross-validation folds
    CV_FOLDS = 5

    # Random seed
    SEED = 42

    # Rolling window for time-series features (if applicable)
    ROLLING_WINDOWS = [7, 14, 30, 60]

    # Sensitivity analysis windows
    SENSITIVITY_WINDOWS = [7, 14, 20, 30, 60, 90]

    # Normalization method: "standard", "minmax", "robust", "none"
    NORMALIZATION = "standard"

    # Outlier handling: "clip", "remove", "none"
    OUTLIER_METHOD = "clip"
    OUTLIER_THRESHOLD = 3.0  # z-score threshold

    # Missing value strategy
    MISSING_NUMERIC = "median"
    MISSING_CATEGORICAL = "mode"

    # Models to train
    MODELS = ["linear", "ridge", "lasso", "elastic_net"]

    # Primary model (for feature importance and final predictions)
    PRIMARY_MODEL = "ridge"

    # Output directory
    OUTPUT_DIR = "output"

    # Plot style
    PLOT_STYLE = "seaborn-v0_8-whitegrid"
    PLOT_FIGSIZE = (12, 8)
    PLOT_DPI = 150


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config):
    """Setup logging for the pipeline."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(config.OUTPUT_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("pipeline")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config, logger=None):
    """Load data from CSV/Excel file."""
    if logger:
        logger.info(f"Loading data from {config.DATA_PATH}")

    path = config.DATA_PATH
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if config.TIME_COL and config.TIME_COL in df.columns:
        df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL])

    if config.ID_COL and config.ID_COL in df.columns:
        df = df.set_index(config.ID_COL)

    for col in config.DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    if logger:
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Target ({config.TARGET}) stats: mean={df[config.TARGET].mean():.4f}, std={df[config.TARGET].std():.4f}")

    return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def eda(df, config, logger=None):
    """Perform exploratory data analysis and generate summary statistics."""
    if logger:
        logger.info("Starting EDA...")

    results = {}

    # Basic statistics
    results["shape"] = df.shape
    results["dtypes"] = df.dtypes.to_dict()
    results["describe"] = df.describe().to_dict()

    # Missing values
    missing = df.isnull().sum()
    results["missing"] = missing[missing > 0].to_dict()

    # Target statistics
    target = df[config.TARGET]
    results["target_stats"] = {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "median": float(target.median()),
        "skew": float(target.skew()),
        "kurtosis": float(target.kurtosis()),
    }

    # Correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()[config.TARGET].sort_values(ascending=False)
    results["correlation_with_target"] = corr.to_dict()

    # Detect categorical columns
    if not config.CATEGORICAL_COLS:
        config.CATEGORICAL_COLS = [
            col for col in df.columns
            if df[col].dtype == "object" and col != config.TARGET
        ]

    # Detect numeric columns
    if not config.NUMERIC_COLS:
        config.NUMERIC_COLS = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != config.TARGET
        ]

    if logger:
        logger.info(f"Numeric columns ({len(config.NUMERIC_COLS)}): {config.NUMERIC_COLS}")
        logger.info(f"Categorical columns ({len(config.CATEGORICAL_COLS)}): {config.CATEGORICAL_COLS}")
        logger.info(f"Target skewness: {results['target_stats']['skew']:.4f}")
        logger.info(f"Top 5 correlated features: {list(corr.head(6).index[1:])}")

    # Save EDA results
    with open(os.path.join(config.OUTPUT_DIR, "eda_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_time_features(df, config, logger=None):
    """Create time-based features from datetime column."""
    if not config.TIME_COL or config.TIME_COL not in df.columns:
        return df

    if logger:
        logger.info("Creating time features...")

    dt = df[config.TIME_COL]

    # Basic time features
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding (sine/cosine transformation)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Peak hours indicator (for electricity: 8-20)
    df["is_peak_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 20)).astype(int)

    if logger:
        logger.info(f"Added time features. New shape: {df.shape}")

    return df


def create_rolling_features(df, config, target_col=None, logger=None):
    """Create rolling window features for time-series data."""
    if not config.TIME_COL or config.TIME_COL not in df.columns:
        return df

    if logger:
        logger.info("Creating rolling features...")

    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    # Use a key feature for rolling stats (or target if specified)
    if target_col is None:
        target_col = config.TARGET

    if target_col not in df.columns:
        return df

    for window in config.ROLLING_WINDOWS:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window, min_periods=1).max()

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Percentage change
    df[f"{target_col}_pct_change_1"] = df[target_col].pct_change(1)
    df[f"{target_col}_pct_change_24"] = df[target_col].pct_change(24)

    if logger:
        logger.info(f"Added rolling features. New shape: {df.shape}")

    return df


def create_interaction_features(df, config, logger=None):
    """Create interaction features between key numeric variables."""
    if logger:
        logger.info("Creating interaction features...")

    numeric = [col for col in config.NUMERIC_COLS if col in df.columns and col != config.TARGET]

    # Pairwise ratios for top correlated features (limit to avoid explosion)
    if len(numeric) > 1:
        for i in range(min(5, len(numeric))):
            for j in range(i + 1, min(5, len(numeric))):
                col_a, col_b = numeric[i], numeric[j]
                # Avoid division by zero
                df[f"{col_a}_div_{col_b}"] = df[col_a] / (df[col_b].replace(0, np.nan) + 1e-8)
                df[f"{col_a}_times_{col_b}"] = df[col_a] * df[col_b]

    if logger:
        logger.info(f"Added interaction features. New shape: {df.shape}")

    return df


def engineer_features(df, config, logger=None):
    """Master feature engineering function."""
    df = create_time_features(df, config, logger)
    df = create_rolling_features(df, config, logger=logger)
    df = create_interaction_features(df, config, logger)

    # Update column lists
    config.NUMERIC_COLS = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != config.TARGET and col not in config.CATEGORICAL_COLS
    ]

    return df


# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def handle_missing_values(df, config, logger=None):
    """Handle missing values according to config."""
    if logger:
        logger.info("Handling missing values...")

    for col in df.columns:
        if col == config.TARGET:
            continue

        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if col in config.NUMERIC_COLS or df[col].dtype in [np.float64, np.int64, float, int]:
            if config.MISSING_NUMERIC == "median":
                df[col] = df[col].fillna(df[col].median())
            elif config.MISSING_NUMERIC == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif config.MISSING_NUMERIC == "zero":
                df[col] = df[col].fillna(0)
            elif config.MISSING_NUMERIC == "ffill":
                df[col] = df[col].ffill().bfill()
        else:
            if config.MISSING_CATEGORICAL == "mode":
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")
            else:
                df[col] = df[col].fillna("unknown")

        if logger:
            logger.info(f"  {col}: filled {n_missing} missing values")

    return df


def handle_outliers(df, config, logger=None):
    """Handle outliers in numeric features."""
    if config.OUTLIER_METHOD == "none":
        return df

    if logger:
        logger.info(f"Handling outliers (method={config.OUTLIER_METHOD})...")

    numeric_cols = [col for col in config.NUMERIC_COLS if col in df.columns]

    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        if std == 0:
            continue

        z_scores = np.abs((df[col] - mean) / std)
        outlier_mask = z_scores > config.OUTLIER_THRESHOLD

        if outlier_mask.sum() == 0:
            continue

        if config.OUTLIER_METHOD == "clip":
            lower = mean - config.OUTLIER_THRESHOLD * std
            upper = mean + config.OUTLIER_THRESHOLD * std
            df[col] = df[col].clip(lower, upper)
            if logger:
                logger.info(f"  {col}: clipped {outlier_mask.sum()} outliers")
        elif config.OUTLIER_METHOD == "remove":
            df = df[~outlier_mask]
            if logger:
                logger.info(f"  {col}: removed {outlier_mask.sum()} outliers")

    return df


def encode_categorical(df, config, logger=None):
    """Encode categorical variables using one-hot or label encoding."""
    if logger:
        logger.info("Encoding categorical variables...")

    for col in config.CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()

        if n_unique <= 10:
            # One-hot encoding for low cardinality
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=col), dummies], axis=1)
            if logger:
                logger.info(f"  {col}: one-hot encoded ({n_unique} categories -> {dummies.shape[1]} columns)")
        else:
            # Label encoding for high cardinality
            df[col] = pd.Categorical(df[col]).codes
            if logger:
                logger.info(f"  {col}: label encoded ({n_unique} categories)")

    return df


def normalize_features(X_train, X_val, X_test, config, logger=None):
    """Normalize features using the specified method."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if config.NORMALIZATION == "none":
        return X_train, X_val, X_test

    if logger:
        logger.info(f"Normalizing features (method={config.NORMALIZATION})...")

    if config.NORMALIZATION == "standard":
        scaler = StandardScaler()
    elif config.NORMALIZATION == "minmax":
        scaler = MinMaxScaler()
    elif config.NORMALIZATION == "robust":
        scaler = RobustScaler()
    else:
        return X_train, X_val, X_test

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def split_data(df, config, logger=None):
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    if logger:
        logger.info("Splitting data...")

    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])

    if config.TIME_COL and config.TIME_COL in X.columns:
        X = X.drop(columns=[config.TIME_COL])

    # Time-based split if time column exists
    if config.TIME_COL and config.TIME_COL in df.columns:
        if logger:
            logger.info("Using time-based split...")
        split_idx = int(len(df) * (1 - config.TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        val_idx = int(len(X_train) * (1 - config.VAL_SIZE))
        X_val, y_val = X_train.iloc[val_idx:], y_train.iloc[val_idx:]
        X_train, y_train = X_train.iloc[:val_idx], y_train.iloc[:val_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=config.VAL_SIZE, random_state=config.SEED
        )

    if logger:
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def get_model(name, config):
    """Get a model instance by name."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=config.SEED),
        "lasso": Lasso(alpha=0.1, random_state=config.SEED, max_iter=10000),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=config.SEED, max_iter=10000),
        "rf": RandomForestRegressor(n_estimators=100, random_state=config.SEED, n_jobs=-1),
        "gbm": GradientBoostingRegressor(n_estimators=100, random_state=config.SEED),
    }

    return models.get(name, LinearRegression())


def train_models(X_train, y_train, X_val, y_val, config, logger=None):
    """Train all configured models and return results."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if logger:
        logger.info(f"Training models: {config.MODELS}")

    results = {}
    trained_models = {}

    for name in config.MODELS:
        if logger:
            logger.info(f"  Training {name}...")

        model = get_model(name, config)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        results[name] = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "overfit_ratio": float(val_rmse / train_rmse) if train_rmse > 0 else float("inf"),
        }

        trained_models[name] = model

        if logger:
            logger.info(f"    Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")

    return trained_models, results


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate(X, y, config, logger=None):
    """Perform k-fold cross-validation on the primary model."""
    from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
    from sklearn.metrics import make_scorer, mean_squared_error

    if logger:
        logger.info(f"Cross-validating {config.PRIMARY_MODEL} with {config.CV_FOLDS} folds...")

    model = get_model(config.PRIMARY_MODEL, config)

    # Use TimeSeriesSplit if time-based, otherwise KFold
    if config.TIME_COL:
        cv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    else:
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                              greater_is_better=False)

    scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)

    cv_results = {
        "mean_rmse": float(-scores.mean()),
        "std_rmse": float(scores.std()),
        "fold_scores": [float(-s) for s in scores],
    }

    if logger:
        logger.info(f"  CV RMSE: {cv_results['mean_rmse']:.4f} +/- {cv_results['std_rmse']:.4f}")

    return cv_results


# ============================================================================
# FEATURE IMPORTANCE & INTERPRETABILITY
# ============================================================================

def feature_importance(model, feature_names, config, logger=None):
    """Extract and rank feature importance from the trained model."""
    if logger:
        logger.info("Extracting feature importance...")

    importance = None

    if hasattr(model, "coef_"):
        # Linear models: use coefficients
        importance = np.abs(model.coef_)
        method = "coefficient_magnitude"
    elif hasattr(model, "feature_importances_"):
        # Tree-based models
        importance = model.feature_importances_
        method = "feature_importance"
    else:
        if logger:
            logger.warning("Model does not support feature importance extraction.")
        return None

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df["rank"] = fi_df.index + 1
    fi_df["method"] = method

    # Normalize importance
    fi_df["importance_normalized"] = fi_df["importance"] / fi_df["importance"].sum()

    if logger:
        logger.info(f"Top 10 features ({method}):")
        for _, row in fi_df.head(10).iterrows():
            logger.info(f"  {row['rank']}. {row['feature']}: {row['importance']:.6f}")

    # Save
    fi_df.to_csv(os.path.join(config.OUTPUT_DIR, "feature_importance.csv"), index=False)

    return fi_df


def get_model_coefficients(model, feature_names, config, logger=None):
    """Get model coefficients with interpretation for linear models."""
    if not hasattr(model, "coef_"):
        return None

    if logger:
        logger.info("Extracting model coefficients...")

    coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_,
    })

    if hasattr(model, "intercept_"):
        intercept_row = pd.DataFrame({
            "feature": ["intercept"],
            "coefficient": [model.intercept_],
        })
        coefs = pd.concat([intercept_row, coefs], ignore_index=True)

    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # Sign interpretation
    coefs["direction"] = coefs["coefficient"].apply(
        lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
    )

    if logger:
        logger.info("Top 10 coefficients:")
        for _, row in coefs.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.6f} ({row['direction']})")

    coefs.to_csv(os.path.join(config.OUTPUT_DIR, "model_coefficients.csv"), index=False)

    return coefs


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis_rolling_window(df, config, target_col, windows=None, logger=None):
    """
    Sensitivity analysis: How robust is the signal when hyperparameters change?
    Tests different rolling windows for feature construction.
    """
    if windows is None:
        windows = config.SENSITIVITY_WINDOWS

    if logger:
        logger.info(f"Sensitivity analysis with windows: {windows}")

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    results = []

    for window in windows:
        if logger:
            logger.info(f"  Testing window={window}...")

        # Create features with this window
        temp_df = df.copy()
        temp_df = temp_df.sort_values(config.TIME_COL).reset_index(drop=True) if config.TIME_COL in temp_df.columns else temp_df

        # Rolling features with current window
        if config.TIME_COL and config.TIME_COL in temp_df.columns and target_col in temp_df.columns:
            temp_df[f"{target_col}_rolling_mean_{window}"] = temp_df[target_col].rolling(window=window, min_periods=1).mean()
            temp_df[f"{target_col}_rolling_std_{window}"] = temp_df[target_col].rolling(window=window, min_periods=1).std()

        # Prepare data
        y = temp_df[config.TARGET]
        X = temp_df.drop(columns=[config.TARGET])
        if config.TIME_COL and config.TIME_COL in X.columns:
            X = X.drop(columns=[config.TIME_COL])
        X = X.select_dtypes(include=[np.number]).fillna(0)

        # Quick train/val split
        split_idx = int(len(X) * 0.8)
        X_tr, X_vl = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_vl = y.iloc[:split_idx], y.iloc[split_idx:]

        model = Ridge(alpha=1.0, random_state=config.SEED)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_vl)
        rmse = np.sqrt(mean_squared_error(y_vl, pred))

        results.append({
            "window": window,
            "val_rmse": float(rmse),
            "n_features": X.shape[1],
        })

    results_df = pd.DataFrame(results)

    # Compute stability metric
    rmse_values = results_df["val_rmse"].values
    results_df["rmse_change_pct"] = results_df["val_rmse"].pct_change() * 100
    results_df["stability_score"] = 1.0 / (1.0 + np.std(rmse_values) / np.mean(rmse_values))

    if logger:
        logger.info(f"Sensitivity results:")
        for _, row in results_df.iterrows():
            logger.info(f"  Window {int(row['window'])}: RMSE={row['val_rmse']:.4f}, stability={row['stability_score']:.4f}")
        logger.info(f"Overall stability score: {results_df['stability_score'].mean():.4f}")

    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "sensitivity_analysis.csv"), index=False)

    return results_df


def sensitivity_analysis_regularization(X_train, y_train, X_val, y_val, config, logger=None):
    """
    Sensitivity analysis for regularization parameter (alpha).
    Tests how model performance changes with different regularization strengths.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    if logger:
        logger.info("Sensitivity analysis for regularization parameter...")

    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    results = []

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=config.SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))

        results.append({
            "alpha": alpha,
            "val_rmse": float(rmse),
            "n_nonzero_coefs": int(np.sum(np.abs(model.coef_) > 1e-8)),
        })

    results_df = pd.DataFrame(results)
    results_df["rmse_change_pct"] = results_df["val_rmse"].pct_change() * 100

    if logger:
        for _, row in results_df.iterrows():
            logger.info(f"  Alpha={row['alpha']:.3f}: RMSE={row['val_rmse']:.4f}, nonzero_coefs={row['n_nonzero_coefs']}")

    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "sensitivity_regularization.csv"), index=False)

    return results_df


# ============================================================================
# SANITY CHECKS
# ============================================================================

def sanity_checks(model, X_test, y_test, predictions, feature_names, config, logger=None):
    """
    Sanity checks: Whether conclusions make practical economic sense
    rather than just statistical sense.
    """
    if logger:
        logger.info("Running sanity checks...")

    checks = {}

    # Check 1: Prediction range
    pred_min, pred_max = predictions.min(), predictions.max()
    actual_min, actual_max = y_test.min(), y_test.max()
    checks["prediction_range"] = {
        "pred_min": float(pred_min),
        "pred_max": float(pred_max),
        "actual_min": float(actual_min),
        "actual_max": float(actual_max),
        "pred_negative_count": int((predictions < 0).sum()),
        "flag": "OK" if pred_min >= actual_min * 0.5 and pred_max <= actual_max * 2.0 else "WARNING",
    }

    # Check 2: Residual distribution
    residuals = y_test - predictions
    checks["residuals"] = {
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
        "skew": float(residuals.skew()),
        "flag": "OK" if abs(residuals.mean()) < 0.1 * y_test.std() else "WARNING",
    }

    # Check 3: Feature sign consistency
    if hasattr(model, "coef_"):
        coefs = dict(zip(feature_names, model.coef_))
        sign_checks = {}

        # Check if well-known relationships have expected signs
        # (these can be customized based on domain knowledge)
        for feature, expected_sign in [("demand_forecast", 1), ("demand_actual", 1),
                                        ("wind_generation", -1), ("solar_generation", -1),
                                        ("gas_price", 1), ("co2_price", 1)]:
            if feature in coefs:
                actual_sign = 1 if coefs[feature] > 0 else (-1 if coefs[feature] < 0 else 0)
                sign_checks[feature] = {
                    "coefficient": float(coefs[feature]),
                    "expected_sign": expected_sign,
                    "actual_sign": actual_sign,
                    "consistent": actual_sign == expected_sign,
                }

        checks["feature_signs"] = {
            "details": sign_checks,
            "n_consistent": sum(1 for v in sign_checks.values() if v["consistent"]),
            "n_total": len(sign_checks),
            "flag": "OK" if all(v["consistent"] for v in sign_checks.values()) else "WARNING",
        }

    # Check 4: R² sanity
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    checks["r2_sanity"] = {
        "r2": float(r2),
        "flag": "OK" if 0 < r2 < 1 else "WARNING",
    }

    # Check 5: Autocorrelation of residuals (time series)
    if len(residuals) > 10:
        from pandas import Series
        residual_series = Series(residuals)
        autocorr_lag1 = residual_series.autocorr(lag=1) if len(residuals) > 1 else 0
        checks["residual_autocorrelation"] = {
            "lag1_autocorr": float(autocorr_lag1) if not np.isnan(autocorr_lag1) else 0.0,
            "flag": "OK" if abs(autocorr_lag1 if not np.isnan(autocorr_lag1) else 0) < 0.5 else "WARNING",
        }

    if logger:
        for check_name, result in checks.items():
            flag = result.get("flag", "INFO")
            logger.info(f"  [{flag}] {check_name}: {result}")

    with open(os.path.join(config.OUTPUT_DIR, "sanity_checks.json"), "w") as f:
        json.dump(checks, f, indent=2, default=str)

    return checks


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(df, model, X_test, y_test, predictions, fi_df, config, logger=None):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if logger:
        logger.info("Generating plots...")

    sns.set_style("whitegrid")
    os.makedirs(os.path.join(config.OUTPUT_DIR, "plots"), exist_ok=True)

    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=config.PLOT_FIGSIZE)
    axes[0].hist(df[config.TARGET], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Distribution of {config.TARGET}")
    axes[0].set_xlabel(config.TARGET)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(np.log1p(df[config.TARGET]), bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[1].set_title(f"Log-transformed {config.TARGET}")
    axes[1].set_xlabel(f"log(1+{config.TARGET})")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "target_distribution.png"), dpi=config.PLOT_DPI)
    plt.close()

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 20:
        top_corr = numeric_df.corr()[config.TARGET].abs().nlargest(20).index
        numeric_df = numeric_df[top_corr]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "correlation_heatmap.png"), dpi=config.PLOT_DPI)
    plt.close()

    # 3. Predicted vs Actual
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    ax.scatter(y_test, predictions, alpha=0.3, s=5)
    lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(lims, lims, "r--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "pred_vs_actual.png"), dpi=config.PLOT_DPI)
    plt.close()

    # 4. Residual plot
    residuals = y_test - predictions
    fig, axes = plt.subplots(1, 2, figsize=config.PLOT_FIGSIZE)
    axes[0].scatter(predictions, residuals, alpha=0.3, s=5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")

    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "residuals.png"), dpi=config.PLOT_DPI)
    plt.close()

    # 5. Feature importance
    if fi_df is not None and len(fi_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, min(20, len(fi_df) * 0.3))))
        top_n = min(20, len(fi_df))
        top_fi = fi_df.head(top_n)
        ax.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "feature_importance.png"), dpi=config.PLOT_DPI)
        plt.close()

    # 6. Time series plot (if applicable)
    if config.TIME_COL and config.TIME_COL in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_df = df[[config.TIME_COL, config.TARGET]].tail(500)
        ax.plot(plot_df[config.TIME_COL], plot_df[config.TARGET], linewidth=0.5, alpha=0.7)
        ax.set_title(f"{config.TARGET} Over Time (Last 500 observations)")
        ax.set_xlabel("Time")
        ax.set_ylabel(config.TARGET)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", "time_series.png"), dpi=config.PLOT_DPI)
        plt.close()

    if logger:
        logger.info(f"Plots saved to {os.path.join(config.OUTPUT_DIR, 'plots')}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config, logger=None):
    """Run the complete quant research pipeline."""
    start_time = time.time()

    if logger is None:
        logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("QUANT RESEARCH PIPELINE START")
    logger.info("=" * 80)

    # 1. Load data
    df = load_data(config, logger)

    # 2. EDA
    eda_results = eda(df, config, logger)

    # 3. Feature engineering
    df = engineer_features(df, config, logger)

    # 4. Handle missing values
    df = handle_missing_values(df, config, logger)

    # 5. Handle outliers
    df = handle_outliers(df, config, logger)

    # 6. Encode categorical
    df = encode_categorical(df, config, logger)

    # 7. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config, logger)

    # 8. Normalize
    X_train_norm, X_val_norm, X_test_norm = normalize_features(
        X_train, X_val, X_test, config, logger
    )

    # 9. Train models
    trained_models, training_results = train_models(
        X_train_norm, y_train, X_val_norm, y_val, config, logger
    )

    # 10. Cross-validation
    X_full = np.vstack([X_train_norm, X_val_norm])
    y_full = np.concatenate([y_train, y_val])
    cv_results = cross_validate(X_full, y_full, config, logger)

    # 11. Feature importance
    primary_model = trained_models[config.PRIMARY_MODEL]
    feature_names = X_train.columns.tolist()
    fi_df = feature_importance(primary_model, feature_names, config, logger)
    coef_df = get_model_coefficients(primary_model, feature_names, config, logger)

    # 12. Sensitivity analysis
    sensitivity_rolling = sensitivity_analysis_rolling_window(df, config, config.TARGET, logger=logger)
    sensitivity_reg = sensitivity_analysis_regularization(X_train_norm, y_train, X_val_norm, y_val, config, logger)

    # 13. Final predictions
    predictions = primary_model.predict(X_test_norm)

    # 14. Sanity checks
    sanity_results = sanity_checks(primary_model, X_test_norm, y_test, predictions, feature_names, config, logger)

    # 15. Plots
    plot_results(df, primary_model, X_test_norm, y_test, predictions, fi_df, config, logger)

    # 16. Summary
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Best model: {config.PRIMARY_MODEL}")
    logger.info(f"Test R2: {sanity_results['r2_sanity']['r2']:.4f}")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")
    logger.info("=" * 80)

    # Save final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "data_shape": list(df.shape),
        "training_results": training_results,
        "cv_results": cv_results,
        "sanity_checks": sanity_results,
        "feature_importance_top10": fi_df.head(10).to_dict() if fi_df is not None else None,
    }

    with open(os.path.join(config.OUTPUT_DIR, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return {
        "model": primary_model,
        "predictions": predictions,
        "training_results": training_results,
        "cv_results": cv_results,
        "feature_importance": fi_df,
        "coefficients": coef_df,
        "sensitivity_rolling": sensitivity_rolling,
        "sensitivity_regularization": sensitivity_reg,
        "sanity_checks": sanity_results,
        "report": report,
    }


if __name__ == "__main__":
    config = Config()
    results = run_pipeline(config)
