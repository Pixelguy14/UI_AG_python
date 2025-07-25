import pandas as pd
import numpy as np
from sklearn.svm import SVR # type: ignore
# from sklearn.preprocessing import StandardScaler # type: ignore

# Total Ion Count (TIC) Normalization
def tic_normalization(df):
    """
    Normalizes each sample (column) by its total ion count (sum of intensities).
    This is also known as Sum Normalization.
    """
    sample_sums = df.sum(axis=0)
    sample_sums = sample_sums.replace(0, 1)  # Avoid division by zero
    return df / sample_sums

# Median Total Ion Count (mTIC) Normalization
def mtic_normalization(df):
    """
    Normalizes each sample by its TIC, then scales to the median TIC of all samples.
    """
    sample_sums = df.sum(axis=0)
    median_tic = sample_sums.median()
    sample_sums = sample_sums.replace(0, 1)
    return (df / sample_sums) * median_tic

# Median Normalization
def median_normalization(df):
    """
    Normalizes each sample (column) by its median intensity.
    Robust to outliers.
    """
    median_vals = df.median(axis=0)
    median_vals = median_vals.replace(0, 1)
    return df / median_vals

# Quantile Normalization
def quantile_normalization(df):
    """
    Forces all samples to have identical intensity distributions.
    May distort biological variance - use mainly for technical batch correction.
    Optimized quantile normalization for large metabolomics datasets.
    """
    # Sort each sample (column)
    sorted_df = np.sort(df.values, axis=0)
    
    # Calculate mean of sorted values across samples
    mean_sorted = np.mean(sorted_df, axis=1)
    
    # Get rank (0-based index) for original positions
    ranks = df.rank(axis=0, method="min").astype(int) - 1
    
    # Map ranks to mean values
    return pd.DataFrame(
        mean_sorted[ranks.values], 
        index=df.index, 
        columns=df.columns
    )

# Probabilistic Quotient Normalization (PQN)
def pqn_normalization(df):
    """
    Corrects for dilution effects based on a reference spectrum (median).
    """
    # Calculate the reference spectrum (median of all samples)
    reference_spectrum = df.median(axis=1).replace(0, np.nan)
    
    # Calculate the quotients for each sample relative to the reference
    quotients = df.div(reference_spectrum, axis=0)
    
    # Calculate the median of these quotients for each sample
    sample_medians = quotients.median(axis=0)
    sample_medians = sample_medians.replace(0, 1)
    
    return df / sample_medians

# Internal Standard (IS) Normalization
def is_normalization(df, is_feature_name):
    """
    Normalizes using a specific feature (row) as an internal standard.
    """
    if is_feature_name not in df.index:
        raise ValueError(f"Internal standard '{is_feature_name}' not found in DataFrame index.")
    
    is_values = df.loc[is_feature_name]
    is_values = is_values.replace(0, np.nan)  # Avoid division by zero, propagate NaN
    return df / is_values

# Support Vector Regression (SVR) Normalization for Batch Correction
def svr_normalization(df, qc_identifiers):
    """
    Uses Support Vector Regression (SVR) to correct for batch effects,
    assuming QC samples are included and their identifiers are known.
    
    Args:
        df (pd.DataFrame): DataFrame with features in rows, samples in columns.
        qc_identifiers (list): A list of column names that are QC samples.
    
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    # Ensure we have QC samples to model the drift
    qc_cols = [col for col in qc_identifiers if col in df.columns]
    if len(qc_cols) < 2:
        raise ValueError("Requires ≥2 QC samples to model the drift.")

    # Transpose for sklearn compatibility (features in columns)
    df_t = df.T
    
    # Separate QC and biological samples
    qc_data = df_t.loc[qc_cols]
    
    # Assume injection order is the index order for simplicity
    # A real implementation might need an explicit injection order series
    injection_order_qc = np.arange(len(qc_data)).reshape(-1, 1)
    injection_order_all = np.arange(len(df_t)).reshape(-1, 1)
    
    df_normalized_t = df_t.copy()

    # Build a model for each feature
    for feature in df_t.columns:
        y = qc_data[feature].values
        
        # Fit SVR model
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr.fit(injection_order_qc, y)
        
        # Predict the drift for all samples
        predicted_drift = svr.predict(injection_order_all)
        
        # Correct the feature values
        # Correction is typically done by subtraction or division.
        # Division is common for multiplicative drift in mass spec.
        # Adding a small epsilon to avoid division by zero.
        predicted_drift[predicted_drift == 0] = 1e-9
        df_normalized_t[feature] = df_t[feature] / (predicted_drift / np.mean(y))

    return df_normalized_t.T
