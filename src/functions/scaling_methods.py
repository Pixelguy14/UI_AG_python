import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore

def standard_scaling(df, with_mean=True, with_std=True):
    """
    Z-score scaling (Standard Scaling).
    Scales each feature (row) to mean=0 and std=1 across samples.
    """
    # Transpose to have features as columns for StandardScaler
    df_t = df.T
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    
    # Fit, transform, and transpose back
    return pd.DataFrame(
        scaler.fit_transform(df_t),
        index=df_t.index,
        columns=df_t.columns
    ).T

def minmax_scaling(df, feature_range=(0, 1)):
    """
    Min-Max scaling to a specified range (default [0, 1]).
    Scales each feature (row) to the given range across samples.
    """
    # Transpose to have features as columns for MinMaxScaler
    df_t = df.T
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit, transform, and transpose back
    return pd.DataFrame(
        scaler.fit_transform(df_t),
        index=df_t.index,
        columns=df_t.columns
    ).T

def pareto_scaling(df):
    """
    Pareto Scaling: (x - mean) / √std
    Compromise between Z-score and no scaling
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    # Handle zero stds
    stds[stds == 0] = 1
    
    # Compute √std with sign preservation
    sqrt_stds = np.sign(stds) * np.sqrt(np.abs(stds))
    
    # Center and scale
    return (df.sub(means, axis=0)).div(sqrt_stds, axis=0)

def range_scaling(df):
    """
    Range Scaling: Scale features to [-1, 1] while preserving sign
    """
    max_vals = df.abs().max(axis=1)
    
    # Handle features with all zeros
    max_vals[max_vals == 0] = 1
    
    return df.div(max_vals, axis=0)

def robust_scaling(df):
    """
    Robust Scaling: (x - median) / IQR
    Resistant to outliers
    """
    medians = df.median(axis=1)
    q1 = df.quantile(0.25, axis=1)
    q3 = df.quantile(0.75, axis=1)
    iqr = q3 - q1
    
    # Handle zero IQR
    iqr[iqr == 0] = 1
    
    return (df.sub(medians, axis=0)).div(iqr, axis=0)

def vast_scaling(df):
    """
    Vast Scaling: ((x - mean) / std) * (mean / std)
    Reduces the influence of noisy variables.
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    # Handle zero stds
    stds[stds == 0] = 1
    
    # Center the data
    centered_df = df.sub(means, axis=0)
    
    # Calculate the scaling factor (mean / std)
    scaling_factor = means / stds
    
    # Apply scaling
    return centered_df.div(stds, axis=0).mul(scaling_factor, axis=0)
