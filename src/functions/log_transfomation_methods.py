import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import PowerTransformer # type: ignore

def _check_for_negatives(df, transformation_name):
    """Warns if negative values are present in the DataFrame."""
    if (df < 0).any().any():
        num_negative_features = (df < 0).any(axis=1).sum()
        warnings.warn(
            f"Negative values detected in {num_negative_features} features! "
            f"The '{transformation_name}' transformation is not suitable for negative values. "
            "Consider using 'arcsinh', 'cube_root', 'glog', or 'yeo_johnson' instead.",
            UserWarning
        )

def log2_transform(df, pseudo_count=None):
    """
    Apply log2 transformation. A small pseudo-count is added to handle zeros.
    """
    _check_for_negatives(df, "log2")
    
    if pseudo_count is None:
        min_positive = df[df > 0].min().min()
        pseudo_count = 0.5 * min_positive if pd.notna(min_positive) else 1e-9
    
    return np.log2(df.clip(0) + pseudo_count)

def log10_transform(df, pseudo_count=None):
    """
    Apply log10 transformation. A small pseudo-count is added to handle zeros.
    """
    _check_for_negatives(df, "log10")
    
    if pseudo_count is None:
        min_positive = df[df > 0].min().min()
        pseudo_count = 0.5 * min_positive if pd.notna(min_positive) else 1e-9
        
    return np.log10(df.clip(0) + pseudo_count)

def sqrt_transform(df):
    """
    Apply square root transformation. Negative values will be converted to NaN.
    """
    _check_for_negatives(df, "square root")
    return np.sqrt(df.where(df >= 0, np.nan))

def cube_root_transform(df):
    """
    Apply cube root transformation. This transformation handles negative values.
    """
    return np.cbrt(df)

def arcsinh_transform(df, cofactor=5):
    """
    Apply inverse hyperbolic sine (arcsinh) transformation.
    Handles zero and negative values gracefully.
    """
    return np.arcsinh(df / cofactor)

def glog_transform(df, lamb=None):
    """
    Apply generalized log (glog) transformation.
    Handles zero and negative values. A lambda parameter can be tuned.
    """
    if lamb is None:
        # Estimate lambda based on the data range, a simple heuristic
        lamb = df.abs().min().min()
        if lamb == 0 or pd.isna(lamb):
            lamb = 1e-9
            
    return np.log(df + np.sqrt(df**2 + lamb))

def yeo_johnson_transform(df):
    """
    Apply Yeo-Johnson power transformation.
    This finds an optimal transformation to stabilize variance and handle non-normality.
    It works with positive, zero, and negative values.
    Requires scikit-learn.
    """
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    return pd.DataFrame(pt.fit_transform(df), index=df.index, columns=df.columns)
