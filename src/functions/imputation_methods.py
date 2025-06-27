from matplotlib.figure import Figure
import pyopenms as oms
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from missingpy import MissForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import BayesianRidge, LinearRegression

## Univariate methods for numerical columns

# Zero Imputation / N value imputation
def nImputed(df,n=0):
    # Only alter numerical columns and columns with nan values
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return df_imputed
    
    for col in num_cols:
        if df_imputed[col].isnull().any():
            df_imputed[col] = df_imputed[col].fillna(n)

    return df_imputed

# Half Minimum imputation
def halfMinimumImputed(df):
    # Only alter numerical columns and columns with nan values
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    for col in num_cols:
        if df_imputed[col].isnull().any():
            col_min = df_imputed[col].min()

            if not pd.isna(col_min):
                imputation_value = col_min / 2
                df_imputed[col] = df_imputed[col].fillna(imputation_value)
            else:
                df_imputed[col] = df_imputed[col].fillna(0) # Fallback if no non-NaN values to get a min from
                
    return df_imputed

# Mean imputation
def meanImputed(df):
    # Only alter numerical columns and columns with nan values
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return df_imputed
    
    for col in num_cols:
        col_mean = df_imputed[col].mean()
        if not pd.isna(col_mean):
            df_imputed[col] = df_imputed[col].fillna(col_mean)
        else:
            df_imputed[col] = df_imputed[col].fillna(0) # Fallback if no non-NaN values to get a min from

    return df_imputed

# Median imputation
def medianImputed(df):
    # Only alter numerical columns and columns with nan values
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return df_imputed
    
    for col in num_cols:
        col_median = df_imputed[col].median()
        if not pd.isna(col_median):
            df_imputed[col] = df_imputed[col].fillna(col_median)
        else:
            df_imputed[col] = df_imputed[col].fillna(0) # Fallback if no non-NaN values to get a min from
                
    return df_imputed

## Multivariate methods for numerical columns

# Random Forest imputation
def missForestImputed(df, max_iter=10, n_estimators=100, random_state=None):
    df_imputed = df.copy() # Operate on a copy to ensure original is not modified
    # MissForest can handle numerical ('np.number'), object (strings), and categorical dtypes directly.
    imputable_cols = df_imputed.select_dtypes(include=[np.number, 'object', 'category']).columns.tolist()

    if not imputable_cols:
        print("No imputable numerical or categorical columns found for MissForest imputation.")
        return df_imputed

    # Create a sub-DataFrame containing only the columns that MissForest will impute
    df_to_impute = df_imputed[imputable_cols]
    if random_state:
        imputer = MissForest(max_iter=max_iter, n_estimators=n_estimators, random_state=random_state)
    else:
        imputer = MissForest(max_iter=max_iter, n_estimators=n_estimators)

    imputed_array = imputer.fit_transform(df_to_impute)
    df_imputed_part = pd.DataFrame(imputed_array, columns=imputable_cols, index=df_to_impute.index)

    for col in imputable_cols:
        df_imputed[col] = df_imputed_part[col]

    return df_imputed[df.columns] 

# Singular Value Decomposition
def svdImputed(df, n_components=5, max_iter=100, tol=1e-4):
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    df_num = df_imputed[num_cols]
    
    # Store original NaN positions
    missing_mask = df_num.isnull()

    # Initialize missing values with zero
    df_filled_zero = df_num.fillna(0)

    # Scale and centralize the data (as mentioned in the description for metabolomics data)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filled_zero), columns=num_cols, index=df_num.index)

    imputed_array_prev = df_scaled.values.copy()

    for i in range(max_iter):
        # Apply SVD
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(imputed_array_prev)
        # Reconstruct the matrix using the n_components
        imputed_array_current = svd.inverse_transform(svd.transform(imputed_array_prev))

        # Only update the originally missing values
        imputed_array_current[~missing_mask.values] = df_scaled.values[~missing_mask.values]

        # Check for convergence
        if np.linalg.norm(imputed_array_current - imputed_array_prev, 'fro') < tol:
            break
        
        imputed_array_prev = imputed_array_current.copy()

    # Inverse scale the imputed data
    imputed_final = scaler.inverse_transform(imputed_array_prev)
    df_imputed[num_cols] = pd.DataFrame(imputed_final, columns=num_cols, index=df_num.index)

    return df_imputed

# K nearest neighbours
def knnImputed(df,n_neighbors=2):
    # Only alter numerical columns and columns with nan values
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    df_num = df_imputed[num_cols]
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = knn_imputer.fit_transform(df_num)
    df_imputed[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df_num.index)

    return df_imputed

# NOT USING QRILC = Quantile Regression Imputation of Left-Censored data
# MICE with bayesian ridge 
def miceBayesianRidgeImputed(df, max_iter=10, random_state=None, initial_strategy='mean', min_value_for_log=1e-9):
    # initial_strategy: 'mean', 'median', 'most_frequent', or 'constant'
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    df_num = df_imputed[num_cols]

    # Apply log transformation (add a small constant to handle zeros or very small values)
    # This is critical for data that is positive-skewed or left-censored, typical in metabolomics
    df_log_transformed = np.log(df_num + min_value_for_log)

    # Initialize IterativeImputer with BayesianRidge as the estimator
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        add_indicator=False # Do not add a binary indicator column for imputed values
    )

    # Fit and transform the log-transformed data
    imputed_log_array = imputer.fit_transform(df_log_transformed)

    # Inverse log transform
    imputed_original_scale = np.exp(imputed_log_array) - min_value_for_log
    
    # Post-hoc adjustment: ensure no imputed values are negative if original data is non-negative
    # This is important for concentration data where values cannot be less than zero.
    imputed_original_scale[imputed_original_scale < 0] = 0

    # Update the numerical columns in the copied DataFrame
    df_imputed[num_cols] = pd.DataFrame(imputed_original_scale, columns=num_cols, index=df_num.index)

    return df_imputed

# MICE with linear regresion
def miceLinearRegressionImputed(df, max_iter=10, random_state=None, initial_strategy='mean', min_value_for_log=1e-9):
    # initial_strategy: 'mean', 'median', 'most_frequent', or 'constant'
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    df_num = df_imputed[num_cols]

    # Apply log transformation (add a small constant to handle zeros or very small values)
    df_log_transformed = np.log(df_num + min_value_for_log)

    # Initialize IterativeImputer with LinearRegression as the estimator
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        add_indicator=False
    )

    # Fit and transform the log-transformed data
    imputed_log_array = imputer.fit_transform(df_log_transformed)

    # Inverse log transform
    imputed_original_scale = np.exp(imputed_log_array) - min_value_for_log

    # Post-hoc adjustment: ensure no imputed values are negative if original data is non-negative
    imputed_original_scale[imputed_original_scale < 0] = 0

    # Update the numerical columns in the copied DataFrame
    df_imputed[num_cols] = pd.DataFrame(imputed_original_scale, columns=num_cols, index=df_num.index)

    return df_imputed

# Apply biological constraints and rescaling
def postprocess_imputation(imputed_df, original_df):
    # Rescale to original metrics
    imputed_df = imputed_df * original_df.std() + original_df.mean()
    
    # Apply biological constraints
    for col in imputed_df.columns:
        min_val = original_df[col].min()
        if min_val >= 0:  # Concentrations can't be negative
            imputed_df[col] = imputed_df[col].clip(lower=min_val/10)
    
    return imputed_df
