import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer # type: ignore
from sklearn.experimental import enable_iterative_imputer  # type: ignore # Has to stay in order for IterativeImputer to work
from sklearn.impute import IterativeImputer # type: ignore
from missingpy import MissForest # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import TruncatedSVD # type: ignore
from sklearn.linear_model import BayesianRidge, LinearRegression # type: ignore

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
    # Drop features with all missing values, as they are uninformative
    df_imputed.dropna(axis=0, how='all', inplace=True)

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
    df_copy = df.copy()
    df_copy.dropna(axis=0, how='all', inplace=True)
    df_imputed = df_copy.T
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

    return df_imputed.T

# K nearest neighbours
def knnImputed(df,n_neighbors=2):
    df_copy = df.copy()
    df_copy.dropna(axis=0, how='all', inplace=True)
    # Only alter numerical columns and columns with nan values
    df_imputed = df_copy.T
    num_cols = df_imputed.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return df_imputed

    df_num = df_imputed[num_cols]
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = knn_imputer.fit_transform(df_num)
    df_imputed[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df_num.index)

    return df_imputed.T

# NOT USING QRILC = Quantile Regression Imputation of Left-Censored data
# MICE with bayesian ridge 
def miceBayesianRidgeImputed(df, max_iter=20, random_state=None, initial_strategy='mean', tol=1e-3):
    df_copy = df.copy()
    df_copy.dropna(axis=0, how='all', inplace=True)
    # initial_strategy: 'mean', 'median', 'most_frequent', or 'constant'
    df_imputed = df_copy.T
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return df_imputed
    
    df_num = df_imputed[num_cols]
    
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        add_indicator=False,
        tol=tol
    )
    
    imputed_array = imputer.fit_transform(df_num)
    
    df_imputed[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df_num.index)
    
    return df_imputed.T

# MICE with linear regresion
def miceLinearRegressionImputed(df, max_iter=20, random_state=None, initial_strategy='mean', tol=1e-3):
    df_copy = df.copy()
    df_copy.dropna(axis=0, how='all', inplace=True)
    # initial_strategy: 'mean', 'median', 'most_frequent', or 'constant'
    df_imputed = df_copy.T
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return df_imputed
    
    df_num = df_imputed[num_cols]
    
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        add_indicator=False,
        tol=tol
    )
    
    imputed_array = imputer.fit_transform(df_num)
    
    # Create DataFrame for imputed values
    df_imputed[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df_num.index)

    return df_imputed.T

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