# source /home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/UI_AG_python/bin/activate

import matplotlib.pyplot as plt
import numpy as np
from src.views.mainView import *
import seaborn as sns

"""
pip install ydata-profiling
pip uninstall ydata-profiling

from ydata_profiling import ProfileReport

df = loadDF_Consensus("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML")
print(df)

# Generate comprehensive profile
#profile = ProfileReport(df, title="Dataset Summary", explorative=True, correlations={"auto": {"calculate": True}}, missing_diagrams={ 'heatmap': True, 'dendrogram': True, })

profile = ProfileReport(df, 
                        title="Dataset Summary",
                        explorative=True)

# Save to HTML report
profile.to_file("dataset_summary.html")

summary_dict = profile.get_description()  # Get full description as dictionary

#summary_stats = summary_dict['Variables']  # Access variables section

# Convert to DataFrame
#summary_df = pd.DataFrame(summary_stats).T
#summary_df.index.name = 'variable'
#summary_df.reset_index(inplace=True)
#summary_df.to_csv('dataset_summary_df.csv', index=False)
"""

df = loadDF_Consensus("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/normalized.consensusXML")
#df = pd.read_csv("/home/pixel/Documents/Cinvestav_2025/UI_Analisis_Genomico/Consensus/xcms_all_features.csv")
#print(df)

def calc_mad(series):
        median_val = np.median(series)
        absolute_deviations = np.abs(series - median_val)
        mad = np.median(absolute_deviations)
        return mad

def preprocessing_summary_perVariable(df):
    # Initialize summary dataframe
    summary = pd.DataFrame({
        'variable': df.columns,
        'type': df.dtypes.astype(str),
        'count': df.count().values,
        'missing': df.isnull().sum().values,
        'missing_pct': (df.isnull().mean() * 100).values,
    })

    # Numerical features
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        summary.loc[summary['variable'] == col, 'mean'] = df[col].mean()
        summary.loc[summary['variable'] == col, 'std'] = df[col].std()
        summary.loc[summary['variable'] == col, 'min'] = df[col].min()
        summary.loc[summary['variable'] == col, 'max'] = df[col].max()
        summary.loc[summary['variable'] == col, 'median'] = df[col].median()
        summary.loc[summary['variable'] == col, 'skew'] = df[col].skew()
        summary.loc[summary['variable'] == col, 'kurtosis'] = df[col].kurtosis()
        summary.loc[summary['variable'] == col, 'variance'] = df[col].var() # Added Variance

        # Percentiles
        percentiles = df[col].quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        for p in percentiles.index:
            summary.loc[summary['variable'] == col, f'p{p}'] = percentiles[p]

        # Added additional numerical metrics
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        summary.loc[summary['variable'] == col, 'range'] = df[col].max() - df[col].min()
        summary.loc[summary['variable'] == col, 'iqr'] = q3 - q1
        #summary.loc[summary['variable'] == col, 'mad'] = df[col].mad() # Median Absolute Deviation
        summary.loc[summary['variable'] == col, 'mad'] = calc_mad(df[col]) # Median Absolute Deviation

    # Categorical features
    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
        summary.loc[summary['variable'] == col, 'nunique'] = df[col].nunique()
        mode = df[col].mode()
        summary.loc[summary['variable'] == col, 'top'] = mode[0] if not mode.empty else None
        summary.loc[summary['variable'] == col, 'freq'] = df[col].value_counts().iloc[0] if not df[col].empty else 0
        summary.loc[summary['variable'] == col, 'freq_pct'] = (df[col].value_counts().iloc[0]/len(df)*100 if not df[col].empty else 0)

    # Additional metrics (zero_count and zero_pct are already there)
    summary['zero_count'] = df.apply(lambda x: (x == 0).sum() if np.issubdtype(x.dtype, np.number) else np.nan)
    summary['zero_pct'] = summary['zero_count'] / len(df) * 100

    return summary.sort_values(by='type')

def preprocessing_general_dataset_statistics(df):
    general_stats = {}
    general_stats['number_of_variables'] = df.shape[1]
    general_stats['number_of_observations'] = df.shape[0]
    general_stats['total_missing_cells'] = df.isnull().sum().sum()
    general_stats['total_missing_cells_pct'] = (general_stats['total_missing_cells'] / (df.shape[0] * df.shape[1])) * 100
    general_stats['total_size_in_memory_bytes'] = df.memory_usage(deep=True).sum()
    general_stats['total_size_in_memory_mb'] = general_stats['total_size_in_memory_bytes'] / (1024**2)

    ### get dimensions of df and df.describe

    # Number of each variable type
    general_stats['variable_types_count'] = df.dtypes.value_counts().to_dict()

    #return general_stats
    return pd.DataFrame([general_stats])

# Usage
summary_table = preprocessing_summary_perVariable(df)
#print(summary_table)
#summary_table.to_csv('dataset_summary.csv', index=False)

general_table = preprocessing_general_dataset_statistics(df)
#general_table_df = pd.DataFrame([general_table])

#combined_df = pd.concat([general_table, summary_table], ignore_index=True)
combined_df = pd.concat([general_table, summary_table], axis=1)

#combined_df.to_csv('dataset_full_summary.csv', index=False)

## then we need correlation matrix (df.corr()), missigno plot library, (msno.matrix(df) or msno.bar(df), )

def visualize_data(df, summary):
    """
    Visualizes key aspects of the DataFrame including missing values,
    data types distribution, correlation matrix, and overall null vs. non-null values.

    Args:
        df (pd.DataFrame): The input DataFrame to visualize.
        summary (pd.DataFrame): A summary DataFrame, expected to have a 'type' column
                                for variable types distribution.
    """

    # --- 1. Missing values heatmap ---
    # Shows the distribution of missing values across the DataFrame.
    # Rows with missing values will appear as lines, columns with missing values as vertical streaks.
    # Y-axis is inverted so that the "max value" (last row index) is at the top.
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='binary_r') # Using 'viridis' for visibility, though binary is common
    plt.title('Missing Values Distribution')
    plt.gca().invert_yaxis() # Invert Y-axis so max value (last row index) is at the top
    plt.show()

    # --- 2. Data types distribution ---
    # Shows a bar plot of the count for each variable type in the DataFrame.
    plt.figure(figsize=(8, 6))
    type_counts = summary['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='mako')
    plt.title('Variable Types Distribution (Counts)')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout()
    plt.show()

    # --- 3. Correlation Matrix ---
    # Computes and visualizes the correlation matrix for numerical columns.
    # Uses the 'viridis' colormap as requested.
    numerical_df = df.select_dtypes(include=[np.number])

    if not numerical_df.empty and numerical_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = numerical_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix (Numerical Columns)')
        plt.show()
    else:
        print("Not enough numerical columns to display a correlation matrix.")

    # --- 4. Pie Plot for Overall Null vs. Non-Null Values ---
    # Calculates the total number of null values and non-null values in the entire DataFrame
    # and displays their proportion in a pie chart.
    total_elements = df.size # Total number of cells in the DataFrame
    total_nulls = df.isnull().sum().sum() # Sum of all nulls across all columns

    if total_elements > 0: # Ensure DataFrame is not empty to avoid division by zero
        total_non_nulls = total_elements - total_nulls

        # Prepare data for the pie chart
        labels = ['Null Values', 'Non-Null Values']
        sizes = [total_nulls, total_non_nulls]
        colors = ['#ff9999', '#66b3ff'] # Custom colors for better distinction
        explode = (0.1, 0)  # Explode the 'Null Values' slice for emphasis if nulls exist

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Overall Null vs. Non-Null Values Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
    else:
        print("DataFrame is empty, cannot plot null vs. non-null distribution.")

# Generate visualizations
visualize_data(df, summary_table)