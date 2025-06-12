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

def loadFile(fileName):
    _, extension = os.path.splitext(fileName)

    if extension.lower() == ".consensusxml":
        print("Loading a consensusXML file")
        return loadDF_Consensus(fileName) # For consensusXML
    elif extension.lower() == ".featurexml":
        print("Loading a featureXML file")
        return loadDF_Consensus(fileName) # For featureXML
    elif extension.lower() == ".tsv":
        print("Loading a TSV file")
        return loadDF_TSV_CSV(fileName, '\t') # For tsv
    elif extension.lower() == ".csv":
        print("Loading a CSV file")
        return loadDF_TSV_CSV(fileName) # For csv
    elif extension.lower() == ".xlsx":
        print("Loading an Excel file")
        return loadDF_Excel(fileName) # For xlsx xls
    elif extension.lower() == ".xls":
        print("Loading an Excel file")
        return loadDF_Excel(fileName) # For xlsx xls
    else:
        return pd.DataFrame() # Explicitly return empty dataframe

def loadDF_Consensus(myConsensusXML):
    try:
        consensus_map = oms.ConsensusMap()
        oms.ConsensusXMLFile().load(myConsensusXML, consensus_map)

        # Extract metadata from original file
        column_headers = consensus_map.getColumnHeaders()
        sorted_columns = sorted(column_headers.items(), key=lambda x: x[0])

        # Modify headers from file
        filenames = [os.path.basename(header.filename) for idx, header in sorted_columns] 

        # Build dataframe
        rows = []
        for cf in consensus_map:
            row = {
                'rt': cf.getRT(),
                'mz': cf.getMZ(),
                'intensity': cf.getIntensity()
            }
            
            # Initialize intensities as NaN
            for filename in filenames:
                row[filename] = float('nan')
            
            # Fill intensities for each file
            for fh in cf.getFeatureList():
                map_idx = fh.getMapIndex()
                if map_idx < len(filenames):
                    filename = filenames[map_idx]  # Short name
                    row[filename] = fh.getIntensity()
            
            rows.append(row)
    except Exception as e:
        print(f"Error loading ConsensusXML file '{myConsensusXML}': {e}")
        return None

    # Create the dataframe and order the rows
    df = pd.DataFrame(rows)
    columns = ['rt', 'mz', 'intensity'] + filenames
    df = df[columns]
    return df

def loadDF_Feature(myFeatureXML):
    try:
        feature_map = oms.FeatureMap()
        oms.FeatureXMLFile().load(myFeatureXML, feature_map)

        rows = []
        for f in feature_map: # Iterate through each Feature
            row = {
                'rt': f.getRT(),
                'mz': f.getMZ(),
                'intensity': f.getIntensity()
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure consistent column order
        columns_order = ['rt', 'mz', 'intensity']
        df = df[columns_order]
        return df

    except Exception as e:
        print(f"Error loading FeatureXML file '{myFeatureXML}': {e}")
        return None

def loadDF_TSV_CSV(myTSVCSVFile,separator=None):
    try:
        # Use pandas to read the TSV file, specifying tab as the separator
        # If there is not TSV, ignore separator
        if not separator:
            df = pd.read_csv(myTSVCSVFile)
            print(f"Successfully loaded CSV file: {myTSVCSVFile}")
        else:
            df = pd.read_csv(myTSVCSVFile, sep=separator)
            print(f"Successfully loaded TSV file: {myTSVCSVFile}")
        return df
    except FileNotFoundError:
        print(f"Error: file not found at '{myTSVCSVFile}'")
        return None
    except Exception as e:
        print(f"Error loading file '{myTSVCSVFile}': {e}")
        return None
    
def loadDF_Excel(myExcelFile, sheet_name=0):
    try:
        # Use pandas to read the Excel file
        df = pd.read_excel(myExcelFile, sheet_name=sheet_name)
        print(f"Successfully loaded Excel file: {myExcelFile}, Sheet: {sheet_name}")
        return df
    except FileNotFoundError:
        print(f"Error: Excel file not found at '{myExcelFile}'")
        return None
    except Exception as e:
        print(f"Error loading Excel file '{myExcelFile}' (Sheet: {sheet_name}): {e}")
        return None

def preprocessing_general_dataset_statistics(df):
    general_stats = {}
    general_stats['number_of_variables'] = df.shape[1]
    general_stats['number_of_observations'] = df.shape[0]
    general_stats['total_missing_cells'] = df.isnull().sum().sum()
    general_stats['total_missing_cells_pct'] = (general_stats['total_missing_cells'] / (df.shape[0] * df.shape[1])) * 100
    general_stats['total_size_in_memory_bytes'] = df.memory_usage(deep=True).sum()
    general_stats['total_size_in_memory_mb'] = general_stats['total_size_in_memory_bytes'] / (1024**2)

    # Number of each variable type
    general_stats['variable_types_count'] = df.dtypes.value_counts().to_dict()

    #return general_stats
    return pd.DataFrame([general_stats])

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

