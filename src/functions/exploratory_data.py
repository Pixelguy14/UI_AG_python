import pyopenms as oms
import pandas as pd
import os
import numpy as np

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
    else:
        return pd.DataFrame() # Explicitly return empty dataframe

def loadDF_Consensus(myConsensusXML):
    try:
        consensus_map = oms.ConsensusMap()
        oms.ConsensusXMLFile().load(myConsensusXML, consensus_map)

        # Extract metadata from original file
        column_headers = consensus_map.getColumnHeaders()
        sorted_columns = sorted(column_headers.items(), key=lambda x: x[0])
        filenames = [os.path.basename(header.filename) for idx, header in sorted_columns]

        # Collect all unique meta value keys across all consensus features
        all_keys = []
        for cf in consensus_map:
            keys = []
            cf.getKeys(keys)  # Correct usage with output parameter
            all_keys.extend(keys)
        
        meta_keys = sorted(set(k.decode() if isinstance(k, bytes) else k for k in all_keys))
        
        # Build dataframe
        rows = []
        for cf in consensus_map:
            row = {
                'rt': cf.getRT(),
                'mz': cf.getMZ(),
                'intensity': cf.getIntensity(),
                'quality': cf.getQuality(),
                'charge': cf.getCharge(),
                'size': cf.size()  # Number of constituent features
            }
            
            # Add meta values
            for key in meta_keys:
                # Convert key to appropriate type for checking
                lookup_key = key.encode() if isinstance(key, str) else key
                if cf.metaValueExists(lookup_key):
                    value = cf.getMetaValue(lookup_key)
                    # Handle OpenMS DataValue types
                    if isinstance(value, oms.DataValue):
                        if value.valueType() == oms.DataValue.INT_VALUE:
                            row[key] = int(value)
                        elif value.valueType() == oms.DataValue.DOUBLE_VALUE:
                            row[key] = float(value)
                        else:
                            row[key] = str(value)
                    else:
                        row[key] = str(value)
                else:
                    row[key] = np.nan
            
            # Initialize intensities as NaN
            for filename in filenames:
                row[filename] = float('nan')
            
            # Fill intensities for each file
            for fh in cf.getFeatureList():
                map_idx = fh.getMapIndex()
                if map_idx < len(filenames):
                    filename = filenames[map_idx]
                    row[filename] = fh.getIntensity()
            
            rows.append(row)
        
        # Create the dataframe
        df = pd.DataFrame(rows)
        
        # Define column order: basic info + meta keys + filenames
        base_columns = ['rt', 'mz', 'intensity', 'quality', 'charge', 'size']
        columns = base_columns + meta_keys + filenames
        
        # Select only existing columns
        existing_columns = [col for col in columns if col in df.columns]
        return df[existing_columns]
        
    except Exception as e:
        print(f"Error loading ConsensusXML file '{myConsensusXML}': {e}")
        return None

def loadDF_Feature(myFeatureXML):
    try:
        feature_map = oms.FeatureMap()
        oms.FeatureXMLFile().load(myFeatureXML, feature_map)

        # Collect all unique meta value keys across all features
        all_keys = []
        for feature in feature_map:
            keys = []
            feature.getKeys(keys)  # Correct usage with output parameter
            all_keys.extend(keys)
        
        meta_keys = sorted(set(k.decode() if isinstance(k, bytes) else k for k in all_keys))
        
        # Get file metadata
        file_meta = {}
        map_keys = []
        feature_map.getKeys(map_keys)
        for key in map_keys:
            k = key.decode() if isinstance(key, bytes) else key
            value = feature_map.getMetaValue(key)
            if isinstance(value, oms.DataValue):
                if value.valueType() == oms.DataValue.STRING_VALUE:
                    file_meta[k] = str(value)
                elif value.valueType() == oms.DataValue.INT_VALUE:
                    file_meta[k] = int(value)
                elif value.valueType() == oms.DataValue.DOUBLE_VALUE:
                    file_meta[k] = float(value)
                else:
                    file_meta[k] = str(value)
            else:
                file_meta[k] = str(value)
        
        rows = []
        for feature in feature_map:
            row = {
                'rt': feature.getRT(),
                'mz': feature.getMZ(),
                'intensity': feature.getIntensity(),
                'quality': feature.getQuality(),
                'charge': feature.getCharge(),
                'width': feature.getWidth()
            }
            
            # Add convex hull points count
            hulls = feature.getConvexHulls()
            if hulls:
                row['hull_points'] = sum(len(hull.getHullPoints()) for hull in hulls)
            else:
                row['hull_points'] = 0
            
            # Add meta values
            for key in meta_keys:
                # Convert key to appropriate type for checking
                lookup_key = key.encode() if isinstance(key, str) else key
                if feature.metaValueExists(lookup_key):
                    value = feature.getMetaValue(lookup_key)
                    if isinstance(value, oms.DataValue):
                        if value.valueType() == oms.DataValue.INT_VALUE:
                            row[key] = int(value)
                        elif value.valueType() == oms.DataValue.DOUBLE_VALUE:
                            row[key] = float(value)
                        else:
                            row[key] = str(value)
                    else:
                        row[key] = str(value)
                else:
                    row[key] = np.nan
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add file-level metadata as new columns
        for key, value in file_meta.items():
            df[key] = value
        
        # Define preferred column order
        base_columns = ['rt', 'mz', 'intensity', 'quality', 'charge', 'width', 'hull_points']
        columns = base_columns + meta_keys + list(file_meta.keys())
        
        # Select only existing columns
        existing_columns = [col for col in columns if col in df.columns]
        return df[existing_columns]
        
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
    
def preprocessing_general_dataset_statistics(df):
    if df.empty:
        return pd.DataFrame()
    
    general_stats = {}
    
    # Basic dataset information
    general_stats['number_of_variables'] = df.shape[1]
    general_stats['number_of_observations'] = df.shape[0]
    general_stats['total_missing_cells'] = df.isnull().sum().sum()
    general_stats['total_missing_cells_%'] = (general_stats['total_missing_cells'] / 
                                             (df.shape[0] * df.shape[1])) * 100
    general_stats['total_size_in_memory_bytes'] = df.memory_usage(deep=True).sum()
    general_stats['total_size_in_memory_mb'] = general_stats['total_size_in_memory_bytes'] / (1024**2)
    
    # Data type distribution
    type_counts = df.dtypes.value_counts().to_dict()
    for dtype, count in type_counts.items():
        general_stats[f'variables_{dtype}'] = count
    
    # Missing values details
    general_stats['variables_with_missing_values'] = df.isnull().any().sum()
    general_stats['observations_with_missing_values'] = df.isnull().any(axis=1).sum()
    
    # Descriptive statistics (similar to df.describe())
    if df.select_dtypes(include=np.number).shape[1] > 0:
        num_stats = df.describe(include=[np.number]).mean(axis=1).to_dict()
        for stat, value in num_stats.items():
            general_stats[f'avg_{stat}_over_numeric_cols'] = value
    
    if df.select_dtypes(exclude=np.number).shape[1] > 0:
        cat_describe = df.describe(exclude=[np.number])
        # The 'top' row contains non-numeric data, so we drop it before calculating the mean.
        # The remaining columns are still 'object' dtype, so we convert to float before calculating the mean.
        if 'top' in cat_describe.index:
            cat_stats = cat_describe.drop('top').astype(float).mean(axis=1).to_dict()
            for stat, value in cat_stats.items():
                general_stats[f'avg_{stat}_over_categorical_cols'] = value

    # Quantile information
    # Only calculate quantiles for numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        quantiles = numeric_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        for q in quantiles.index:
            q_stats = quantiles.loc[q].mean()
            general_stats[f'avg_{int(q*100)}_percentile'] = q_stats
    
    # Create DataFrame with sorted columns for better readability
    result_df = pd.DataFrame([general_stats])
    
    # Sort columns logically
    preferred_order = [
        'number_of_variables', 'number_of_observations',
        'total_missing_cells', 'total_missing_cells_%',
        'variables_with_missing_values', 'observations_with_missing_values',
        'total_size_in_memory_bytes', 'total_size_in_memory_mb'
    ]
    
    # Add remaining columns
    remaining_cols = [col for col in result_df.columns if col not in preferred_order]
    sorted_cols = preferred_order + sorted(remaining_cols)
    
    return result_df[sorted_cols]

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