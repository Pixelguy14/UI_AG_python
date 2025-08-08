"""
This module contains functions for performing statistical tests on metabolomics data.
Each public function is designed to take a DataFrame and group information,
perform a specific statistical test, and return the results as a DataFrame.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, f_oneway, kruskal # type: ignore
from statsmodels.stats.multitest import multipletests # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd # type: ignore
import statsmodels.formula.api as smf # type: ignore
from skbio.stats.distance import permanova, DistanceMatrix # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from typing import List, Dict, Any, Tuple, Optional # type: ignore

def run_t_test(
    data: pd.DataFrame, 
    group1_cols: List[str], 
    group2_cols: List[str], 
    paired: bool = False, 
    paired_map: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    # Validate paired_map if needed
    if paired:
        if not paired_map:
            raise ValueError("Paired t-test requires a non-empty 'paired_map'.")
        group1_cols_aligned = [item[0] for item in paired_map]
        group2_cols_aligned = [item[1] for item in paired_map]
        test_func = ttest_rel
    else:
        group1_cols_aligned, group2_cols_aligned = group1_cols, group2_cols
        test_func = ttest_ind

    # Extract aligned data
    group1_data = data[group1_cols_aligned]
    group2_data = data[group2_cols_aligned]

    # Log2FC with consistent samples & NaN skipping
    mean_group1 = group1_data.mean(axis=1, skipna=True)
    mean_group2 = group2_data.mean(axis=1, skipna=True)
    epsilon = 1e-10
    log2fc = np.log2(mean_group2 + epsilon) - np.log2(mean_group1 + epsilon)

    # T-test (handles NaNs internally)
    stat, p_val = test_func(group1_data, group2_data, axis=1, nan_policy='omit')

    # Handle zero-variance cases
    var_group1 = group1_data.var(axis=1, skipna=True)
    var_group2 = group2_data.var(axis=1, skipna=True)
    mask_both_zero_var = (var_group1 == 0) & (var_group2 == 0)
    if mask_both_zero_var.any():
        mean_equal = np.isclose(mean_group1, mean_group2, atol=1e-8)
        p_val = p_val.copy()  # Avoid mutation issues
        p_val[mask_both_zero_var & mean_equal] = 1.0
        p_val[mask_both_zero_var & ~mean_equal] = 0.0

    return pd.DataFrame({'p_value': p_val, 'log2FC': log2fc}, index=data.index)

def run_wilcoxon_rank_sum(
    data: pd.DataFrame, 
    group1_cols: List[str], 
    group2_cols: List[str]
) -> pd.DataFrame:
    # Vectorized NaN handling
    group1_mask = data[group1_cols].notna()
    group2_mask = data[group2_cols].notna()
    
    # Precompute medians for log2FC
    median_group1 = data[group1_cols].median(axis=1, skipna=True)
    median_group2 = data[group2_cols].median(axis=1, skipna=True)
    epsilon = 1e-10
    log2fc = np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon)

    # Vectorized p-value calculation
    p_values = []
    for idx, row in data.iterrows():
        g1 = row[group1_cols][group1_mask.loc[idx]].values
        g2 = row[group2_cols][group2_mask.loc[idx]].values
        
        if len(g1) < 1 or len(g2) < 1:
            p_values.append(np.nan)
            continue
            
        # Warn on small sample sizes
        if len(g1) < 3 or len(g2) < 3:
            raise ValueError(f"Small sample size (n1={len(g1)}, n2={len(g2)}) for index {idx}")
            
        try:
            _, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            p_values.append(p_val)
        except Exception as e:  # Narrow exception handling
            p_values.append(np.nan)
            raise ValueError(f"Error at index {idx}: {str(e)}")

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc}, index=data.index)

def run_anova(data: pd.DataFrame, group_map: Dict[str, List[str]]) -> pd.DataFrame:
    # Precompute group information
    group_names = list(group_map.keys())
    # group_cols = list(group_map.values())
    group_indices = {name: data.columns.get_indexer(cols) for name, cols in group_map.items()}

    # Preallocate results
    p_values = np.full(len(data), np.nan)
    eta_squared = np.full(len(data), np.nan)
    
    for i, row in enumerate(data.itertuples(index=False)):
        row_data = np.array(row)
        group_arrays = []
        valid_groups = []
        
        # Extract group data with NaN handling
        for name in group_names:
            idx = group_indices[name]
            group_vals = row_data[idx]
            valid_mask = ~np.isnan(group_vals)
            
            # Only consider groups with at least 2 valid samples
            if np.sum(valid_mask) >= 2:
                group_arrays.append(group_vals[valid_mask])
                valid_groups.append(name)
        
        # Skip if less than 2 valid groups for ANOVA
        if len(valid_groups) < 2:
            continue
        
        # Check for zero variance within any group
        if any(np.var(g) == 0 for g in group_arrays):
            # If any group has zero variance, check if all means are equal
            if all(np.isclose(np.mean(g), np.mean(group_arrays[0])) for g in group_arrays):
                p_values[i] = 1.0 # All means are equal, p-value is 1
            else:
                p_values[i] = 0.0 # Means are not equal, p-value is 0
            eta_squared[i] = 0.0 # No variance explained if means are equal or only one group varies
            continue # Skip to next row
            
        try:
            # Perform ANOVA
            f_stat, p_val = f_oneway(*group_arrays)
            p_values[i] = p_val
            
            # Calculate effect size
            all_vals = np.concatenate(group_arrays)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_arrays)
            ss_total = np.sum((all_vals - grand_mean)**2)
            eta_squared[i] = ss_between / ss_total if ss_total > 0 else 0
        except ValueError as e:
            # Catch specific ValueError from f_oneway (e.g., if all values are identical after NaN removal)
            # Log the error for debugging, but continue processing other rows
            import logging
            logging.warning(f"ANOVA ValueError for row {data.index[i]}: {e}")
            p_values[i] = np.nan
            eta_squared[i] = np.nan
        except Exception as e:
            # Catch any other unexpected errors
            import logging
            logging.error(f"Unexpected error during ANOVA for row {data.index[i]}: {e}", exc_info=True)
            p_values[i] = np.nan
            eta_squared[i] = np.nan

    return pd.DataFrame({
        'p_value': p_values,
        'eta_squared': eta_squared
    }, index=data.index)

def format_anova_results_html(
    data: pd.DataFrame,
    group_map: Dict[str, List[str]],
    anova_results: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Formats ANOVA results with post-hoc tests and group stats into an HTML-ready DataFrame.
    This function should be called *after* run_anova.
    """
    formatted_results = []
    for index, row in data.iterrows():
        p_val_series = anova_results.loc[index, 'p_value']
        # If p_val_series is a Series with multiple items (due to non-unique index), take the first one.
        # This is a pragmatic fix; ideally, the DataFrame index should be unique.
        p_val = p_val_series.iloc[0] if isinstance(p_val_series, pd.Series) else p_val_series
        
        post_hoc_html = 'Not run (p >= 0.05)'
        if pd.notna(p_val) and p_val < alpha:
            try:
                groups_data = [row[cols].dropna() for cols in group_map.values()]
                all_data = np.concatenate(groups_data)
                group_labels = np.concatenate([[name] * len(g) for name, g in zip(group_map.keys(), groups_data)])
                
                # Add check for sufficient samples for Tukey HSD
                if any(len(g) < 2 for g in groups_data):
                    post_hoc_html = "<div class='text-warning'>Tukey HSD requires at least 2 samples per group.</div>"
                elif len(np.unique(group_labels)) > 1:
                    tukey_res = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
                    post_hoc_df = pd.DataFrame(data=tukey_res._results_table.data[1:], columns=tukey_res._results_table.data[0])
                    
                    html_items = [
                        f"<li><strong>{r['group1']} vs {r['group2']}:</strong> {r['p-adj']:.3f}{'*' if r['reject'] else ''}</li>"
                        for _, r in post_hoc_df.iterrows()
                    ]
                    post_hoc_html = f"<ul class='list-unstyled mb-0'>{''.join(html_items)}</ul>"
                else:
                    post_hoc_html = "Only one group with data."
            except Exception as e:
                post_hoc_html = f"<span class='text-danger'>Error: {e}</span>"

        # Group statistics
        stats_html_items = []
        for name, cols in group_map.items():
            g = row[cols].dropna()
            if len(g) > 0:
                stats_html_items.append(f"<li><b>{name}:</b> mean={np.mean(g):.3f}, std={np.std(g):.3f}</li>")
        group_stats_html = f"<ul class='list-unstyled mb-0'>{''.join(stats_html_items)}</ul>"

        formatted_results.append({
            'post_hoc_tukey': post_hoc_html,
            'group_stats': group_stats_html
        })

    # Combine with original ANOVA results
    formatted_df = pd.DataFrame(formatted_results, index=data.index)
    return anova_results.join(formatted_df)

def run_kruskal_wallis(data: pd.DataFrame, group_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Performs the Kruskal-Wallis H-test row by row using .apply().
    """
    def _kruskal_row(row: pd.Series) -> float:
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        if any(len(g) < 1 for g in groups_data):
            return np.nan
        
        if all(np.var(g) == 0 for g in groups_data if len(g) > 0):
            return 1.0

        try:
            _, p_val = kruskal(*groups_data)
            return p_val
        except ValueError:
            return np.nan

    p_values = data.apply(_kruskal_row, axis=1)

    # Calculate log2FC for Kruskal-Wallis (similar logic to ANOVA for multi-group)
    log2fc_values = []
    epsilon = 1e-10
    for index, row in data.iterrows():
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        
        if len(groups_data) == 2: # If exactly two groups, calculate FC between their medians
            median_group1 = np.median(groups_data[0]) if len(groups_data[0]) > 0 else np.nan
            median_group2 = np.median(groups_data[1]) if len(groups_data[1]) > 0 else np.nan
            if pd.isna(median_group1) or pd.isna(median_group2):
                log2fc_values.append(np.nan)
            else:
                log2fc_values.append(np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon))
        elif len(groups_data) > 2: # If more than two groups, calculate FC relative to overall median
            all_vals = np.concatenate([g for g in groups_data if len(g) > 0])
            if len(all_vals) == 0:
                log2fc_values.append(np.nan)
            else:
                overall_median = np.median(all_vals)
                # For simplicity, take the median of the first group vs overall median as a proxy
                median_first_group = np.median(groups_data[0]) if len(groups_data[0]) > 0 else np.nan
                if pd.isna(median_first_group) or overall_median == 0:
                    log2fc_values.append(np.nan)
                else:
                    log2fc_values.append(np.log2(median_first_group + epsilon) - np.log2(overall_median + epsilon))
        else:
            log2fc_values.append(np.nan)

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc_values}, index=data.index)

def run_linear_model(data: pd.DataFrame, formula: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a linear model for each feature using .apply().
    """
    if 'value' in metadata.columns:
        raise ValueError("'value' is a reserved column name for this function. Please rename it in your metadata.")

    def _lm_row(row: pd.Series) -> Tuple[Optional[float], str]:
        # Combine metadata with the single feature's data
        model_df = metadata.copy()
        model_df['value'] = row.values
        
        try:
            model = smf.ols(formula, data=model_df).fit()
            # Try to get p-value for 'group', a common predictor, but handle its absence
            p_val = model.pvalues.get('group', np.nan)
            return p_val, str(model.summary())
        except Exception as e:
            return np.nan, str(e)

    results = data.apply(_lm_row, axis=1, result_type='expand')
    results.columns = ['p_value', 'model_summary']
    return results

def run_permanova(
    data: pd.DataFrame, 
    group_vector: Dict[str, Any],  # More flexible typing
    distance_metric: str = 'euclidean', 
    permutations: int = 999
) -> Dict[str, Any]:
    # Validate permutations
    if permutations < 100:
        raise ValueError("A minimum of 100 permutations is required for a meaningful test.")

    df_t = data.T  # Samples as rows, features as columns
    
    # Extract group assignments in consistent order
    samples_with_groups = []
    grouping = []
    for sample in df_t.index:
        if sample not in group_vector:
            continue
        group_info = group_vector[sample].get('groups')
        if group_info is None:
            continue
            
        # Handle both list and scalar group assignments
        if isinstance(group_info, list) and group_info:
            group_val = group_info[0]
        elif group_info:
            group_val = group_info
        else:  # Empty list/None
            continue
            
        samples_with_groups.append(sample)
        grouping.append(group_val)

    # Validate groups
    if not samples_with_groups:
        raise ValueError("No samples with group assignments found for PERMANOVA.")
    if len(set(grouping)) < 2:
        raise ValueError("PERMANOVA requires at least two distinct groups for analysis.")

    # Filter data and preserve order
    df_t_filtered = df_t.loc[samples_with_groups]
    
    # Check for NaNs
    if df_t_filtered.isnull().any().any():
        raise ValueError("PERMANOVA cannot handle NaN values in data")

    # Compute distance matrix
    try:
        distance_array = pdist(df_t_filtered.values, metric=distance_metric)
        dm = DistanceMatrix(squareform(distance_array), ids=df_t_filtered.index)
    except Exception as e:
        raise ValueError(f"Distance matrix computation failed: {str(e)}")

    # Run PERMANOVA
    result = permanova(dm, grouping, permutations=permutations)
    return dict(result)

def apply_multiple_test_correction(
    p_values: pd.Series, 
    method: str = 'fdr_bh', 
    alpha: float = 0.05
) -> Tuple[pd.Series, pd.Series]:
    # Drop NaNs and non-finite values
    finite_p_values = p_values.dropna()
    finite_p_values = finite_p_values[np.isfinite(finite_p_values)]
    
    if len(finite_p_values) == 0:
        p_adj = pd.Series(index=p_values.index, dtype=float)
        rejected = pd.Series(index=p_values.index, dtype=bool)
        return p_adj, rejected
    
    # Apply correction
    rejected, p_adj, _, _ = multipletests(
        finite_p_values, alpha=alpha, method=method
    )
    
    # Reindex to original
    p_adj_series = pd.Series(p_adj, index=finite_p_values.index)
    rejected_series = pd.Series(rejected, index=finite_p_values.index).reindex(p_values.index, fill_value=False) # Fill NaNs with False
    return p_adj_series.reindex(p_values.index), rejected_series
