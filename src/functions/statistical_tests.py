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
    """
    Performs an independent or paired t-test row by row on the data in a vectorized manner.

    Args:
        data: DataFrame with features in rows and samples in columns.
        group1_cols: List of column names for group 1.
        group2_cols: List of column names for group 2.
        paired: If True, performs a paired t-test.
        paired_map: Required if paired=True. A list of tuples mapping a sample from group 1 to group 2.

    Returns:
        A DataFrame with p-values and log2 fold change for each feature.
    """
    if paired:
        if not paired_map:
            raise ValueError("Paired t-test requires a 'paired_map'.")
        # Ensure the map is not empty
        if not paired_map:
            raise ValueError("The 'paired_map' cannot be empty for a paired t-test.")
        group1_cols_aligned = [item[0] for item in paired_map]
        group2_cols_aligned = [item[1] for item in paired_map]
        test_func = ttest_rel
    else:
        group1_cols_aligned, group2_cols_aligned = group1_cols, group2_cols
        test_func = ttest_ind

    # --- Vectorized Calculations ---
    group1_data = data[group1_cols_aligned]
    group2_data = data[group2_cols_aligned]

    # Log2 Fold Change (using original group columns for mean calculation)
    mean_group1 = data[group1_cols].mean(axis=1)
    mean_group2 = data[group2_cols].mean(axis=1)
    epsilon = 1e-10
    log2fc = np.log2(mean_group2 + epsilon) - np.log2(mean_group1 + epsilon)

    # Perform t-test along the rows (axis=1)
    # The nan_policy='omit' handles missing values within a row
    stat, p_val = test_func(group1_data, group2_data, axis=1, nan_policy='omit')

    # For rows where variance is zero in both groups, p-value can be NaN. Set to 1.
    p_val[(group1_data.var(axis=1) == 0) & (group2_data.var(axis=1) == 0)] = 1.0

    return pd.DataFrame({'p_value': p_val, 'log2FC': log2fc}, index=data.index)

def run_wilcoxon_rank_sum(data: pd.DataFrame, group1_cols: List[str], group2_cols: List[str]) -> pd.DataFrame:
    """
    Performs the Wilcoxon rank-sum test (Mann-Whitney U) row by row using .apply().
    """
    def _wilcoxon_row(row: pd.Series) -> float:
        group1_data = row[group1_cols].dropna()
        group2_data = row[group2_cols].dropna()

        if len(group1_data) < 1 or len(group2_data) < 1:
            return np.nan
        # If both groups have no variance, the test is not meaningful.
        if np.var(group1_data) == 0 and np.var(group2_data) == 0:
            return 1.0
        
        try:
            _, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            return p_val
        except ValueError:
            # This can happen if all values are identical
            return 1.0

    p_values = data.apply(_wilcoxon_row, axis=1)

    # Calculate log2FC based on medians for Wilcoxon
    median_group1 = data[group1_cols].median(axis=1)
    median_group2 = data[group2_cols].median(axis=1)
    epsilon = 1e-10
    log2fc = np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon)

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc}, index=data.index)

def run_anova(data: pd.DataFrame, group_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Performs one-way ANOVA row by row, returning p-value and eta-squared.
    This function is optimized to use .apply() for better performance.
    """
    def _anova_row(row: pd.Series) -> Tuple[float, float]:
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        
        if any(len(g) < 2 for g in groups_data):
            return np.nan, np.nan

        # If all groups have no variance, ANOVA is not meaningful.
        if all(np.var(g) == 0 for g in groups_data if len(g) > 0):
            return 1.0, 0.0

        # ANOVA
        try:
            _, p_val = f_oneway(*groups_data)
        except ValueError:
            return np.nan, np.nan

        # Effect Size (Eta Squared)
        all_vals = np.concatenate([g for g in groups_data if len(g) > 0])
        if len(all_vals) == 0:
            return p_val, 0.0
            
        ss_between = sum(len(g) * (np.mean(g) - np.mean(all_vals))**2 for g in groups_data if len(g) > 0)
        ss_total = np.sum((all_vals - np.mean(all_vals))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return p_val, eta_squared

    results = data.apply(lambda row: _anova_row(row), axis=1, result_type='expand')
    results.columns = ['p_value', 'eta_squared']

    # Calculate log2FC for ANOVA
    log2fc_values = []
    epsilon = 1e-10
    for index, row in data.iterrows():
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        
        if len(groups_data) == 2: # If exactly two groups, calculate FC between them
            mean_group1 = np.mean(groups_data[0]) if len(groups_data[0]) > 0 else np.nan
            mean_group2 = np.mean(groups_data[1]) if len(groups_data[1]) > 0 else np.nan
            if pd.isna(mean_group1) or pd.isna(mean_group2):
                log2fc_values.append(np.nan)
            else:
                log2fc_values.append(np.log2(mean_group2 + epsilon) - np.log2(mean_group1 + epsilon))
        elif len(groups_data) > 2: # If more than two groups, calculate FC relative to overall mean
            all_vals = np.concatenate([g for g in groups_data if len(g) > 0])
            if len(all_vals) == 0:
                log2fc_values.append(np.nan)
            else:
                overall_mean = np.mean(all_vals)
                # For simplicity, take the mean of the first group vs overall mean as a proxy
                # A more sophisticated approach might involve pairwise comparisons or specific contrasts
                mean_first_group = np.mean(groups_data[0]) if len(groups_data[0]) > 0 else np.nan
                if pd.isna(mean_first_group) or overall_mean == 0:
                    log2fc_values.append(np.nan)
                else:
                    log2fc_values.append(np.log2(mean_first_group + epsilon) - np.log2(overall_mean + epsilon))
        else:
            log2fc_values.append(np.nan)

    results['log2FC'] = log2fc_values

    return results

def format_anova_results_html(data: pd.DataFrame, group_map: Dict[str, List[str]], anova_results: pd.DataFrame) -> pd.DataFrame:
    """
    Formats ANOVA results with post-hoc tests and group stats into an HTML-ready DataFrame.
    This function should be called *after* run_anova.
    """
    formatted_results = []
    for index, row in data.iterrows():
        p_val = anova_results.loc[index, 'p_value']
        
        post_hoc_html = 'Not run (p >= 0.05)'
        if pd.notna(p_val) and p_val < 0.05:
            try:
                groups_data = [row[cols].dropna() for cols in group_map.values()]
                all_data = np.concatenate(groups_data)
                group_labels = np.concatenate([[name] * len(g) for name, g in zip(group_map.keys(), groups_data)])
                
                if len(np.unique(group_labels)) > 1:
                    tukey_res = pairwise_tukeyhsd(all_data, group_labels, alpha=0.05)
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

def run_permanova(data: pd.DataFrame, group_vector: Dict, distance_metric: str = 'euclidean', permutations: int = 999) -> Dict[str, Any]:
    """
    Performs PERMANOVA on the entire dataset.

    Args:
        data: DataFrame with features in rows and samples in columns.
        group_vector: Dictionary mapping sample names to their group info.
        distance_metric: The distance metric to use (any valid scipy.spatial.distance.pdist metric).
        permutations: The number of permutations to perform.

    Returns:
        A dictionary containing the PERMANOVA results.
    """
    if permutations < 100:
        raise ValueError("A minimum of 100 permutations is required for a meaningful test.")

    df_t = data.T
    
    grouping, samples_with_groups = [], []
    for sample in df_t.index:
        if sample in group_vector and group_vector[sample].get('groups'):
            grouping.append(group_vector[sample]['groups'][0])
            samples_with_groups.append(sample)

    if not grouping:
        raise ValueError("No samples with group assignments found for PERMANOVA.")
    if len(set(grouping)) < 2:
        raise ValueError("PERMANOVA requires at least two distinct groups for analysis.")

    df_t_filtered = df_t.loc[samples_with_groups]

    try:
        distance_array = pdist(df_t_filtered.values, metric=distance_metric)
        dm = DistanceMatrix(squareform(distance_array), ids=df_t_filtered.index)
    except Exception as e:
        raise ValueError(f"Failed to compute distance matrix with metric '{distance_metric}'. Error: {e}")
    
    result = permanova(dm, grouping, permutations=permutations)
    return dict(result)

def apply_multiple_test_correction(p_values: pd.Series, method: str = 'fdr_bh', alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    """
    Applies multiple testing correction to a Series of p-values.
    """
    p_array = p_values.dropna().to_numpy()
    if len(p_array) == 0:
        return pd.Series(index=p_values.index, dtype=float), pd.Series(index=p_values.index, dtype=bool)
    
    rejected, p_adj, _, _ = multipletests(p_array, alpha=alpha, method=method)
    
    p_adj_series = pd.Series(p_adj, index=p_values.dropna().index)
    rejected_series = pd.Series(rejected, index=p_values.dropna().index)
    
    return p_adj_series.reindex(p_values.index), rejected_series.reindex(p_values.index)
