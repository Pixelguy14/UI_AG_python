"""
This module contains functions for performing statistical tests on metabolomics data.
Each public function is designed to take a DataFrame and group information,
perform a specific statistical test, and return the results as a DataFrame.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, f_oneway, kruskal, shapiro, friedmanchisquare, kstest # type: ignore
from statsmodels.stats.multitest import multipletests # type: ignore
from statsmodels.stats.diagnostic import lilliefors # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison # type: ignore
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf # type: ignore
from skbio.stats.distance import permanova, DistanceMatrix # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from typing import List, Dict, Any, Tuple, Optional # type: ignore

def run_t_test(
    data: pd.DataFrame, 
    group1_cols: List[str], 
    group2_cols: List[str], 
    paired: bool = False, 
    paired_map: Optional[List[Tuple[str, str]]] = None,
    is_log_transformed: bool = False
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
    mean_group1 = group1_data.mean(axis=1, skipna=True) # type: ignore
    mean_group2 = group2_data.mean(axis=1, skipna=True) # type: ignore
    
    if is_log_transformed:
        log2fc = mean_group2 - mean_group1
    else:
        epsilon = 1
        log2fc = np.log2(mean_group2 + epsilon) - np.log2(mean_group1 + epsilon)

    # T-test (handles NaNs internally)
    stat, p_val = test_func(group1_data, group2_data, axis=1, nan_policy='omit')

    # Handle zero-variance cases
    var_group1 = group1_data.var(axis=1, skipna=True) # type: ignore
    var_group2 = group2_data.var(axis=1, skipna=True) # type: ignore
    mask_both_zero_var = (var_group1 == 0) & (var_group2 == 0)
    if mask_both_zero_var.any(): # type: ignore
        mean_equal = np.isclose(mean_group1, mean_group2, atol=1e-8) # type: ignore
        p_val = p_val.copy()  # Avoid mutation issues
        p_val[mask_both_zero_var & mean_equal] = 1.0
        p_val[mask_both_zero_var & ~mean_equal] = 0.0

    return pd.DataFrame({'p_value': p_val, 'log2FC': log2fc}, index=data.index)

def run_mann_whitney_u(
    data: pd.DataFrame, 
    group1_cols: List[str], 
    group2_cols: List[str],
    is_log_transformed: bool = False
) -> pd.DataFrame:
    # Vectorized NaN handling
    group1_mask = data[group1_cols].notna()
    group2_mask = data[group2_cols].notna()
    
    # Precompute medians for log2FC
    median_group1 = data[group1_cols].median(axis=1, skipna=True) # type: ignore
    median_group2 = data[group2_cols].median(axis=1, skipna=True) # type: ignore
    
    if is_log_transformed:
        log2fc = median_group2 - median_group1
    else:
        epsilon = 1
        log2fc = np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon)

    # Vectorized p-value calculation
    p_values = []
    for idx, row in data.iterrows():
        g1 = row[group1_cols][group1_mask.loc[idx]].values
        g2 = row[group2_cols][group2_mask.loc[idx]].values
        
        if len(g1) < 1 or len(g2) < 1:
            p_values.append(np.nan)
            continue
            
        # For small sample sizes, the test is not meaningful.
        if len(g1) < 3 or len(g2) < 3:
            p_values.append(np.nan)
            continue
            
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
        if any(np.var(g) == 0 for g in group_arrays): # type: ignore
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
    alpha: float = 0.05,
    posthoc_method: str = 'tukeyhsd'
) -> pd.DataFrame:
    """
    Formats ANOVA results with post-hoc tests and group stats into an HTML-ready DataFrame.
    This function should be called *after* run_anova.
    """
    formatted_results = []
    for index, row in data.iterrows():
        p_val_series = anova_results.loc[index, 'p_value']
        p_val = p_val_series.iloc[0] if isinstance(p_val_series, pd.Series) else p_val_series
        
        post_hoc_html = 'Not run (p >= 0.05)'
        if pd.notna(p_val) and p_val < alpha:
            try:
                groups_data = [row[cols].dropna() for cols in group_map.values()]
                all_data = np.concatenate([g.to_numpy(dtype=float) for g in groups_data])
                group_labels = np.concatenate([[name] * len(g) for name, g in zip(group_map.keys(), groups_data)])
                
                if any(len(g) < 2 for g in groups_data):
                    post_hoc_html = f"<div class='text-warning'>Post-hoc test requires at least 2 samples per group.</div>"
                elif len(np.unique(group_labels)) > 1:
                    if posthoc_method == 'tukeyhsd':
                        tukey_res = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
                        post_hoc_df = pd.DataFrame(data=tukey_res._results_table.data[1:], columns=tukey_res._results_table.data[0])
                        html_items = [
                            f"<li><strong>{r['group1']} vs {r['group2']}:</strong> {r['p-adj']:.3f}{'*' if r['reject'] else ''}</li>"
                            for _, r in post_hoc_df.iterrows()
                        ]
                        post_hoc_html = f"<h6>Tukey's HSD</h6><ul class='list-unstyled mb-0'>{''.join(html_items)}</ul>"
                    else: # For bonferroni, holm, etc.
                        mc = MultiComparison(all_data, group_labels)
                        results_table, _, _ = mc.allpairtest(alpha=alpha, method=posthoc_method)
                        df = pd.DataFrame(results_table.data[1:], columns=[str(x) for x in results_table.data[0]])
                        
                        html_items = [
                            f"<li><strong>{r.group1} vs {r.group2}:</strong> {r.pval_corr:.3f}{'*' if r.reject else ''}</li>"
                            for r in df.itertuples()
                        ]
                        post_hoc_html = f"<h6>{posthoc_method.capitalize()} Correction</h6><ul class='list-unstyled mb-0'>{''.join(html_items)}</ul>"
                else:
                    post_hoc_html = "Only one group with data."
            except Exception as e:
                post_hoc_html = f"<span class='text-danger'>Error: {e}</span>"

        # Group statistics
        stats_html_items = []
        for name, cols in group_map.items():
            g = row[cols].dropna()
            if len(g) > 0:
                stats_html_items.append(f"<li><b>{name}:</b> mean={np.mean(g.to_numpy(dtype=float)):.3f}, std={np.std(g.to_numpy(dtype=float)):.3f}</li>")
        group_stats_html = f"<ul class='list-unstyled mb-0'>{''.join(stats_html_items)}</ul>"

        formatted_results.append({
            'post_hoc': post_hoc_html, # Changed column name to be generic
            'group_stats': group_stats_html
        })

    # Combine with original ANOVA results
    formatted_df = pd.DataFrame(formatted_results, index=data.index)
    # Drop old column if it exists to avoid conflicts
    if 'post_hoc_tukey' in anova_results.columns:
        anova_results = anova_results.drop(columns=['post_hoc_tukey'])
    return anova_results.join(formatted_df)

def run_kruskal_wallis(data: pd.DataFrame, group_map: Dict[str, List[str]], is_log_transformed: bool = False) -> pd.DataFrame:
    """
    Performs the Kruskal-Wallis H-test row by row using .apply().
    """
    def _kruskal_row(row: pd.Series) -> float:
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        if any(len(g) < 1 for g in groups_data):
            return np.nan
        
        if all(np.var(g) == 0 for g in groups_data if len(g) > 0): # type: ignore
            return 1.0

        try:
            _, p_val = kruskal(*groups_data)
            return p_val
        except ValueError:
            return np.nan

    p_values = data.apply(_kruskal_row, axis=1)

    # Calculate log2FC for Kruskal-Wallis (similar logic to ANOVA for multi-group)
    log2fc_values = []
    epsilon = 1
    for index, row in data.iterrows():
        groups_data = [row[cols].dropna() for cols in group_map.values()]
        
        if len(groups_data) == 2: # If exactly two groups, calculate FC between their medians
            median_group1 = np.median(groups_data[0].to_numpy(dtype=float)) if len(groups_data[0]) > 0 else np.nan
            median_group2 = np.median(groups_data[1].to_numpy(dtype=float)) if len(groups_data[1]) > 0 else np.nan
            if pd.isna(median_group1) or pd.isna(median_group2):
                log2fc_values.append(np.nan)
            else:
                if is_log_transformed:
                    log2fc_values.append(median_group2 - median_group1)
                else:
                    log2fc_values.append(np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon))
        elif len(groups_data) > 2: # If more than two groups, calculate FC relative to overall median
            all_vals = np.concatenate([g.to_numpy(dtype=float) for g in groups_data if len(g) > 0])
            if len(all_vals) == 0:
                log2fc_values.append(np.nan)
            else:
                overall_median = np.median(all_vals)
                # For simplicity, take the median of the first group vs overall median as a proxy
                median_first_group = np.median(groups_data[0].to_numpy(dtype=float)) if len(groups_data[0]) > 0 else np.nan
                if pd.isna(median_first_group) or overall_median == 0:
                    log2fc_values.append(np.nan)
                else:
                    if is_log_transformed:
                        log2fc_values.append(median_first_group - overall_median)
                    else:
                        log2fc_values.append(np.log2(median_first_group + epsilon) - np.log2(overall_median + epsilon))
        else:
            log2fc_values.append(np.nan)

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc_values}, index=data.index)

def run_friedman_test(
    data: pd.DataFrame,
    paired_map: List[List[str]],
    is_log_transformed: bool = False
) -> pd.DataFrame:
    """
    Performs the Friedman test row by row for paired multi-group data.
    The paired_map defines the mapping of samples for each subject across groups.
    """
    p_values = []
    log2fc_values = [] # For multi-group, log2FC is often relative to a baseline or overall mean
    epsilon = 1 # For log2FC calculation

    if not paired_map:
        raise ValueError("Friedman test requires a non-empty 'paired_map'.")

    # Determine the number of groups from the paired_map (length of inner list)
    num_groups = len(paired_map[0])
    if num_groups < 3:
        raise ValueError("Friedman test requires at least 3 related groups.")

    # Extract group names from the first entry of paired_map for log2FC calculation
    # This assumes the order of samples in paired_map corresponds to the order of groups
    # as selected by the user. For log2FC, we'll compare the last group to the first group.
    # This is a simplification; a more robust approach might involve defining a reference group.
    group_sample_names_ordered = [paired_map[0][i] for i in range(num_groups)]

    for index, row in data.iterrows():
        subject_data_for_friedman = []
        valid_subjects_count = 0

        # Collect data for each subject across all groups
        for subject_samples in paired_map:
            values_for_subject = []
            is_subject_valid = True
            for sample_name in subject_samples:
                if sample_name in row and pd.notna(row[sample_name]):
                    values_for_subject.append(row[sample_name])
                else:
                    is_subject_valid = False
                    break # This subject is missing data for one of the groups
            
            if is_subject_valid and len(values_for_subject) == num_groups:
                subject_data_for_friedman.append(values_for_subject)
                valid_subjects_count += 1
        
        if valid_subjects_count < 1: # Friedman requires at least one complete subject
            p_values.append(np.nan)
            log2fc_values.append(np.nan)
            continue

        # Transpose data for friedmanchisquare: each list is a group's values across subjects
        # e.g., [group1_s1, group1_s2, ...], [group2_s1, group2_s2, ...]
        friedman_input_groups = [[] for _ in range(num_groups)]
        for subject_values in subject_data_for_friedman:
            for i, val in enumerate(subject_values):
                friedman_input_groups[i].append(val)

        # Check if all groups have at least one observation
        if any(len(g) == 0 for g in friedman_input_groups):
            p_values.append(np.nan)
            log2fc_values.append(np.nan)
            continue

        try:
            _, p_val = friedmanchisquare(*[np.array(g) for g in friedman_input_groups])
            p_values.append(p_val)
        except ValueError: # e.g., if all values are identical within a group
            p_values.append(1.0) # Or np.nan, depending on desired behavior for no variance
        except Exception as e:
            p_values.append(np.nan)
            import logging
            logging.error(f"Error during Friedman test for row {index}: {e}")

        # Calculate log2FC (e.g., last group vs first group median)
        median_first_group = np.median(friedman_input_groups[0])
        median_last_group = np.median(friedman_input_groups[-1])

        if is_log_transformed:
            log2fc = median_last_group - median_first_group
        else:
            log2fc = np.log2(median_last_group + epsilon) - np.log2(median_first_group + epsilon)
        log2fc_values.append(log2fc)

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc_values}, index=data.index)

def run_anova_rm(
    data: pd.DataFrame,
    paired_map: List[List[str]],
    is_log_transformed: bool = False
) -> pd.DataFrame:
    """
    Performs Repeated Measures ANOVA row by row for paired multi-group data.
    The paired_map defines the mapping of samples for each subject across groups.
    """
    p_values = []
    eta_squared_values = [] # Partial eta-squared for RM ANOVA
    epsilon = 1 # For log2FC calculation (though not directly used for ANOVA RM output)

    if not paired_map:
        raise ValueError("Repeated Measures ANOVA requires a non-empty 'paired_map'.")

    num_groups = len(paired_map[0])
    if num_groups < 2: # RM ANOVA can be used for 2 or more groups
        raise ValueError("Repeated Measures ANOVA requires at least 2 related groups.")

    # Extract group names from the first entry of paired_map
    group_sample_names_ordered = [paired_map[0][i] for i in range(num_groups)]

    for index, row in data.iterrows():
        long_data_for_feature = []
        
        # Collect data for each subject across all groups and transform to long format
        for subject_idx, subject_samples in enumerate(paired_map):
            for group_col_idx, sample_name in enumerate(subject_samples):
                if sample_name in row and pd.notna(row[sample_name]):
                    long_data_for_feature.append({
                        'value': row[sample_name],
                        'subject_id': f'S_{subject_idx}', # Unique ID for each subject
                        'group': f'G_{group_col_idx}' # Group identifier
                    })
        
        if not long_data_for_feature:
            p_values.append(np.nan)
            eta_squared_values.append(np.nan)
            continue

        df_long = pd.DataFrame(long_data_for_feature)

        # Ensure there's enough data for the model
        if len(df_long['subject_id'].unique()) < 1 or len(df_long['group'].unique()) < 2:
            p_values.append(np.nan)
            eta_squared_values.append(np.nan)
            continue

        try:
            # Fit the Repeated Measures ANOVA model using statsmodels
            # Formula: value ~ C(group) + C(subject_id)
            # C(group) is the fixed effect we are interested in
            # C(subject_id) accounts for the within-subject variability
            model = smf.ols('value ~ C(group)', data=df_long).fit()
            # For repeated measures, we need to account for the subject factor
            # A more appropriate model for RM ANOVA is often Mixed Linear Model or using AnovaRM
            # However, for simplicity and common use cases, a basic OLS with subject as a fixed effect
            # can approximate, or we can use a more direct RM ANOVA implementation if available.
            # For now, let's use a simple OLS and extract the group p-value.
            # A proper RM ANOVA would involve `statsmodels.stats.anova.AnovaRM` or `mixedlm`.
            # Let's use `AnovaRM` for a more correct approach.

            # Ensure data is balanced for AnovaRM (same number of observations per subject per group)
            # This is implicitly handled by how long_data_for_feature is constructed from paired_map
            
            # Check for sufficient data points per subject per group
            counts = df_long.groupby(['subject_id', 'group']).size()
            if not all(c > 0 for c in counts): # Ensure no empty cells after grouping
                p_values.append(np.nan)
                eta_squared_values.append(np.nan)
                continue

            aovrm = AnovaRM(data=df_long, depvar='value', subject='subject_id', within=['group'])
            res = aovrm.fit()
            
            # Extract p-value for the 'group' effect
            p_val = res.anova_table.loc['group', 'Pr > F']
            p_values.append(p_val)

            # Calculate partial eta-squared for the 'group' effect
            # This requires SS_effect / (SS_effect + SS_error)
            # From the AnovaRM results, we can get F-value, df, MS, etc.
            # SS_group = MS_group * df_group
            # SS_error = MS_error * df_error
            # Partial Eta-squared = SS_group / (SS_group + SS_error)
            
            ss_group = res.anova_table.loc['group', 'sum_sq']
            ss_error = res.anova_table.loc['Residual', 'sum_sq']
            
            if (ss_group + ss_error) > 0:
                partial_eta_squared = ss_group / (ss_group + ss_error)
            else:
                partial_eta_squared = np.nan
            eta_squared_values.append(partial_eta_squared)

        except ValueError as e:
            # This can happen if there's not enough variance or data for the model
            p_values.append(np.nan)
            eta_squared_values.append(np.nan)
            import logging
            logging.warning(f"ValueError during AnovaRM for row {index}: {e}")
        except Exception as e:
            p_values.append(np.nan)
            eta_squared_values.append(np.nan)
            import logging
            logging.error(f"Error during AnovaRM for row {index}: {e}")

    return pd.DataFrame({'p_value': p_values, 'eta_squared': eta_squared_values}, index=data.index)

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
    finite_p_values = finite_p_values[np.isfinite(finite_p_values)] # type: ignore
    
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

def check_normality(data: pd.Series) -> Tuple[str, float]:
    """
    Performs an appropriate normality test based on sample size.
    - Shapiro-Wilk for N <= 5000
    - Lilliefors (Kolmogorov-Smirnov) for N > 5000
    Returns the test name and its p-value.
    """
    clean_data = data.dropna()
    n_samples = len(clean_data)

    if n_samples < 3:
        return "N/A", np.nan

    try:
        if n_samples <= 5000:
            # Shapiro-Wilk is generally more powerful for smaller samples
            stat, p_value = shapiro(clean_data)
            return "Shapiro-Wilk", p_value
        else:
            # For larger samples where parameters are estimated, Lilliefors test is appropriate.
            stat, p_value = lilliefors(clean_data, dist='norm')
            return "Lilliefors (K-S)", p_value
    except Exception as e:
        # Handle cases where tests might fail (e.g., all values identical)
        import logging
        logging.warning(f"Normality test failed: {e}")
        return "Error", np.nan

def run_wilcoxon_paired(
    data: pd.DataFrame, 
    paired_map: List[Tuple[str, str]],
    is_log_transformed: bool = False
) -> pd.DataFrame:
    if not paired_map:
        raise ValueError("Paired Wilcoxon test requires a non-empty 'paired_map'.")
    
    group1_cols_aligned = [item[0] for item in paired_map]
    group2_cols_aligned = [item[1] for item in paired_map]

    group1_data = data[group1_cols_aligned]
    group2_data = data[group2_cols_aligned]

    # Log2FC
    median_group1 = group1_data.median(axis=1, skipna=True)
    median_group2 = group2_data.median(axis=1, skipna=True)
    
    if is_log_transformed:
        log2fc = median_group2 - median_group1
    else:
        epsilon = 1
        log2fc = np.log2(median_group2 + epsilon) - np.log2(median_group1 + epsilon)

    # Wilcoxon test
    p_values = []
    for i in range(len(data)):
        row_g1 = group1_data.iloc[i].values
        row_g2 = group2_data.iloc[i].values
        
        # Create pairs and remove pairs with NaN
        pairs = [(x, y) for x, y in zip(row_g1, row_g2) if not np.isnan(x) and not np.isnan(y)]
        if len(pairs) < 1:
            p_values.append(np.nan)
            continue
        
        g1_paired = np.array([p[0] for p in pairs])
        g2_paired = np.array([p[1] for p in pairs])

        if np.all(g1_paired == g2_paired):
            p_values.append(1.0)
            continue

        try:
            _, p_val = wilcoxon(g1_paired, g2_paired)
            p_values.append(p_val)
        except ValueError:
            p_values.append(1.0)

    return pd.DataFrame({'p_value': p_values, 'log2FC': log2fc}, index=data.index)