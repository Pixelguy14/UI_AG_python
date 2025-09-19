from flask import Blueprint, jsonify, request, session, render_template, flash # type: ignore
import pandas as pd
import numpy as np
import json
import warnings

from ...functions.statistical_tests import (
    run_t_test, run_mann_whitney_u, run_anova, run_anova_rm, run_kruskal_wallis,
    run_friedman_test, run_permanova, apply_multiple_test_correction, format_anova_results_html, check_normality, run_wilcoxon_paired
)
from ... import data_manager

differential_analysis_bp = Blueprint('differential_analysis', __name__, url_prefix='/differential_analysis')

@differential_analysis_bp.route('/run_normality_check', methods=['POST'])
def run_normality_check():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    data = request.json
    groups = data.get('groups')
    if not groups or len(groups) < 1:
        return jsonify({'error': 'Please select at least one group to check for normality.'}), 400

    df = data_manager.load_dataframe(history_paths[-1])
    group_vector = session['group_vector']
    group_names = session.get('group_names', {})

    normality_results = {}
    all_normal = True
    warning_messages = []

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for group_id in groups:
                gid = int(group_id)
                group_cols = [col for col, info in group_vector.items() if gid in info.get('groups', [])]
                if not group_cols:
                    continue

                group_data = df[group_cols].values.flatten()
                group_data = group_data[~np.isnan(group_data)]
                
                group_name = group_names.get(str(gid), f'Group {gid}')

                if len(group_data) > 2:
                    test_name, p_val_normality = check_normality(pd.Series(group_data))
                    if test_name != "Error" and not pd.isna(p_val_normality):
                        normality_results[group_name] = f'{test_name}: p={p_val_normality:.3g}'
                        if p_val_normality < 0.05:
                            all_normal = False
                    else:
                        normality_results[group_name] = 'N/A (calculation error)'
                        all_normal = False
                else:
                    normality_results[group_name] = 'N/A (too few samples)'
                    all_normal = False
            
            for warning_item in w:
                if warning_item.category.__name__ == 'SmallSampleWarning' or issubclass(warning_item.category, RuntimeWarning):
                    warning_messages.append(str(warning_item.message))

        if not normality_results:
            return jsonify({'recommendation': 'Please select groups with data to check normality.', 'results': {}, 'warnings': list(set(warning_messages))})

        if all_normal:
            recommendation = "Data appears normally distributed. Parametric tests are suitable."
        else:
            recommendation = "Data may not be normally distributed. Consider a non-parametric test."

        return jsonify({'recommendation': recommendation, 'results': normality_results, 'warnings': list(set(warning_messages))})

    except Exception as e:
        return jsonify({'error': f'An error occurred during normality check: {str(e)}'}), 500

@differential_analysis_bp.route('/verify_multi_group_pairing', methods=['POST'])
def verify_multi_group_pairing():
    data = request.json
    groups = [int(g) for g in data.get('groups', [])]
    subject_id_col = data.get('subject_id_col')

    if not groups or not subject_id_col:
        return jsonify({'error': 'Missing groups or subject ID column.'}), 400

    df_main = data_manager.load_dataframe(session.get('df_history_paths', [])[-1])
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    group_vector = session.get('group_vector', {})
    group_names = session.get('group_names', {})

    if df_main is None or df_metadata is None:
        return jsonify({'error': 'Dataframes not loaded.'}), 400

    if subject_id_col not in df_metadata.columns:
        return jsonify({'error': f'Subject ID column "{subject_id_col}" not found in metadata.'}), 400

    unique_subjects = df_metadata[subject_id_col].dropna().unique().tolist()
    
    subject_completeness = {}
    all_subjects_complete = True

    for subject_id in unique_subjects:
        subject_completeness[subject_id] = {'present_in_groups': [], 'missing_from_groups': []}
        
        # Get samples associated with this subject_id from metadata
        subject_samples_meta = df_metadata[df_metadata[subject_id_col] == subject_id].index.tolist()
        
        for group_id in groups:
            group_name = group_names.get(str(group_id), f'Group {group_id}')
            
            # Get samples assigned to this group from group_vector
            group_samples_gv = [sample for sample, info in group_vector.items() if group_id in info.get('groups', [])]
            
            # Check for overlap between subject's samples and group's samples
            # This assumes sample names in df_main columns match metadata index
            overlapping_samples = [s for s in subject_samples_meta if s in group_samples_gv and s in df_main.columns]

            if overlapping_samples:
                subject_completeness[subject_id]['present_in_groups'].append(group_name)
            else:
                subject_completeness[subject_id]['missing_from_groups'].append(group_name)
                all_subjects_complete = False
    
    summary = {
        'total_unique_subjects': len(unique_subjects),
        'all_subjects_complete': all_subjects_complete,
        'subject_details': []
    }

    for subject_id, details in subject_completeness.items():
        status = 'Complete' if not details['missing_from_groups'] else 'Incomplete'
        summary['subject_details'].append({
            'subject_id': subject_id,
            'status': status,
            'present_in': ', '.join(details['present_in_groups']) if details['present_in_groups'] else 'None',
            'missing_from': ', '.join(details['missing_from_groups']) if details['missing_from_groups'] else 'None'
        })

    return jsonify({'success': True, 'summary': summary})

@differential_analysis_bp.route('/clear_pairing_data', methods=['POST'])
def clear_pairing_data():
    session['paired_data'] = {}
    session.modified = True
    return jsonify({'success': True, 'message': 'Pairing data cleared.'})

@differential_analysis_bp.route('/save_pairing_data', methods=['POST'])
def save_pairing_data():
    data = request.form.to_dict()
    paired_map_str = data.get('paired_map', '[]')
    groups_str = data.get('groups', '[]')
    
    paired_map = json.loads(paired_map_str)
    groups = json.loads(groups_str)

    session['paired_data'] = {'paired_map': paired_map, 'groups': groups}
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Pairing data saved successfully'})

@differential_analysis_bp.route('/run_differential_analysis', methods=['POST'])
def run_differential_analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    data = request.json
    test_type = data.get('test_type')
    groups = data.get('groups')
    correction_method = data.get('correction_method')
    
    paired_data = session.get('paired_data', {})
    paired_map = paired_data.get('paired_map')
    
    is_paired_test_type = test_type in ['paired_ttest', 'paired_wilcoxon', 'anova_rm', 'friedman']
    if is_paired_test_type:
        paired_for_groups = sorted(paired_data.get('groups', []))
        current_groups_sorted = sorted(groups)
        
        if not paired_data or paired_for_groups != current_groups_sorted:
            return jsonify({'error': 'The selected groups do not match the groups for which paired samples were defined. Please define pairs for the current group selection.'}), 400
        
        if not paired_map:
            return jsonify({'error': 'Paired test requires a non-empty paired map. Please define pairs.'}), 400

    df = data_manager.load_dataframe(history_paths[-1])
    df.columns = df.columns.map(str)
    group_vector = session['group_vector']
    results_df = pd.DataFrame()

    is_log_transformed = 1 in session.get('step_transformation', [])

    reference_group_id = data.get('reference_group_id')
    comparison_group_id = data.get('comparison_group_id')

    normality_results = {}
    normality_recommendation = ""

    warning_messages = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            parametric_tests = ['ttest', 'paired_ttest', 'anova', 'anova_rm']
            if test_type in parametric_tests:
                groups_to_check = []
                if len(groups) == 2:
                    group1_id = int(reference_group_id)
                    group2_id = int(comparison_group_id)
                    group1_cols = [col for col, info in group_vector.items() if group1_id in info.get('groups', [])]
                    group2_cols = [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
                    groups_to_check.append((group1_id, group1_cols))
                    groups_to_check.append((group2_id, group2_cols))
                else:
                    for gid in groups:
                        group_cols = [col for col, info in group_vector.items() if int(gid) in info.get('groups', [])]
                        groups_to_check.append((int(gid), group_cols))
                
                all_normal = True
                for gid, g_cols in groups_to_check:
                    group_data = df[g_cols].values.flatten()
                    group_data = group_data[~np.isnan(group_data)]
                    group_name = session['group_names'][str(gid)]
                    if len(group_data) > 2:
                        test_name, p_val_normality = check_normality(pd.Series(group_data))
                        if test_name != "Error" and not pd.isna(p_val_normality):
                            normality_results[group_name] = f'{test_name}: p={p_val_normality:.3f}'
                            if p_val_normality < 0.05:
                                all_normal = False
                        else:
                            normality_results[group_name] = 'N/A (calculation error)'
                            all_normal = False
                    else:
                        normality_results[group_name] = 'N/A (too few samples)'
                        all_normal = False
                
                if all_normal:
                    normality_recommendation = "Data appears normally distributed. Parametric test is suitable."
                else:
                    normality_recommendation = "Data may not be normally distributed. Consider a non-parametric test."

            if test_type == 'ttest' or test_type == 'paired_ttest':
                if len(groups) != 2:
                    return jsonify({'error': f'{test_type.replace("_", " ").title()} requires exactly two groups.'})
                if not reference_group_id or not comparison_group_id:
                    return jsonify({'error': 'Please select both reference and comparison groups.'})

                group1_id = int(reference_group_id)
                group2_id = int(comparison_group_id)
                group1_cols = [col for col, info in group_vector.items() if group1_id in info.get('groups', [])]
                group2_cols = [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
                
                is_paired = (test_type == 'paired_ttest')
                results_df = run_t_test(df, group1_cols, group2_cols, paired=is_paired, paired_map=paired_map, is_log_transformed=is_log_transformed)

            elif test_type == 'mann_whitney':
                if len(groups) != 2:
                    return jsonify({'error': f'{test_type.replace("_", " ").title()} requires exactly two groups.'})
                if not reference_group_id or not comparison_group_id:
                    return jsonify({'error': 'Please select both reference and comparison groups.'})

                group1_id = int(reference_group_id)
                group2_id = int(comparison_group_id)
                group1_cols = [col for col, info in group_vector.items() if group1_id in info.get('groups', [])]
                group2_cols = [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
                results_df = run_mann_whitney_u(df, group1_cols, group2_cols, is_log_transformed=is_log_transformed)

            elif test_type == 'paired_wilcoxon':
                if len(groups) != 2:
                    return jsonify({'error': 'Paired Wilcoxon test requires exactly two groups.'}), 400
                results_df = run_wilcoxon_paired(df, paired_map, is_log_transformed=is_log_transformed)

            elif test_type == 'anova_rm':
                results_df = run_anova_rm(df, paired_map, is_log_transformed=is_log_transformed)

            elif test_type == 'anova':
                if len(groups) < 2:
                    return jsonify({'error': 'ANOVA requires at least two groups.'})
                
                posthoc_method = data.get('posthoc_method', 'tukeyhsd')

                if len(groups) == 2:
                    if not reference_group_id or not comparison_group_id:
                        return jsonify({'error': 'Please select both reference and comparison groups for two-group ANOVA.'})
                    group1_id = int(reference_group_id)
                    group2_id = int(comparison_group_id)
                    group_map = {
                        session['group_names'][str(group1_id)]: [col for col, info in group_vector.items() if group1_id in info.get('groups', [])],
                        session['group_names'][str(group2_id)]: [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
                    }
                else:
                    group_map = {session['group_names'][gid]: [col for col, info in group_vector.items() if int(gid) in info.get('groups', [])] for gid in groups}
                
                anova_results = run_anova(df, group_map)
                results_df = format_anova_results_html(df, group_map, anova_results, posthoc_method=posthoc_method)

            elif test_type == 'kruskal_wallis':
                if len(groups) < 2:
                    return jsonify({'error': 'Kruskal-Wallis requires at least two groups.'})
                
                if len(groups) == 2:
                    if not reference_group_id or not comparison_group_id:
                        return jsonify({'error': 'Please select both reference and comparison groups for two-group Kruskal-Wallis.'})
                    group1_id = int(reference_group_id)
                    group2_id = int(comparison_group_id)
                    group_map = {
                        session['group_names'][str(group1_id)]: [col for col, info in group_vector.items() if group1_id in info.get('groups', [])],
                        session['group_names'][str(group2_id)]: [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
                    }
                else:
                    group_map = {session['group_names'][gid]: [col for col, info in group_vector.items() if int(gid) in info.get('groups', [])] for gid in groups}
                
                results_df = run_kruskal_wallis(df, group_map, is_log_transformed=is_log_transformed)

            elif test_type == 'friedman':
                if len(groups) < 2:
                    return jsonify({'error': 'Friedman test requires at least two groups.'})
                results_df = run_friedman_test(df, paired_map, is_log_transformed=is_log_transformed)

            else:
                return jsonify({'error': 'Invalid test type'})

            if 'p_value' in results_df.columns and correction_method != 'none':
                p_adj, rejected = apply_multiple_test_correction(results_df['p_value'], method=correction_method)
                results_df['p_adj'] = p_adj
                results_df['rejected'] = rejected

        for warning_item in w:
            if warning_item.category.__name__ == 'SmallSampleWarning' or issubclass(warning_item.category, RuntimeWarning):
                warning_messages.append(str(warning_item.message))

        data_manager.save_dataframe(results_df, 'differential_analysis_results_path', 'differential_analysis_results')
        significant_count = 0
        if 'rejected' in results_df.columns:
            significant_count = int(results_df['rejected'].sum())
        elif 'p_adj' in results_df.columns:
            significant_count = int((results_df['p_adj'] < 0.05).sum())
        elif 'p_value' in results_df.columns:
            significant_count = int((results_df['p_value'] < 0.05).sum())

        session['latest_differential_analysis_method'] = test_type.replace("_", " ").title()
        return jsonify({
            'html': results_df.to_html(classes='table table-striped table-hover', table_id='resultsTable', escape=False), 
            'normality_results': normality_results, 
            'normality_recommendation': normality_recommendation,
            'significant_count': significant_count,
            'warnings': list(set(warning_messages))
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'})

@differential_analysis_bp.route('/run_permanova', methods=['POST'])
def run_permanova_route():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    data = request.json
    distance_metric = data.get('distance_metric', 'euclidean')
    permutations = int(data.get('permutations', 999))

    df = data_manager.load_dataframe(history_paths[-1])
    group_vector = session['group_vector']

    try:
        result = run_permanova(df, group_vector, distance_metric, permutations)
        result_summary = {
            'test_statistic': result['test statistic'],
            'p_value': result['p-value']
        }
        session['permanova_results'] = result_summary
        del df
        return jsonify({'success': True, 'result': result_summary})
    except Exception as e:
        del df
        return jsonify({'error': f'PERMANOVA failed: {str(e)}'})

@differential_analysis_bp.route('/get_pairing_table', methods=['GET'])
def get_pairing_table():
    requested_groups = request.args.getlist('groups[]')
    
    group_vector = session.get('group_vector', {})
    group_names = session.get('group_names', {})
    paired_data = session.get('paired_data', {})
    
    saved_groups = paired_data.get('groups', [])

    requested_groups_str = sorted([str(g) for g in requested_groups])
    saved_groups_str = sorted([str(g) for g in saved_groups])

    if saved_groups_str and requested_groups_str != saved_groups_str:
        flash('The selected groups have changed. Previously saved pairing data has been cleared.', 'warning')
        session['paired_data'] = {}
        session.modified = True
        existing_paired_map = []
    else:
        existing_paired_map = paired_data.get('paired_map', [])

    base_group_samples = []
    if requested_groups:
        base_group_id = int(requested_groups[0])
        base_group_samples = [s for s, info in group_vector.items() if base_group_id in info.get('groups', [])]

    return render_template('_paired_samples_table.html',
                           selected_groups=requested_groups,
                           group_names=group_names,
                           group_vector=group_vector,
                           existing_paired_map=existing_paired_map,
                           base_group_samples=base_group_samples)
