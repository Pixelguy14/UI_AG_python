from flask import Blueprint, jsonify, request, session # type: ignore
import pandas as pd
import numpy as np
import re
from scipy.cluster.hierarchy import linkage, dendrogram # type: ignore

# Import functions from other modules
from ..functions.exploratory_data import preprocessing_summary_perVariable # type: ignore
from ..functions.plot_definitions import ( # type: ignore
    create_pie_chart, create_density_plot, create_boxplot, create_violinplot,
    create_pca_plot, create_hca_plot, create_plsda_plot, create_oplsda_plot
)
from ..functions.statistical_tests import ( # type: ignore
    run_t_test, run_wilcoxon_rank_sum, run_anova, run_kruskal_wallis,
    run_linear_model, run_permanova, apply_multiple_test_correction, format_anova_results_html
)
from .. import data_manager

api_bp = Blueprint('api', __name__, url_prefix='/api')

def _get_column_info(df, column_name):
    if df is None:
        return jsonify({'error': 'No data loaded'})
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    col_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_data)
    total_elements = len(col_data)
    total_nulls = col_data.isnull().sum().sum()
    total_non_nulls = total_elements - total_nulls
    null_plot = create_pie_chart(
        labels=['Null Values', 'Non-Null Values'],
        values=[total_nulls, total_non_nulls],
        title=f'Null Distribution - {column_name}'
    )
    return jsonify({
        'stats': stats.T.to_html(classes='table table-sm'),
        'null_plot': null_plot
    })

def _get_column_density_plot(df, column_name):
    if df is None:
        return jsonify({'error': 'No data loaded'})
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    col_stats_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_stats_data)
    col_plot_data = df[column_name].dropna()
    if col_plot_data.empty:
        return jsonify({'error': 'No data available for this column'})
    density_plot = create_density_plot(col_plot_data, column_name)
    return jsonify({
        'stats': stats.T.to_html(classes='table table-sm'),
        'density_plot': density_plot
    })

@api_bp.route('/column_info/<column_name>')
def column_info(column_name):
    df = data_manager.load_dataframe('df_main_path')
    result = _get_column_info(df, column_name)
    del df
    return result

@api_bp.route('/column_info_analysis/<column_name>')
def column_info_analysis(column_name):
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No data loaded'})
    df = data_manager.load_dataframe(history_paths[-1])
    return _get_column_info(df, column_name)

@api_bp.route('/column_density_plot/<column_name>')
def column_density_plot(column_name):
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No data loaded'})
    df = data_manager.load_dataframe(history_paths[-1])
    return _get_column_density_plot(df, column_name)

@api_bp.route('/column_density_plot_main/<column_name>')
def column_density_plot_main(column_name):
    df = data_manager.load_dataframe('df_main_path')
    return _get_column_density_plot(df, column_name)

@api_bp.route('/distribution_plot/<plot_type>/<context>', methods=['GET'])
def get_distribution_plot(plot_type, context):
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No data available'})
    
    if context == 'before':
        df_sample = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])
    else:
        df_sample = data_manager.load_dataframe(history_paths[-1])
    
    numeric_df = df_sample.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return jsonify({'error': 'No numeric data to plot'})

    try:
        if plot_type == 'boxplot':
            plot = create_boxplot(numeric_df, title='Current Data Distribution', group_vector=session.get('group_vector'), group_names=session.get('group_names'))
        elif plot_type == 'violinplot':
            plot = create_violinplot(numeric_df, title='Current Data Distribution', group_vector=session.get('group_vector'), group_names=session.get('group_names'))
        else:
            return jsonify({'error': 'Invalid plot type'})
        
        return jsonify({'plot': plot})
        
    except Exception as e:
        return jsonify({'error': f'Plot generation failed: {str(e)}'})

@api_bp.route('/metadata_columns')
def get_metadata_columns():
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    if df_metadata is not None:
        return jsonify({'columns': df_metadata.columns.tolist()})
    return jsonify({'columns': []})

@api_bp.route('/pca_plot/<int:history_index>')
def get_pca_plot(history_index):
    history_paths = session.get('df_history_paths', [])
    if not history_paths or history_index >= len(history_paths):
        return jsonify({'error': 'Invalid history index'})
    df = data_manager.load_dataframe(history_paths[history_index])
    plot = create_pca_plot(df, 'PCA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    return jsonify({'plot': plot})

@api_bp.route('/hca_plot', methods=['POST'])
def get_hca_plot():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'Invalid history index'})
    data = request.json
    distance_metric = data.get('distance_metric', 'euclidean')
    linkage_method = data.get('linkage_method', 'average')
    df = data_manager.load_dataframe(history_paths[-1])
    plot = create_hca_plot(df, 'HCA Plot', 
                           group_vector=session.get('group_vector'), 
                           group_names=session.get('group_names'),
                           distance_metric=distance_metric,
                           linkage_method=linkage_method)
    return jsonify({'plot': plot})

@api_bp.route('/plsda_plot/<int:history_index>')
def get_plsda_plot(history_index):
    history_paths = session.get('df_history_paths', [])
    if not history_paths or history_index >= len(history_paths):
        return jsonify({'error': 'Invalid history index'})
    df = data_manager.load_dataframe(history_paths[history_index])
    plot = create_plsda_plot(df, 'PLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    return jsonify({'plot': plot})

@api_bp.route('/oplsda_plot/<int:history_index>')
def get_oplsda_plot(history_index):
    history_paths = session.get('df_history_paths', [])
    if not history_paths or history_index >= len(history_paths):
        return jsonify({'error': 'Invalid history index'})
    df = data_manager.load_dataframe(history_paths[history_index])
    plot = create_oplsda_plot(df, 'OPLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    return jsonify({'plot': plot})

@api_bp.route('/comparison_data/<int:history_index>')
def get_comparison_data(history_index):
    history_paths = session.get('df_history_paths', [])
    if not history_paths or history_index >= len(history_paths):
        return jsonify({'error': 'Invalid history index'})

    df_processed = data_manager.load_dataframe(history_paths[history_index])
    processed_html = df_processed.to_html(classes='table table-striped table-sm', table_id='processed-table')

    return jsonify({
        'processed_html': processed_html,
        'processed_shape': df_processed.shape
    })

@api_bp.route('/run_differential_analysis', methods=['POST'])
def run_differential_analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    data = request.json
    test_type = data.get('test_type')
    groups = data.get('groups')
    correction_method = data.get('correction_method')
    paired_map = data.get('paired_map')
    formula = data.get('formula')

    df = data_manager.load_dataframe(history_paths[-1])
    df.columns = df.columns.map(str) # Ensure df columns are strings
    group_vector = session['group_vector']
    results_df = pd.DataFrame()

    try:
        if test_type in ['ttest', 'paired_ttest']:
            if len(groups) != 2:
                return jsonify({'error': 'T-test requires exactly two groups.'})
            group1_id, group2_id = int(groups[0]), int(groups[1])
            group1_cols = [col for col, info in group_vector.items() if group1_id in info.get('groups', [])]
            group2_cols = [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
            results_df = run_t_test(df, group1_cols, group2_cols, paired=(test_type == 'paired_ttest'), paired_map=paired_map)

        elif test_type == 'wilcoxon':
            if len(groups) != 2:
                return jsonify({'error': 'Wilcoxon test requires exactly two groups.'})
            group1_id, group2_id = int(groups[0]), int(groups[1])
            group1_cols = [col for col, info in group_vector.items() if group1_id in info.get('groups', [])]
            group2_cols = [col for col, info in group_vector.items() if group2_id in info.get('groups', [])]
            results_df = run_wilcoxon_rank_sum(df, group1_cols, group2_cols)

        elif test_type == 'anova':
            if len(groups) < 2:
                return jsonify({'error': 'ANOVA requires at least two groups.'})
            group_map = {session['group_names'][gid]: [col for col, info in group_vector.items() if int(gid) in info.get('groups', [])] for gid in groups}
            
            # Run ANOVA to get stats
            anova_results = run_anova(df, group_map)
            
            results_df = format_anova_results_html(df, group_map, anova_results)

        elif test_type == 'kruskal_wallis':
            if len(groups) < 2:
                return jsonify({'error': 'Kruskal-Wallis requires at least two groups.'})
            group_map = {session['group_names'][gid]: [col for col, info in group_vector.items() if int(gid) in info.get('groups', [])] for gid in groups}
            results_df = run_kruskal_wallis(df, group_map)

        elif test_type == 'linear_model':
            if not formula:
                return jsonify({'error': 'Linear model requires a formula.'})
            # Note: The run_linear_model function expects the metadata dataframe
            metadata_df = data_manager.load_dataframe('df_metadata_path')
            if metadata_df is None:
                 return jsonify({'error': 'Linear models require metadata with covariates.'})
            results_df = run_linear_model(df, formula, metadata_df)

        else:
            return jsonify({'error': 'Invalid test type'})

        # Apply multiple test correction
        if 'p_value' in results_df.columns and correction_method != 'none':
            p_adj, rejected = apply_multiple_test_correction(results_df['p_value'], method=correction_method)
            results_df['p_adj'] = p_adj
            results_df['rejected'] = rejected

        data_manager.save_dataframe(results_df, 'differential_analysis_results_path', 'differential_analysis_results')
        session['latest_differential_analysis_method'] = test_type.replace("_", " ").title()
        return jsonify({'html': results_df.to_html(classes='table table-striped table-hover', table_id='resultsTable', escape=False)})

    except Exception as e:
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'})

@api_bp.route('/run_permanova', methods=['POST'])
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

@api_bp.route('/clustergram_data', methods=['POST'])
def clustergram_data():
    differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
    if differential_analysis_results is None or differential_analysis_results.empty:
        return jsonify({'error': 'No analysis results found'}), 404

    data = request.json
    top_n = int(data.get('top_n', 50))
    distance_metric = data.get('distance_metric', 'euclidean')
    linkage_method = data.get('linkage_method', 'average')
    y_axis_label = data.get('y_axis_label')
    color_palette = data.get('color_palette', 'RdBu')

    results_df = differential_analysis_results
    history_paths = session.get('df_history_paths', [])
    data_df = data_manager.load_dataframe(history_paths[-1])

    def format_value(val):
        if isinstance(val, (int, float)):
            if abs(val) > 10000 or (abs(val) < 0.0001 and val != 0):
                return f'{val:.4e}'
            return round(val, 4)
        return val

    # Get feature-level metadata (rt, mz, intensity)
    feature_metadata_df = None
    feature_metadata_df = data_manager.load_dataframe('df_meta_thd_path')
    if feature_metadata_df is None or feature_metadata_df.empty:
        feature_metadata_df = data_manager.load_dataframe('df_metadata_path')

    # Determine which p-value column to use
    p_value_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
    
    # Determine significant features
    if 'rejected' in results_df.columns:
        significant_features_df = results_df[results_df['rejected']]
    else:
        significant_features_df = results_df[results_df[p_value_col] < 0.05]

    if significant_features_df.empty:
        return jsonify({'error': 'No significant features found to generate a clustergram. Adjust p-value threshold or check analysis results.'}), 404

    significant_features = significant_features_df.nsmallest(top_n, p_value_col).index.unique()
    plot_data = data_df.loc[significant_features]
    numeric_plot_data = plot_data.select_dtypes(include=[np.number])
    
    plot_data_zscored = numeric_plot_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0, axis=1).fillna(0)
    # plot_data_zscored = numeric_plot_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0, axis=0).fillna(0)

    # row_linkage is for features (rows)
    if plot_data_zscored.empty:
        return jsonify({'error': 'No data for clustergram after z-scoring. This might happen if all significant features have zero variance.'}), 404
    row_linkage = linkage(plot_data_zscored.values, method=linkage_method, metric=distance_metric)
    # col_linkage is for samples (columns)
    col_linkage = linkage(plot_data_zscored.T.values, method=linkage_method, metric=distance_metric)
    
    row_dendro = dendrogram(row_linkage, no_plot=True)
    col_dendro = dendrogram(col_linkage, no_plot=True)
    
    # Reorder data using dendrogram leaves
    plot_data_reordered = plot_data_zscored.iloc[row_dendro['leaves'], col_dendro['leaves']]

    # New: Determine y-axis labels based on user selection
    y_labels = plot_data_reordered.index.tolist()
    if y_axis_label and feature_metadata_df is not None and y_axis_label in feature_metadata_df.columns:
        # Ensure metadata is aligned with the reordered data
        aligned_metadata = feature_metadata_df.reindex(plot_data_reordered.reordered.index)
        y_labels = aligned_metadata[y_axis_label].tolist()

    y_labels = [format_value(label) for label in y_labels]

    n_rows = len(plot_data_reordered.index) # Number of features
    n_cols = len(plot_data_reordered.columns) # Number of samples

    # Generate positions for heatmap (5, 15, 25, ...)
    heatmap_x = [5 + 10*i for i in range(n_cols)] # Based on samples
    heatmap_y = [5 + 10*i for i in range(n_rows)] # Based on features

    # Heatmap Trace
    max_abs = np.max(np.abs(plot_data_reordered.values))
    heatmap_trace = {
        'z': plot_data_reordered.values.tolist(),
        'x': heatmap_x,
        'y': heatmap_y,
        'type': 'heatmap',
        'colorscale': color_palette,
        'zmin': -max_abs,
        'zmax': max_abs,
        'colorbar': {'title': 'Z-score'},
        'hoverinfo': 'x+y+z',
    }

    # Prepare customdata for hover (combining sample group and feature metadata)
    full_custom_data = []
    metadata_column_names = ['Group'] # Start with 'Group' for sample metadata

    if feature_metadata_df is not None and not feature_metadata_df.empty:
        # Ensure feature_metadata_df is aligned with the reordered features (rows of heatmap)
        feature_metadata_df = feature_metadata_df.reindex(plot_data_reordered.index)
        metadata_column_names.extend(feature_metadata_df.columns.tolist())

    group_vector = session.get('group_vector', {})
    group_names_map = session.get('group_names', {})

    for row_label in plot_data_reordered.index: # Iterate through features (rows)
        row_custom_data = []
        for col_label in plot_data_reordered.columns: # Iterate through samples (columns)
            cell_custom_data = []

            # Add sample group info
            group_info = group_vector.get(col_label, {})
            groups_assigned = group_info.get('groups', [])
            if groups_assigned:
                display_names = [group_names_map.get(str(gid), f'Group {gid}') for gid in groups_assigned]
                cell_custom_data.append(', '.join(display_names))
            else:
                cell_custom_data.append('None')

            # Add feature metadata (rt, mz, intensity)
            if feature_metadata_df is not None and not feature_metadata_df.empty:
                feature_meta_values = feature_metadata_df.loc[row_label].values.tolist()
                formatted_meta_values = [format_value(v) for v in feature_meta_values]
                cell_custom_data.extend(formatted_meta_values)
            
            row_custom_data.append(cell_custom_data)
        full_custom_data.append(row_custom_data)

    # Column Dendrogram Traces (for samples)
    col_dendro_traces = []
    for i in range(len(col_dendro['icoord'])):
        xs = col_dendro['icoord'][i]
        ys = col_dendro['dcoord'][i]
        col_dendro_traces.append({
            'x': xs,
            'y': ys,
            'mode': 'lines',
            'line': {'color': 'rgb(255,133,27)', 'width': 1},
            'hoverinfo': 'none',
        })

    # Row Dendrogram Traces (for features)
    row_dendro_traces = []
    for i in range(len(row_dendro['icoord'])):
        xs = row_dendro['dcoord'][i]
        ys = row_dendro['icoord'][i]
        row_dendro_traces.append({
            'x': xs,
            'y': ys,
            'mode': 'lines',
            'line': {'color': 'rgb(255,133,27)', 'width': 1},
            'hoverinfo': 'none',
        })

    response_data = {
        'heatmap': heatmap_trace,
        'col_dendro': col_dendro_traces,
        'row_dendro': row_dendro_traces,
        'column_labels': plot_data_reordered.columns.tolist(), # Samples
        'row_labels': y_labels, # Use the potentially updated labels
        'heatmap_x': heatmap_x,
        'heatmap_y': heatmap_y,
        'heatmap_customdata': full_custom_data,
        'metadata_column_names': metadata_column_names
    }

    # Clean up all dataframes
    del differential_analysis_results
    del results_df
    del data_df
    if feature_metadata_df is not None:
        del feature_metadata_df
    del significant_features_df
    del plot_data
    del numeric_plot_data
    del plot_data_zscored
    del plot_data_reordered
    if 'aligned_metadata' in locals():
        del aligned_metadata

    return jsonify(response_data)

@api_bp.route('/apply_regex_grouping', methods=['POST'])
def apply_regex_grouping():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'success': False, 'message': 'No sample data available. Please upload a file and assign metadata first.'})
    df_sample = data_manager.load_dataframe(history_paths[-1])
    if df_sample is None or df_sample.empty:
        return jsonify({'success': False, 'message': 'No sample data available. Please upload a file and assign metadata first.'})

    data = request.json
    group_id = str(data.get('groupId'))
    regex_pattern = data.get('regexPattern')

    if not regex_pattern:
        return jsonify({'success': False, 'message': 'Regex pattern cannot be empty.'})

    try:
        compiled_regex = re.compile(regex_pattern)
    except re.error as e:
        return jsonify({'success': False, 'message': f'Invalid regex pattern: {e}'})

    df_sample_columns = df_sample.columns.tolist()
    group_assignments = session.get('group_assignments', {})
    group_names = session.get('group_names', {})
    matched_columns_count = 0

    for col in df_sample_columns:
        if compiled_regex.match(col):
            current_groups = group_assignments.get(col, [])
            if int(group_id) not in current_groups:
                current_groups.append(int(group_id))
                group_assignments[col] = sorted(current_groups)
                matched_columns_count += 1

    session['group_assignments'] = group_assignments
    session.get('group_regexes', {})[group_id] = regex_pattern

    group_vector = {}
    for col in df_sample_columns:
        group_info = {
            'groups': group_assignments.get(col, []),
            'group_names': [group_names.get(str(gid), f'Group {gid}') for gid in group_assignments.get(col, [])]
        }
        group_vector[col] = group_info
    
    session['group_vector'] = group_vector
    session.modified = True

    return jsonify({
        'success': True, 
        'message': f'Assigned {matched_columns_count} columns to group {group_names.get(group_id, group_id)}.',
        'groupAssignments': group_assignments,
        'groupVector': group_vector
    })