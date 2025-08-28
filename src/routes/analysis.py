from flask import Blueprint, render_template, session, flash, redirect, url_for, request, jsonify
import pandas as pd

# Import plot definitions if needed for pre-rendering, e.g., volcano plot
from ..functions.plot_definitions import create_volcano_plot # type: ignore
from .. import data_manager

analysis_bp = Blueprint('analysis', __name__, template_folder='../../templates')

def _create_group_vector(sample_cols, group_assignments, group_names):
    group_vector = {}
    for col in sample_cols:
        str_col = str(col) # Ensure key is a string
        if str_col in group_assignments and group_assignments[str_col]:
            # Column belongs to multiple groups
            group_info = {
                'groups': group_assignments[str_col],
                'group_names': [group_names.get(str(gid), f'Group {gid}') for gid in group_assignments[str_col]]
            }
        else:
            # Column doesn't belong to any group
            group_info = {
                'groups': [],
                'group_names': []
            }
        group_vector[str_col] = group_info
    return group_vector

@analysis_bp.route('/metadata', methods=['GET', 'POST'])
def metadata():
    df = data_manager.load_dataframe('df_main_path')
    if df is None:
        flash('Please upload a file first')
        return redirect(url_for('core.upload_file'))
    
    if request.method == 'POST':
        data = request.json
        assignments = data.get('assignments', {})
        group_assignments = data.get('groupAssignments', {})
        group_names = data.get('groupNames', {})
        n_groups = data.get('nGroups', 0)
        
        # Process type assignments
        metadata_cols = [col for col, assign in assignments.items() if assign == 'metadata']
        sample_cols = [col for col, assign in assignments.items() if assign in ['sample', 'undefined']]
        removed_cols = [col for col, assign in assignments.items() if assign == 'removed']
        
        # Create dataframes
        df_metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame()
        df_sample = df[sample_cols].copy() if sample_cols else pd.DataFrame()
        df_original = df.drop(columns=removed_cols).copy() if removed_cols else df.copy()
        
        # Store in session
        data_manager.save_dataframe(df_metadata, 'df_metadata_path', 'df_metadata')
        data_manager.save_dataframe(df_sample, 'df_history_0', 'df_history')
        session['df_history_paths'] = ['df_history_0']
        data_manager.save_dataframe(df_original, 'df_main_path', 'df_main')

        # Reset processing state since sample data might have changed
        session['processing_steps'] = []
        session['imputation_performed'] = False
        session['imputed_mask'] = None
        
        # Store group information
        session['group_assignments'] = group_assignments
        session['group_names'] = group_names
        session['n_groups'] = n_groups
        
        # Create group vector for sample columns only
        group_vector = _create_group_vector(sample_cols, group_assignments, group_names)
        session['group_vector'] = group_vector

        # Clean up dataframes from memory
        del df
        del df_metadata
        del df_sample
        del df_original
        
        return jsonify({'success': True, 'message': 'Metadata assignments and groups saved successfully'})
    
    # Get existing assignments
    existing_metadata = []
    existing_sample = []
    
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    if df_metadata is not None and not df_metadata.empty:
        existing_metadata = df_metadata.columns.tolist()
    
    history_paths = session.get('df_history_paths', [])
    if history_paths:
        df_sample_initial = data_manager.load_dataframe(history_paths[0])
        if df_sample_initial is not None and not df_sample_initial.empty:
            existing_sample = df_sample_initial.columns.tolist()
    
    return render_template('metadata.html', 
                         columns=df.columns.tolist(),
                         existing_metadata=existing_metadata,
                         existing_sample=existing_sample,
                         group_regexes=session.get('group_regexes', {}))

@analysis_bp.route('/groups')
def view_groups():
    """View current group assignments"""
    if not session.get('group_vector'):
        flash('No group assignments found. Please define groups in metadata first.')
        return redirect(url_for('analysis.metadata'))
    
    group_vector = session['group_vector']
    group_names = session.get('group_names', {})
    n_groups = session.get('n_groups', 0)
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    # Create summary statistics
    group_summary = {}
    for group_id, group_name in group_names.items():
        if group_id != '0':  # Skip undefined group
            count = 0
            for col, info in group_vector.items():
                if int(group_id) in info.get('groups', []):
                    count += 1
            group_summary[group_name] = count
    
    # Create a DataFrame for display
    group_data = []
    for col, info in group_vector.items():
        if info['groups']:
            group_names_str = ', '.join(info['group_names'])
            group_ids_str = ', '.join(map(str, info['groups']))
        else:
            group_names_str = 'No groups'
            group_ids_str = 'None'
        
        group_data.append({
            'Column': col,
            'Group IDs': group_ids_str,
            'Group Names': group_names_str
        })
    
    # Create summary statistics
    group_summary = {}
    for group_id, group_name in group_names.items():
        if group_id != '0':  # Skip undefined group
            count = 0
            for col, info in group_vector.items():
                if int(group_id) in info.get('groups', []):
                    count += 1
            group_summary[group_name] = count

    return render_template('groups.html',
                         group_vector=group_vector,
                         group_assignments=session.get('group_assignments', {}),
                         group_names=group_names,
                         n_groups=n_groups,
                         group_summary=group_summary,
                         processing_steps=session.get('processing_steps', []),
                         group_regexes=session.get('group_regexes', {}))

@analysis_bp.route('/update_groups', methods=['POST'])
def update_groups():
    """Update group assignments without resetting data processing."""
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'success': False, 'message': 'No sample data available. Please upload a file and assign metadata first.'})

    data = request.json
    group_assignments = data.get('groupAssignments', {})
    group_names = data.get('groupNames', {})
    n_groups = data.get('nGroups', 0)
    group_regexes = data.get('groupRegexes', {})

    # Update group information in session
    session['group_assignments'] = group_assignments
    session['group_names'] = group_names
    session['n_groups'] = n_groups
    session['group_regexes'] = group_regexes

    # Recreate group vector
    df_sample_initial = data_manager.load_dataframe(history_paths[0])
    sample_cols = df_sample_initial.columns.tolist()
    group_vector = _create_group_vector(sample_cols, group_assignments, group_names)
    session['group_vector'] = group_vector
    session.modified = True

    return jsonify({'success': True, 'message': 'Group assignments updated successfully.'})

@analysis_bp.route('/analysis')
def analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))
    
    df = data_manager.load_dataframe(history_paths[-1])
    df_html = df.to_html(classes='table table-striped table-hover', table_id='analysis-table')
    
    return render_template('analysis.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@analysis_bp.route('/multivariate_analysis')
def multivariate_analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))

    processing_steps = session.get('processing_steps', [])

    history_options = []
    for i, step in enumerate(processing_steps):
        history_options.append((i, step['message']))

    return render_template('multivariate_analysis.html',
                           history_options=history_options,
                           selected_history_index=len(history_paths) - 1)

@analysis_bp.route('/comparison')
def comparison():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_original = data_manager.load_dataframe(history_paths[0])
    processing_steps = session.get('processing_steps', [])

    history_options = []
    if len(history_paths) > 1:
        for i, step in enumerate(processing_steps):
            history_options.append((i + 1, step['message']))

    df_processed = data_manager.load_dataframe(history_paths[-1])
    processed_html = df_processed.to_html(classes='table table-striped table-sm', table_id='processed-table')
    original_html = df_original.to_html(classes='table table-striped table-sm', table_id='original-table')

    return render_template('comparison.html',
                         original_html=original_html,
                         processed_html=processed_html,
                         original_shape=df_original.shape,
                         processed_shape=df_processed.shape,
                         history_options=history_options,
                         selected_history_index=len(history_paths) - 1)

@analysis_bp.route('/differential_analysis')
def differential_analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))
    
    group_names = session.get('group_names', {})
    group_vector = session.get('group_vector', {})
    
    # Check for existing results in session
    results_html = None
    differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
    if differential_analysis_results is not None and not differential_analysis_results.empty:
        results_html = differential_analysis_results.to_html(classes='table table-striped table-sm', table_id='resultsTable', escape=False)
    
    if differential_analysis_results is not None:
        del differential_analysis_results

    return render_template('differential_analysis.html',
                           group_names=group_names,
                           group_vector=group_vector,
                           results_html=results_html,
                           latest_differential_analysis_method=session.get('latest_differential_analysis_method'))

@analysis_bp.route('/result_visualization')
def result_visualization():
    differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
    if differential_analysis_results is None or differential_analysis_results.empty:
        flash('Please run a differential analysis first.', 'warning')
        return redirect(url_for('analysis.differential_analysis'))

    results_df = differential_analysis_results
    
    # Determine which p-value column to use
    p_value_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
    
    # Determine significant features
    if 'rejected' in results_df.columns:
        significant_features_df = results_df[results_df['rejected']]
    else:
        significant_features_df = results_df[results_df[p_value_col] < 0.05]
    
    max_features = len(significant_features_df)

    metadata_df = None
    metadata_df = data_manager.load_dataframe('df_meta_thd_path')
    if metadata_df is None or metadata_df.empty:
        metadata_df = data_manager.load_dataframe('df_metadata_path')
    
    volcano_plot_json = create_volcano_plot(
        results_df=results_df,
        metadata_df=metadata_df
    )
    
    # Get metadata columns for clustergram y-axis selection
    metadata_columns = []
    if metadata_df is not None:
        metadata_columns = metadata_df.columns.tolist()

    del differential_analysis_results
    del results_df
    if metadata_df is not None:
        del metadata_df

    return render_template('result_visualization.html', 
                           volcano_plot_json=volcano_plot_json,
                           metadata_columns=metadata_columns,
                           max_features=max_features)