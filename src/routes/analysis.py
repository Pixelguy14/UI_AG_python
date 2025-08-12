from flask import Blueprint, render_template, session, flash, redirect, url_for, request, jsonify
import pandas as pd

# Import plot definitions if needed for pre-rendering, e.g., volcano plot
from ..functions.plot_definitions import create_volcano_plot # type: ignore

analysis_bp = Blueprint('analysis', __name__, template_folder='../../templates')

@analysis_bp.route('/metadata', methods=['GET', 'POST'])
def metadata():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('core.upload_file'))
    
    df = session['df_main']
    
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
        session['df_metadata'] = df_metadata if not df_metadata.empty else None
        session['df_sample'] = df_sample if not df_sample.empty else None
        session['df_history'] = [df_sample] if not df_sample.empty else []
        session['df_main'] = df_original

        # Reset processing state since sample data might have changed
        session['processing_steps'] = []
        session['imputation_performed'] = False
        session['imputed_mask'] = None
        
        # Store group information
        session['group_assignments'] = group_assignments
        session['group_names'] = group_names
        session['n_groups'] = n_groups
        
        # Create group vector for sample columns only
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
        
        session['group_vector'] = group_vector
        
        return jsonify({'success': True, 'message': 'Metadata assignments and groups saved successfully'})
    
    # Get existing assignments
    existing_metadata = []
    existing_sample = []
    
    if session.get('df_metadata') is not None and not session.get('df_metadata').empty:
        existing_metadata = session['df_metadata'].columns.tolist()
    if session.get('df_sample') is not None and not session.get('df_sample').empty:
        existing_sample = session['df_sample'].columns.tolist()
    
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
        return redirect(url_for('core.metadata'))
    
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
    df_sample = session.get('df_sample')
    if df_sample is None or df_sample.empty:
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
    sample_cols = session['df_sample'].columns.tolist()
    group_vector = {}
    for col in sample_cols:
        str_col = str(col) # Ensure key is a string
        if str_col in group_assignments and group_assignments[str_col]:
            group_info = {
                'groups': group_assignments[str_col],
                'group_names': [group_names.get(str(gid), f'Group {gid}') for gid in group_assignments[str_col]]
            }
        else:
            group_info = {
                'groups': [],
                'group_names': []
            }
        group_vector[str_col] = group_info
    
    session['group_vector'] = group_vector
    session.modified = True

    return jsonify({'success': True, 'message': 'Group assignments updated successfully.'})

@analysis_bp.route('/analysis')
def analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))
    
    df = session['df_history'][-1]
    df_html = df.to_html(classes='table table-striped table-hover', table_id='analysis-table')
    
    return render_template('analysis.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@analysis_bp.route('/multivariate_analysis')
def multivariate_analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))

    df_history = session.get('df_history', [])
    processing_steps = session.get('processing_steps', [])

    history_options = []
    for i, step in enumerate(processing_steps):
        history_options.append((i, step['message']))

    return render_template('multivariate_analysis.html',
                           history_options=history_options,
                           selected_history_index=len(df_history) - 1)

@analysis_bp.route('/comparison')
def comparison():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('core.metadata'))
    
    df_original = session['df_sample']
    df_history = session.get('df_history', [])
    processing_steps = session.get('processing_steps', [])

    history_options = []
    if len(df_history) > 1:
        for i, step in enumerate(processing_steps):
            history_options.append((i + 1, step['message']))

    df_processed = df_history[-1]
    processed_html = df_processed.to_html(classes='table table-striped table-sm', table_id='processed-table')
    original_html = df_original.to_html(classes='table table-striped table-sm', table_id='original-table')

    return render_template('comparison.html',
                         original_html=original_html,
                         processed_html=processed_html,
                         original_shape=df_original.shape,
                         processed_shape=df_processed.shape,
                         history_options=history_options,
                         selected_history_index=len(df_history) - 1)

@analysis_bp.route('/differential_analysis')
def differential_analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('processing.imputation'))
    
    group_names = session.get('group_names', {})
    group_vector = session.get('group_vector', {})
    
    # Check for existing results in session
    results_html = None
    if session.get('differential_analysis_results') is not None and not session['differential_analysis_results'].empty:
        results_df = session['differential_analysis_results']
        results_html = results_df.to_html(classes='table table-striped table-sm', table_id='resultsTable', escape=False)
    
    return render_template('differential_analysis.html',
                           group_names=group_names,
                           group_vector=group_vector,
                           results_html=results_html,
                           latest_differential_analysis_method=session.get('latest_differential_analysis_method'))

@analysis_bp.route('/result_visualization')
def result_visualization():
    if 'differential_analysis_results' not in session or session['differential_analysis_results'] is None or session['differential_analysis_results'].empty:
        flash('Please run a differential analysis first.', 'warning')
        return redirect(url_for('analysis.differential_analysis'))

    results_df = session['differential_analysis_results']
    
    # Calculate max features for clustergram
    p_value_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
    if 'rejected' in results_df.columns:
        significant_features_df = results_df[results_df['rejected']]
    else:
        significant_features_df = results_df[results_df[p_value_col] < 0.05]
    max_features = len(significant_features_df)

    metadata_df = None
    if session.get('df_meta_thd') is not None and not session.get('df_meta_thd').empty:
        metadata_df = session.get('df_meta_thd')
    elif session.get('df_metadata') is not None and not session.get('df_metadata').empty:
        metadata_df = session.get('df_metadata')
    
    volcano_plot_json = create_volcano_plot(
        results_df=results_df,
        metadata_df=metadata_df
    )
    
    # Get metadata columns for clustergram y-axis selection
    metadata_columns = []
    if metadata_df is not None:
        metadata_columns = metadata_df.columns.tolist()

    return render_template('result_visualization.html', 
                           volcano_plot_json=volcano_plot_json,
                           metadata_columns=metadata_columns,
                           max_features=max_features)
