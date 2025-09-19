from flask import Blueprint, render_template, request, session, redirect, url_for, flash, Response, current_app, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import uuid
import re

# Import functions from other modules within the src package
from ..functions.exploratory_data import loadFile, preprocessing_summary_perVariable, preprocessing_general_dataset_statistics # type: ignore
from ..functions.plot_definitions import create_bar_plot 
from .. import data_manager

core_bp = Blueprint('core', __name__, template_folder='../../templates', static_folder='../../static')

@core_bp.route('/')
def index():
    return render_template('index.html')

@core_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        orientation = request.form.get('orientation', 'cols')
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file:
            session.clear()
            
            session['session_id'] = str(uuid.uuid4())

            session_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], session['session_id'])
            os.makedirs(session_folder, exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(session_folder, filename)
            file.save(filepath)
            
            is_log_transformed_input = 'is_log_transformed' in request.form
            is_scaled_input = 'is_scaled' in request.form

            session['step_transformation'] = [1] if is_log_transformed_input else [0]
            session['step_scaling'] = [1] if is_scaled_input else [0]
            session['step_normalization'] = [0]

            initial_message = "Initial data upload."
            if is_log_transformed_input:
                initial_message += " (Pre-transformed)"
            if is_scaled_input:
                initial_message += " (Pre-scaled)"
            session['processing_steps'] = [{'icon': 'fa-upload', 'color': 'text-info', 'message': initial_message}]

            try:
                df = loadFile(filepath)
                
                if df is None or df.empty:
                    flash('Failed to load data or the file is empty.', 'warning')
                    return redirect(request.url)

                if orientation == 'rows':
                    if df.shape[1] > 1:
                        df = df.set_index(df.columns[0]).T
                        df.index.name = None
                    else:
                        df = df.T
                
                df.columns = df.columns.map(str)

                rename_map = {
                    col: os.path.basename(col) 
                    for col in df.columns 
                    if isinstance(col, str) and ('/' in col or '\\' in col)
                }
                if rename_map:
                    df.rename(columns=rename_map, inplace=True)
                
                data_manager.save_dataframe(df, 'df_main_path', 'df_main')
                session['original_shape'] = df.shape
                session['current_shape'] = df.shape

                df_preview_html = df.head(10).to_html(classes='table table-striped table-hover table-sm', table_id='dataframe-preview-table', border=0)
                
                summary_stats = preprocessing_summary_perVariable(df)
                type_counts = summary_stats['type'].value_counts()
                data_types_plot = create_bar_plot(
                    x=type_counts.index.tolist(),
                    y=type_counts.values.tolist(),
                    title='Data Types Distribution',
                    xaxis_title='Data Type',
                    yaxis_title='Count'
                )

                flash('File uploaded successfully! A preview is shown below.', 'success')
                
                return render_template('upload.html', 
                                     df_preview_html=df_preview_html,
                                     shape=df.shape,
                                     data_types_plot=data_types_plot)
                
            except Exception as e:
                flash(f'An error occurred while processing the file: {str(e)}', 'danger')
                return redirect(request.url)
    
    return render_template('upload.html', df_preview_html=None, shape=None,
                           any_log_transformed=any(session.get('step_transformation', [])),
                           any_scaled=any(session.get('step_scaling', [])))

@core_bp.route('/summary')
def summary():
    # Determine which dataframe to use for stats
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        df_sample = data_manager.load_dataframe('df_main_path')
    else:
        df_sample = data_manager.load_dataframe(history_paths[-1])

    if df_sample is None:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('core.upload_file'))

    # Generate general stats and render the main page
    general_stats = preprocessing_general_dataset_statistics(df_sample)
    general_stats_html = general_stats.to_html(classes='table table-striped')
    
    del df_sample

    return render_template('summary.html', 
                         general_stats=general_stats_html)

@core_bp.route('/reset')
def reset():
    if session.get('df_main_path') is None:
        flash('No data to reset','warning')
        return redirect(url_for('core.index'))
    
    data_manager.delete_all_session_dataframes()
    session.clear()
    flash('Page reset successfully','success')
    return redirect(url_for('core.upload_file'))

def _create_group_vector(sample_cols, group_assignments, group_names):
    group_vector = {}
    for col in sample_cols:
        str_col = str(col)
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
    return group_vector

@core_bp.route('/metadata', methods=['GET', 'POST'])
def metadata():
    df = data_manager.load_dataframe('df_main_path')
    if df is None:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('core.upload_file'))
    
    if request.method == 'POST':
        data = request.json
        assignments = data.get('assignments', {})
        group_assignments = data.get('groupAssignments', {})
        group_names = data.get('groupNames', {})
        n_groups = data.get('nGroups', 0)
        
        metadata_cols = [col for col, assign in assignments.items() if assign == 'metadata']
        sample_cols = [col for col, assign in assignments.items() if assign in ['sample', 'undefined']]
        removed_cols = [col for col, assign in assignments.items() if assign == 'removed']
        
        df_metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame()
        df_sample = df[sample_cols].copy() if sample_cols else pd.DataFrame()
        df_original = df.drop(columns=removed_cols).copy() if removed_cols else df.copy()
        
        data_manager.save_dataframe(df_metadata, 'df_metadata_path', 'df_metadata')
        data_manager.save_dataframe(df_sample, 'df_history_0', 'df_history')
        session['df_history_paths'] = ['df_history_0']
        data_manager.save_dataframe(df_original, 'df_main_path', 'df_main')

        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for preprocessing.'}]
        session['imputation_performed'] = False
        session['imputed_mask'] = None
        
        session['group_assignments'] = group_assignments
        session['group_names'] = group_names
        session['n_groups'] = n_groups
        
        group_vector = _create_group_vector(sample_cols, group_assignments, group_names)
        session['group_vector'] = group_vector

        del df
        del df_metadata
        del df_sample
        del df_original
        
        return jsonify({'success': True, 'message': 'Metadata assignments and groups saved successfully'})
    
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

@core_bp.route('/groups')
def view_groups():
    if not session.get('group_vector'):
        flash('No group assignments found. Please define groups in metadata first.', 'warning')
        return redirect(url_for('core.metadata'))
    
    group_vector = session['group_vector']
    group_names = session.get('group_names', {})
    n_groups = session.get('n_groups', 0)
    
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for preprocessing.'}]

    group_summary = {}
    for group_id, group_name in group_names.items():
        if group_id != '0':
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
                         group_regexes=session.get('group_regexes', {}))

@core_bp.route('/update_groups', methods=['POST'])
def update_groups():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'success': False, 'message': 'No sample data available. Please upload a file and assign metadata first.'})

    data = request.json
    group_assignments = data.get('groupAssignments', {})
    group_names = data.get('groupNames', {})
    n_groups = data.get('nGroups', 0)
    group_regexes = data.get('groupRegexes', {})

    session['group_assignments'] = group_assignments
    session['group_names'] = group_names
    session['n_groups'] = n_groups
    session['group_regexes'] = group_regexes

    df_sample_initial = data_manager.load_dataframe(history_paths[0])
    sample_cols = df_sample_initial.columns.tolist()
    group_vector = _create_group_vector(sample_cols, group_assignments, group_names)
    session['group_vector'] = group_vector
    session.modified = True

    return jsonify({'success': True, 'message': 'Group assignments updated successfully.'})

@core_bp.route('/apply_regex_grouping', methods=['POST'])
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

@core_bp.route('/metadata_columns')
def get_metadata_columns():
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    if df_metadata is not None:
        return jsonify({'columns': df_metadata.columns.tolist()})
    return jsonify({'columns': []})