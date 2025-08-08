from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import io
import os
import numpy as np
from flask_session import Session  # Import Flask-Session
from werkzeug.utils import secure_filename
from scipy.cluster.hierarchy import linkage, dendrogram # type: ignore

from src.functions.exploratory_data import (
    loadFile,
    preprocessing_general_dataset_statistics,
    preprocessing_summary_perVariable,
)
from src.functions.imputation_methods import (
    halfMinimumImputed,
    knnImputed,
    meanImputed,
    medianImputed,
    miceBayesianRidgeImputed,
    miceLinearRegressionImputed,
    missForestImputed,
    nImputed,
    postprocess_imputation,
    svdImputed,
)
from src.functions.normalization_methods import (
    tic_normalization,
    mtic_normalization,
    median_normalization,
    quantile_normalization,
    pqn_normalization,
)
from src.functions.log_transfomation_methods import (
    log2_transform,
    log10_transform,
    sqrt_transform,
    cube_root_transform,
    arcsinh_transform,
    glog_transform,
    yeo_johnson_transform,
)
from src.functions.scaling_methods import (
    standard_scaling,
    minmax_scaling,
    pareto_scaling,
    range_scaling,
    robust_scaling,
    vast_scaling,
)
from src.functions.plot_definitions import (
    create_bar_plot,
    create_boxplot,
    create_density_plot,
    create_distribution_plot,
    create_heatmap,
    create_heatmap_BW,
    create_hca_plot,
    create_pca_plot,
    create_pie_chart,
    create_violinplot,
    create_plsda_plot,
    create_oplsda_plot,
    create_volcano_plot
)

from src.functions.statistical_tests import (
    run_t_test, 
    run_wilcoxon_rank_sum, 
    run_anova, 
    run_kruskal_wallis,
    run_linear_model, 
    run_permanova, 
    apply_multiple_test_correction,
    format_anova_results_html
)

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Silence the verbose DEBUG messages from matplotlib's font manager
logging.getLogger('matplotlib').setLevel(logging.INFO)

app = Flask(__name__)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session_cache'  # Directory to store session files
app.config['UPLOAD_FOLDER'] = 'src/uploads'
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024  # 800MB max file size

# Initialize Flask-Session
Session(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True) # Ensure session cache directory exists

@app.before_request
def before_request():
    # Ensure session variables are initialized
    if 'df_main' not in session:
        session['df_main'] = None
    if 'df_metadata' not in session:
        session['df_metadata'] = None
    if 'df_meta_thd' not in session:
        session['df_meta_thd'] = None
    if 'df_sample' not in session:
        session['df_sample'] = None
    if 'df_history' not in session:
        session['df_history'] = []
    if 'imputed_mask' not in session:
        session['imputed_mask'] = None
    if 'df_original' not in session:
        session['df_original'] = None
    if 'current_column' not in session:
        session['current_column'] = ''
    if 'orientation' not in session:
        session['orientation'] = 'cols'
    if 'imputation_performed' not in session:
        session['imputation_performed'] = False
    if 'group_assignments' not in session:
        session['group_assignments'] = {}
    if 'group_names' not in session:
        session['group_names'] = {}
    if 'n_groups' not in session:
        session['n_groups'] = 0
    if 'group_vector' not in session:
        session['group_vector'] = {}
    if 'processing_steps' not in session:
        session['processing_steps'] = []
    if 'differential_analysis_results' not in session:
        session['differential_analysis_results'] = None
    if 'latest_differential_analysis_method' not in session:
        session['latest_differential_analysis_method'] = None
    if 'group_regexes' not in session:
        session['group_regexes'] = {}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
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
            # Reset session state for a new file upload
            session['df_main'] = None
            session['df_metadata'] = None
            session['df_meta_thd'] = None
            session['df_sample'] = None
            session['df_history'] = []
            session['imputed_mask'] = None
            session['df_original'] = None
            session['current_column'] = ''
            session['orientation'] = 'cols'
            session['imputation_performed'] = False
            session['group_assignments'] = {}
            session['group_names'] = {}
            session['n_groups'] = 0
            session['group_vector'] = {}
            if 'processing_steps' in session:
                session['processing_steps'] = []
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Load the file
                df = loadFile(filepath)
                
                if df is None or df.empty:
                    flash('Failed to load data or the file is empty.', 'warning')
                    return redirect(request.url)
                # Handle orientation
                if orientation == 'rows':
                    # This option indicates the file has samples in rows and features in columns.
                    # The data needs to be transposed to match the app's internal format
                    # (samples in columns). A simple `df.T` can fail if the first column
                    # (containing sample names) is not treated as the index.
                    # The approach for transposition is to set the first column
                    # as the index and then transpose.
                    if df.shape[1] > 1:
                        df = df.set_index(df.columns[0]).T
                        df.index.name = None
                    else:
                        # If only one column, a simple transpose is sufficient.
                        df = df.T
                print(df.head())
                
                # Ensure column names are strings to avoid type mismatches later
                df.columns = df.columns.map(str)

                # Clean column names (remove paths)
                rename_map = {
                    col: os.path.basename(col) 
                    for col in df.columns 
                    if isinstance(col, str) and ('/' in col or '\\' in col)
                }
                if rename_map:
                    df.rename(columns=rename_map, inplace=True)
                # Store full DataFrame in session
                session['df_main'] = df
                session['df_original'] = df
                session['orientation'] = orientation
                
                # Create a preview for the upload page
                df_preview_html = df.head(10).to_html(classes='table table-striped table-hover table-sm', table_id='dataframe-preview-table', border=0)
                
                # Generate data types distribution plot for upload page
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
                
                # Re-render the upload page, now with the preview
                return render_template('upload.html', 
                                     df_preview_html=df_preview_html,
                                     shape=df.shape,
                                     data_types_plot=data_types_plot)
                
            except Exception as e:
                flash(f'An error occurred while processing the file: {str(e)}', 'danger')
                return redirect(request.url)
    
    # For GET requests, render the page without any preview
    return render_template('upload.html', df_preview_html=None, shape=None)

@app.route('/summary')
def summary():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('upload_file'))
    
    df = session['df_main']
    
    plots = {}
    # Correlation matrix (if we have sample data)
    if session.get('df_history') and not session['df_history'][-1].empty:
        df_sample = session['df_history'][-1]
        general_stats = preprocessing_general_dataset_statistics(df_sample)
        numeric_df = df_sample.select_dtypes(include=[np.number])
        
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            # Fill NaN values (e.g., from constant columns) with 0 for plotting
            corr_matrix = corr_matrix.fillna(0)
            plots['correlation'] = create_heatmap(
                corr_matrix,
                title='Correlation Matrix'
            )
        
        # Missing values heatmap
        plots['missing_heatmap'] = create_heatmap_BW(
            df_sample,
            title='Missing Values Distribution',
            imputed=session['imputation_performed'],
            null_mask=session.get('imputed_mask')
        )
        
        # Mean intensity bar chart
        if not numeric_df.empty:
            mean_values = numeric_df.mean()
            plots['mean_intensity'] = create_bar_plot(
                x=mean_values.index.tolist(),
                y=mean_values.values.tolist(),
                title=f'Mean Intensity ({len(mean_values)} samples)',
                xaxis_title='Samples',
                yaxis_title='Mean log2 intensity',
                group_vector=session.get('group_vector'),
                group_names=session.get('group_names')
            )
            mean_values= mean_values.dropna()
            if not mean_values.empty:
                plots['mean_intensity_distribution'] = create_distribution_plot(
                    mean_values,
                    'Mean Intensity Distribution'
                )

        # Boxplot with all points
        numeric_df_for_boxplot = df_sample.select_dtypes(include=[np.number])
        if not numeric_df_for_boxplot.empty:
            plots['boxplot_distribution'] = create_boxplot(
                numeric_df_for_boxplot,
                title='Distribution of Sample Data',
                group_vector=session.get('group_vector'),
                group_names=session.get('group_names')
            )
    else:
        # Missing values heatmap
        plots['missing_heatmap'] = create_heatmap_BW(
            df.isnull().astype(int),
            title='Missing Values Distribution'
        )
        general_stats = preprocessing_general_dataset_statistics(df)

    return render_template('summary.html', 
                         general_stats=general_stats.to_html(classes='table table-striped'),
                         plots=plots)

@app.route('/dataframe')
def dataframe_view():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('upload_file'))
    
    df = session['df_main']
    
    # Convert DataFrame to HTML with pagination
    #df_html = df.head(100).to_html(classes='table table-striped table-hover', table_id='dataframe-table')
    df_html = df.to_html(classes='table table-striped table-hover', table_id='dataframe-table')
    
    return render_template('dataframe.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@app.route('/column_info/<column_name>')
def column_info(column_name):
    if session.get('df_main') is None:
        return jsonify({'error': 'No data loaded'})
    
    df = session['df_main']
    
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    
    col_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_data)
    
    # Create null distribution plot
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
    
@app.route('/column_info_analysis/<column_name>')
def column_info_analysis(column_name):
    if not session.get('df_history'):
        return jsonify({'error': 'No data loaded'})
    
    df = session['df_history'][-1]
    
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    
    col_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_data)
    
    # Create null distribution plot
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

@app.route('/column_density_plot/<column_name>')
def column_density_plot(column_name):
    if not session.get('df_history'):
        return jsonify({'error': 'No data loaded'})

    df = session['df_history'][-1]

    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})

    # Calculate stats
    col_stats_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_stats_data)
    
    # Create plot
    col_plot_data = df[column_name].dropna()

    if col_plot_data.empty:
        return jsonify({'error': 'No data available for this column'})

    density_plot = create_density_plot(col_plot_data, column_name)

    return jsonify({
        'stats': stats.T.to_html(classes='table table-sm'),
        'density_plot': density_plot
    })

@app.route('/column_density_plot_main/<column_name>')
def column_density_plot_main(column_name):
    if session.get('df_main') is None:
        return jsonify({'error': 'No data loaded'})

    df = session['df_main']

    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})

    # Calculate stats
    col_stats_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_stats_data)
    
    # Create plot
    col_plot_data = df[column_name].dropna()

    if col_plot_data.empty:
        return jsonify({'error': 'No data available for this column'})

    density_plot = create_density_plot(col_plot_data, column_name)

    return jsonify({
        'stats': stats.T.to_html(classes='table table-sm'),
        'density_plot': density_plot
    })

@app.route('/metadata', methods=['GET', 'POST'])
def metadata():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('upload_file'))
    
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

@app.route('/groups')
def view_groups():
    """View current group assignments"""
    if not session.get('group_vector'):
        flash('No group assignments found. Please define groups in metadata first.')
        return redirect(url_for('metadata'))
    
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

@app.route('/update_groups', methods=['POST'])
def update_groups():
    """Update group assignments without resetting data processing."""
    df_sample = session.get('df_sample')
    if df_sample is None or df_sample.empty:
        logging.error("apply_regex_grouping: No sample data found in session.")
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

@app.route('/apply_regex_grouping', methods=['POST'])
def apply_regex_grouping():
    df_sample = session.get('df_sample')
    if df_sample is None or df_sample.empty:
        # logging.error("apply_regex_grouping: No sample data found in session.")
        return jsonify({'success': False, 'message': 'No sample data available. Please upload a file and assign metadata first.'})

    data = request.json
    group_id = str(data.get('groupId'))
    regex_pattern = data.get('regexPattern')
    # logging.debug(f"apply_regex_grouping: Received groupId={group_id}, regexPattern='{regex_pattern}'")

    if not regex_pattern:
        # logging.warning("apply_regex_grouping: Regex pattern is empty.")
        return jsonify({'success': False, 'message': 'Regex pattern cannot be empty.'})

    try:
        import re
        compiled_regex = re.compile(regex_pattern)
    except re.error as e:
        # logging.error(f"apply_regex_grouping: Invalid regex pattern '{regex_pattern}': {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Invalid regex pattern: {e}'})
    except Exception as e:
        # logging.error(f"apply_regex_grouping: Unexpected error during regex compilation: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'An unexpected error occurred during regex compilation: {e}'})

    df_sample_columns = session['df_sample'].columns.tolist()
    group_assignments = session.get('group_assignments', {})
    group_names = session.get('group_names', {})

    matched_columns_count = 0
    try:
        for col in df_sample_columns:
            if compiled_regex.match(col):
                # Ensure the group_id is added as an integer
                current_groups = group_assignments.get(col, [])
                if int(group_id) not in current_groups:
                    current_groups.append(int(group_id))
                    group_assignments[col] = sorted(current_groups) # Keep sorted for consistency
                    matched_columns_count += 1

        session['group_assignments'] = group_assignments
        session['group_regexes'][group_id] = regex_pattern # Store the regex

        # Recreate group vector based on updated group_assignments
        group_vector = {}
        for col in df_sample_columns:
            if col in group_assignments and group_assignments[col]:
                group_info = {
                    'groups': group_assignments[col],
                    'group_names': [group_names.get(str(gid), f'Group {gid}') for gid in group_assignments[col]]
                }
            else:
                group_info = {
                    'groups': [],
                    'group_names': []
                }
            group_vector[col] = group_info
        
        session['group_vector'] = group_vector
        session.modified = True

        # logging.info(f"apply_regex_grouping: Successfully assigned {matched_columns_count} columns to group {group_names.get(group_id, group_id)}.")
        return jsonify({'success': True, 'message': f'Successfully assigned {matched_columns_count} columns to group {group_names.get(group_id, group_id)}.', 'groupAssignments': group_assignments, 'groupVector': group_vector})

    except Exception as e:
        # logging.error(f"apply_regex_grouping: Error during column assignment or session update: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'An error occurred during column assignment: {e}'})

@app.route('/imputation')
def imputation():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_sample = session['df_history'][-1]
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = {}
    # Missing values heatmap for imputation tab
    plots['missing_heatmap'] = create_heatmap_BW(
        df_sample,
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session['imputation_performed'],
        null_mask=session.get('imputed_mask')
    )

    return render_template('imputation.html',
                         original_shape=session['df_sample'].shape,
                         current_shape=df_sample.shape,
                         processing_steps=session['processing_steps'],
                         plots=plots)

@app.route('/threshold', methods=['POST'])
def threshold():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})
    
    threshold_percent = float(request.json.get('threshold', 80))
    
    df_sample = session['df_history'][-1]
    df_metadata = session.get('df_metadata')
    
    # Apply thresholding
    num_columns = len(df_sample.columns)
    threshold_count = max(1, int((threshold_percent / 100.0) * num_columns)) if num_columns > 0 else 0
    
    df_thresholded = df_sample.dropna(thresh=threshold_count)
    
    # Cut metadata if it exists
    if df_metadata is not None and not df_metadata.empty:
        df_meta_thd = df_metadata.loc[df_thresholded.index]
        session['df_meta_thd'] = df_meta_thd
    else:
        session['df_meta_thd'] = None
    
    # Store result
    session['df_history'].append(df_thresholded)
    
    # Update processing steps
    session['processing_steps'].append({'icon': 'fa-filter', 'color': 'text-info', 'message': f'Applied thresholding: {threshold_percent}% non-null values. New shape: {df_thresholded.shape[0]} rows, {df_thresholded.shape[1]} columns.'})
    session.modified = True # Mark session as modified

    # Generate and return the updated heatmap data
    updated_heatmap = create_heatmap_BW(
        df_thresholded,
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session['imputation_performed'],
        null_mask=session.get('imputed_mask')
    )

    return jsonify({
        'success': True,
        'original_shape': session['df_sample'].shape,
        'new_shape': df_thresholded.shape,
        'message': f'Thresholding applied with {threshold_percent}%',
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap
    })

@app.route('/apply_imputation', methods=['POST'])
def apply_imputation():
    if not session.get('df_history'):
        return jsonify({'error': 'No thresholded sample data available'})
    
    method = request.json.get('method')
    params = request.json.get('params', {})
    
    df_before_imputation = session['df_history'][-1]

    try:
        # Apply scaling for advanced methods
        df_scaled = (df_before_imputation - df_before_imputation.mean()) / df_before_imputation.std()
        
        if method == 'n_imputation':
            n_val = params.get('n_value', 0)
            imputed_df = nImputed(df_before_imputation, n=n_val)
        elif method == 'half_minimum':
            imputed_df = halfMinimumImputed(df_before_imputation)
        elif method == 'mean':
            imputed_df = meanImputed(df_before_imputation)
        elif method == 'median':
            imputed_df = medianImputed(df_before_imputation)
        elif method == 'miss_forest':
            max_iter = params.get('max_iter', 10)
            n_estimators = params.get('n_estimators', 100)
            imputed_df = missForestImputed(df_scaled, max_iter=max_iter, n_estimators=n_estimators)
            imputed_df = postprocess_imputation(imputed_df, df_before_imputation)
        elif method == 'svd':
            n_components = params.get('n_components', 5)
            imputed_df = svdImputed(df_scaled, n_components=n_components)
            imputed_df = postprocess_imputation(imputed_df, df_before_imputation)
        elif method == 'knn':
            n_neighbors = params.get('n_neighbors', 2)
            imputed_df = knnImputed(df_scaled, n_neighbors=n_neighbors)
            imputed_df = postprocess_imputation(imputed_df, df_before_imputation)
        elif method == 'mice_bayesian':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceBayesianRidgeImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df_before_imputation)
        elif method == 'mice_linear':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceLinearRegressionImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df_before_imputation)
        else:
            return jsonify({'error': 'Unknown imputation method'})
        
        # Store results
        session['df_history'].append(imputed_df)
        
        # Calculate imputed mask: where was it null before and now it's not null
        imputed_mask = df_before_imputation.isnull() & ~imputed_df.isnull()
        session['imputed_mask'] = imputed_mask

        # Update processing steps
        session['processing_steps'].append({'icon': 'fa-magic', 'color': 'text-primary', 'message': f'Applied {method} imputation.'})
        session.modified = True # Mark session as modified
        session['imputation_performed'] = True # Set flag that imputation has occurred
        
        # Generate and return the updated heatmap data
        updated_heatmap = create_heatmap_BW(
            imputed_df,
            title='Missing Values Distribution (Imputed Highlighted)',
            imputed=session['imputation_performed'],
            null_mask=session.get('imputed_mask')
        )

        return jsonify({
            'success': True,
            'message': f'Imputation with {method} completed successfully',
            'steps': session['processing_steps'],
            'new_shape': imputed_df.shape,
            'missing_heatmap': updated_heatmap
        })
        
    except Exception as e:
        # logging.error(f"Imputation failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Imputation failed: {str(e)}'})

@app.route('/replace_zeros', methods=['POST'])
def replace_zeros():
    if not session.get('df_history'):
        return jsonify({'error': 'No data available to process.'})

    df_current = session['df_history'][-1]
    df_cleaned = df_current.replace(0, np.nan)

    session['df_history'].append(df_cleaned)
    session['processing_steps'].append({
        'icon': 'fa-broom',
        'color': 'text-warning',
        'message': 'Replaced all zero values with NaN.'
    })
    session.modified = True

    updated_heatmap = create_heatmap_BW(
        df_cleaned,
        title='Missing Values Distribution',
        imputed=session.get('imputation_performed', False),
        null_mask=session.get('imputed_mask')
    )

    return jsonify({
        'success': True,
        'message': 'All zero values have been replaced with NaN.',
        'new_shape': df_cleaned.shape,
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap
    })

@app.route('/normalization')
def normalization():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][0] # Assuming the first entry is the original sample data

    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_normalization_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

    return render_template('normalization.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@app.route('/apply_normalization', methods=['POST'])
def apply_normalization():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    df = session['df_history'][-1]

    try:
        if method == 'tic':
            normalized_df = tic_normalization(df)
        elif method == 'mtic':
            normalized_df = mtic_normalization(df)
        elif method == 'median':
            normalized_df = median_normalization(df)
        elif method == 'quantile':
            normalized_df = quantile_normalization(df)
        elif method == 'pqn':
            normalized_df = pqn_normalization(df)
        else:
            return jsonify({'error': 'Unknown normalization method'})

        session['df_history'].append(normalized_df)
        session['processing_steps'].append({
            'icon': 'fa-chart-bar',
            'color': 'text-success',
            'message': f'Applied {method.upper()} normalization.'
        })
        session.modified = True

        # Get the original data before normalization for comparison plots
        df_before_normalization = session['df_history'][0]
        
        # Generate plots
        plots = get_normalization_plots(df_before_normalization, normalized_df, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.upper()} normalization applied successfully.',
            'new_shape': normalized_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Normalization failed: {str(e)}'})

@app.route('/transformation')
def transformation():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][0]

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_transformation_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

    return render_template('transformation.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@app.route('/apply_transformation', methods=['POST'])
def apply_transformation():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = session['df_history'][-1]

    try:
        transformed_df = None
        if method == 'log2':
            transformed_df = log2_transform(df, pseudo_count=params.get('pseudo_count'))
        elif method == 'log10':
            transformed_df = log10_transform(df, pseudo_count=params.get('pseudo_count'))
        elif method == 'sqrt':
            transformed_df = sqrt_transform(df)
        elif method == 'cube_root':
            transformed_df = cube_root_transform(df)
        elif method == 'arcsinh':
            transformed_df = arcsinh_transform(df, cofactor=params.get('cofactor', 5))
        elif method == 'glog':
            transformed_df = glog_transform(df, lamb=params.get('lamb'))
        elif method == 'yeo_johnson':
            transformed_df = yeo_johnson_transform(df)
        else:
            return jsonify({'error': 'Unknown transformation method'})

        session['df_history'].append(transformed_df)
        session['processing_steps'].append({
            'icon': 'fa-exchange-alt',
            'color': 'text-info',
            'message': f'Applied {method.replace("_", " ").title()} transformation.'
        })
        session.modified = True

        df_before_transformation = session['df_history'][0]
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_transformation_plots(df_before_transformation, transformed_df, plot_type, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} transformation applied successfully.',
            'new_shape': transformed_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        # logging.error(f"Transformation failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Transformation failed: {str(e)}'})

@app.route('/scaling')
def scaling():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][0]

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_scaling_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

    return render_template('scaling.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@app.route('/apply_scaling', methods=['POST'])
def apply_scaling():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = session['df_history'][-1]

    try:
        scaled_df = None
        if method == 'standard':
            scaled_df = standard_scaling(df, with_mean=params.get('with_mean', True), with_std=params.get('with_std', True))
        elif method == 'minmax':
            scaled_df = minmax_scaling(df, feature_range=tuple(params.get('feature_range', [0, 1])))
        elif method == 'pareto':
            scaled_df = pareto_scaling(df)
        elif method == 'range':
            scaled_df = range_scaling(df)
        elif method == 'robust':
            scaled_df = robust_scaling(df)
        elif method == 'vast':
            scaled_df = vast_scaling(df)
        else:
            return jsonify({'error': 'Unknown scaling method'})

        session['df_history'].append(scaled_df)
        session['processing_steps'].append({
            'icon': 'fa-compress-arrows-alt',
            'color': 'text-warning',
            'message': f'Applied {method.replace("_", " ").title()} scaling.'
        })

        df_before_scaling = session['df_history'][0]
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_scaling_plots(df_before_scaling, scaled_df, plot_type, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} scaling applied successfully.',
            'new_shape': scaled_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        # logging.error(f"Scaling failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Scaling failed: {str(e)}'})

@app.route('/get_distribution_plot/<plot_type>/<context>', methods=['GET'])
def get_distribution_plot(plot_type, context):
    if not session.get('df_history'):
        return jsonify({'error': 'No data available'})
    
    if context == 'before':
        df_sample = session['df_history'][0]
    else:
        df_sample = session['df_history'][-1]
    
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
        
        # Return direct Plotly-compatible structure
        return jsonify({'plot': plot})
        
    except Exception as e:
        return jsonify({'error': f'Plot generation failed: {str(e)}'})

@app.route('/reset')
def reset():
    if session.get('df_original') is None:
        flash('No original data to reset to')
        return redirect(url_for('index'))
    
    # Reset all dataframes to original state
    session['df_main'] = session['df_original']
    session['df_metadata'] = None
    session['df_meta_thd'] = None
    session['df_sample'] = None
    session['df_history'] = []
    session['imputed_mask'] = None
    session['current_column'] = ''
    session['processing_steps'] = [] # Clear processing steps
    session['imputation_performed'] = False # Reset imputation flag
    session['differential_analysis_results'] = None
    session['latest_differential_analysis_method'] = None
    session['group_assignments'] = {} # Reset group-related session variables
    session['group_names'] = {}
    session['n_groups'] = 0
    session['group_vector'] = {}
    
    flash('Data reset to original state')
    return redirect(url_for('summary'))

def get_normalization_plots(df_before, df_after, plot_type='boxplot', group_vector=None, group_names=None):
    """Helper function to generate all plots for the normalization page."""
    plots = {}
    # Before normalization
    plots['dist_before'] = create_boxplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Processing', group_vector=group_vector, group_names=group_names)
    
    # After normalization
    plots['dist_after'] = create_boxplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Processing', group_vector=group_vector, group_names=group_names)
    return plots

def get_transformation_plots(df_before, df_after, plot_type='boxplot', group_vector=None, group_names=None):
    """Helper function to generate all plots for the transformation page."""
    plots = {}
    plots['dist_before'] = create_boxplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Processing', group_vector=group_vector, group_names=group_names)
    
    plots['dist_after'] = create_boxplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Processing', group_vector=group_vector, group_names=group_names)
    return plots

def get_scaling_plots(df_before, df_after, plot_type='boxplot', group_vector=None, group_names=None):
    """Helper function to generate all plots for the scaling page."""
    plots = {}
    plots['dist_before'] = create_boxplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Processing', group_vector=group_vector, group_names=group_names)
    
    plots['dist_after'] = create_boxplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names) if plot_type == 'boxplot' else create_violinplot(df_after, 'After Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Processing', group_vector=group_vector, group_names=group_names)
    return plots

@app.route('/reset_sample_step', methods=['POST'])
def reset_sample_step():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available to reset'})

    context = request.json.get('context', 'imputation') # Default to imputation for safety

    if len(session['df_history']) > 1:
        session['df_history'].pop()
        if session.get('processing_steps'):
            session['processing_steps'].pop()
    
    session.modified = True

    df_current = session['df_history'][-1]
    response_data = {
        'success': True,
        'message': 'Last processing step undone.',
        'new_shape': df_current.shape,
        'steps': session.get('processing_steps', [])
    }

    if context == 'imputation':
        session['imputation_performed'] = False
        session['imputed_mask'] = None
        response_data['missing_heatmap'] = create_heatmap_BW(
            df_current,
            title='Missing Values Distribution',
            imputed=False
        )
    elif context == 'normalization':
        df_before = session['df_history'][0]
        response_data['plots'] = get_normalization_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))
    elif context == 'transformation':
        df_before = session['df_history'][0]
        response_data['plots'] = get_transformation_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))
    elif context == 'scaling':
        df_before = session['df_history'][0]
        response_data['plots'] = get_scaling_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

    return jsonify(response_data)

@app.route('/analysis')
def analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('imputation'))
    
    df = session['df_history'][-1]
    df_html = df.to_html(classes='table table-striped table-hover', table_id='analysis-table')
    
    return render_template('analysis.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@app.route('/multivariate_analysis')
def multivariate_analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('imputation'))

    df_history = session.get('df_history', [])
    processing_steps = session.get('processing_steps', [])

    #history_options = [(-1, 'Original Data')]
    history_options = []
    for i, step in enumerate(processing_steps):
        history_options.append((i, step['message']))

    return render_template('multivariate_analysis.html',
                           history_options=history_options,
                           selected_history_index=len(df_history) - 1)

@app.route('/get_pca_plot/<int:history_index>')
def get_pca_plot(history_index):
    if not session.get('df_history') or history_index >= len(session['df_history']):
        return jsonify({'error': 'Invalid history index'})

    df = session['df_history'][history_index]
    
    plot = create_pca_plot(df, 'PCA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    
    return jsonify({'plot': plot})

@app.route('/get_hca_plot/', methods=['POST'])
def get_hca_plot():
    if not session.get('df_history'):
        return jsonify({'error': 'Invalid history index'})

    data = request.json
    distance_metric = data.get('distance_metric', 'euclidean')
    linkage_method = data.get('linkage_method', 'average')

    df = session['df_history'][-1]

    plot = create_hca_plot(df, 'HCA Plot', 
                           group_vector=session.get('group_vector'), 
                           group_names=session.get('group_names'),
                           distance_metric=distance_metric,
                           linkage_method=linkage_method)
    
    return jsonify({'plot': plot})

@app.route('/get_plsda_plot/<int:history_index>')
def get_plsda_plot(history_index):
    if not session.get('df_history') or history_index >= len(session['df_history']):
        return jsonify({'error': 'Invalid history index'})

    df = session['df_history'][history_index]
    
    plot = create_plsda_plot(df, 'PLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    
    return jsonify({'plot': plot})

@app.route('/get_oplsda_plot/<int:history_index>')
def get_oplsda_plot(history_index):
    if not session.get('df_history') or history_index >= len(session['df_history']):
        return jsonify({'error': 'Invalid history index'})

    df = session['df_history'][history_index]
    
    plot = create_oplsda_plot(df, 'OPLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
    
    return jsonify({'plot': plot})

@app.route('/get_metadata_columns')
def get_metadata_columns():
    if session.get('df_metadata') is not None:
        return jsonify({'columns': session['df_metadata'].columns.tolist()})
    return jsonify({'columns': []})

@app.route('/comparison')
def comparison():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_original = session['df_sample']
    df_history = session.get('df_history', [])
    processing_steps = session.get('processing_steps', [])

    # Create a list of tuples for the history dropdown
    history_options = []
    if len(df_history) > 1:
        for i, step in enumerate(processing_steps):
            # The first df in history is the original sample, so we skip it
            history_options.append((i + 1, step['message']))

    # Default to the last processed dataframe
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

@app.route('/get_comparison_data/<int:history_index>')
def get_comparison_data(history_index):
    if not session.get('df_history') or history_index >= len(session['df_history']):
        return jsonify({'error': 'Invalid history index'})

    df_processed = session['df_history'][history_index]
    processed_html = df_processed.to_html(classes='table table-striped table-sm', table_id='processed-table')

    return jsonify({
        'processed_html': processed_html,
        'processed_shape': df_processed.shape
    })

@app.route('/export_dataframe/<string:format>/<string:context>')
def export_dataframe(format, context):
    df = None
    # Define contexts that should include metadata
    contexts_with_metadata = ['analysis', 'comparison_original', 'comparison_processed']

    if context == 'main':
        df = session.get('df_main')
    elif context == 'analysis':
        if session.get('df_history'):
            df = session['df_history'][-1]
    elif context == 'differential_analysis_results':
        df = session.get('differential_analysis_results')
    elif context == 'comparison_original':
        df = session.get('df_sample')
    elif context == 'comparison_processed':
        if session.get('df_history'):
            df = session['df_history'][-1]
    else:
        return "Invalid context", 400

    if df is None:
        return "No data available to export", 404

    # Use a copy to avoid modifying the session data
    df = df.copy()

    # Append metadata if the context requires it
    if context in contexts_with_metadata:
        metadata_df = None
        if session.get('df_meta_thd') is not None and not session.get('df_meta_thd').empty:
            metadata_df = session.get('df_meta_thd')
        elif session.get('df_metadata') is not None and not session.get('df_metadata').empty:
            metadata_df = session.get('df_metadata')

        if metadata_df is not None and not metadata_df.empty:
            # The dataframes (df and metadata_df) share the same index (features).
            # We can concatenate them horizontally.
            df = pd.concat([metadata_df, df], axis=1)

    output = io.BytesIO()
    if format == 'csv':
        df.to_csv(output, index=True)
        mimetype = 'text/csv'
        filename = f'{context}_data.csv'
    elif format == 'tsv':
        df.to_csv(output, sep='\t', index=True)
        mimetype = 'text/tab-separated-values'
        filename = f'{context}_data.tsv'
    elif format == 'excel':
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=True)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        filename = f'{context}_data.xlsx'
    else:
        return "Invalid format", 400

    output.seek(0)

    return send_file(output, as_attachment=True, download_name=filename, mimetype=mimetype)


@app.route('/distribution/<column_name>')
def distribution(column_name):
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})
    
    df = session['df_history'][-1]
    
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    
    data_series = df[column_name].dropna()
    
    if data_series.empty:
        return jsonify({'error': 'No data available for this column'})
    
    # Create distribution plot
    plot = create_distribution_plot(data_series, column_name)
    
    return jsonify({'plot': plot})

@app.route('/differential_analysis')
def differential_analysis():
    if not session.get('df_history'):
        flash('Please process sample data first')
        return redirect(url_for('imputation'))
    
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


@app.route('/run_differential_analysis', methods=['POST'])
def run_differential_analysis():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    data = request.json
    test_type = data.get('test_type')
    groups = data.get('groups')
    correction_method = data.get('correction_method')
    paired_map = data.get('paired_map')
    formula = data.get('formula')

    df = session['df_history'][-1]
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
            metadata_df = session.get('df_metadata')
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

        session['differential_analysis_results'] = results_df
        session['latest_differential_analysis_method'] = test_type.replace("_", " ").title()
        # print(results_df.head(10))
        return jsonify({'html': results_df.to_html(classes='table table-striped table-hover', table_id='resultsTable', escape=False)})

    except Exception as e:
        # logging.error(f"Differential analysis failed: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'})

@app.route('/run_permanova', methods=['POST'])
def run_permanova_route():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    data = request.json
    distance_metric = data.get('distance_metric', 'euclidean')
    permutations = int(data.get('permutations', 999))

    df = session['df_history'][-1]
    group_vector = session['group_vector']

    try:
        result = run_permanova(df, group_vector, distance_metric, permutations)
        result_summary = {
            'test_statistic': result['test statistic'],
            'p_value': result['p-value']
        }
        session['permanova_results'] = result_summary
        return jsonify({'success': True, 'result': result_summary})
    except Exception as e:
        # logging.error(f"PERMANOVA analysis failed: {e}", exc_info=True)
        return jsonify({'error': f'PERMANOVA failed: {str(e)}'})

@app.route('/result_visualization')
def result_visualization():
    if 'differential_analysis_results' not in session or session['differential_analysis_results'] is None or session['differential_analysis_results'].empty:
        flash('Please run a differential analysis first.', 'warning')
        return redirect(url_for('differential_analysis'))

    results_df = session['differential_analysis_results']
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
                           metadata_columns=metadata_columns)

@app.route('/api/clustergram_data', methods=['POST'])
def clustergram_data():
    if 'differential_analysis_results' not in session or session['differential_analysis_results'].empty:
        return jsonify({'error': 'No analysis results found'}), 404

    data = request.json
    top_n = int(data.get('top_n', 50))
    distance_metric = data.get('distance_metric', 'euclidean')
    linkage_method = data.get('linkage_method', 'average')
    y_axis_label = data.get('y_axis_label') # New: get selected y-axis label

    results_df = session['differential_analysis_results']
    data_df = session['df_history'][-1]

    p_value_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
    if 'rejected' in results_df.columns:
        significant_features_df = results_df[results_df['rejected']]
    else:
        significant_features_df = results_df[results_df[p_value_col] < 0.05]

    if significant_features_df.empty:
        return jsonify({'error': 'No significant features found to generate a clustergram. Adjust p-value threshold or check analysis results.'}), 404

    significant_features = significant_features_df.nsmallest(top_n, p_value_col).index.unique()
    plot_data = data_df.loc[significant_features]
    numeric_plot_data = plot_data.select_dtypes(include=[np.number])
    
    plot_data_zscored = numeric_plot_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0, axis=0).fillna(0)

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
        aligned_metadata = feature_metadata_df.reindex(plot_data_reordered.index)
        y_labels = aligned_metadata[y_axis_label].tolist()

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
        'colorscale': 'RdBu',
        'zmin': -max_abs,
        'zmax': max_abs,
        'colorbar': {'title': 'Z-score'},
        'hoverinfo': 'x+y+z',
    }

    # Prepare customdata for hover (combining sample group and feature metadata)
    full_custom_data = []
    metadata_column_names = ['Group'] # Start with 'Group' for sample metadata

    # Get feature-level metadata (rt, mz, intensity)
    feature_metadata_df = None
    if session.get('df_meta_thd') is not None:
        feature_metadata_df = session.get('df_meta_thd')
    elif session.get('df_metadata') is not None:
        feature_metadata_df = session.get('df_metadata')

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
                cell_custom_data.extend(feature_meta_values)
            
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

    return jsonify({
        'heatmap': heatmap_trace,
        'col_dendro': col_dendro_traces,
        'row_dendro': row_dendro_traces,
        'column_labels': plot_data_reordered.columns.tolist(), # Samples
        'row_labels': y_labels, # Use the potentially updated labels
        'heatmap_x': heatmap_x,
        'heatmap_y': heatmap_y,
        'heatmap_customdata': full_custom_data,
        'metadata_column_names': metadata_column_names
    })

if __name__ == '__main__':
    app.run(debug=True)
