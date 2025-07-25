# import json
import logging
import os

import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_session import Session  # Import Flask-Session
from werkzeug.utils import secure_filename

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
    # is_normalization,
    # svr_normalization,
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
    create_pca_plot,
    create_pie_chart,
    create_violinplot,
)
from src.views.distributionTabView import distribution_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.register_blueprint(distribution_bp)

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
            reset()
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
                    df = df.T
                
                # Clean column names (remove paths)
                rename_map = {col: os.path.basename(col) for col in df.columns if '/' in col or '\\' in col}
                if rename_map:
                    df.rename(columns=rename_map, inplace=True)
                
                # Store full DataFrame in session
                session['df_main'] = df
                session['df_original'] = df
                session['orientation'] = orientation
                
                # Create a preview for the upload page
                df_preview_html = df.head(10).to_html(classes='table table-striped table-hover table-sm', table_id='dataframe-preview-table', border=0)
                
                flash('File uploaded successfully! A preview is shown below.', 'success')
                
                # Re-render the upload page, now with the preview
                return render_template('upload.html', 
                                     df_preview_html=df_preview_html,
                                     shape=df.shape)
                
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

    # Generate summary statistics
    general_stats = preprocessing_general_dataset_statistics(df)
    summary_stats = preprocessing_summary_perVariable(df)
    
    # Create plots
    plots = {}
    
    # Data types distribution
    type_counts = summary_stats['type'].value_counts()
    plots['data_types'] = create_bar_plot(
        x=type_counts.index.tolist(),
        y=type_counts.values.tolist(),
        title='Data Types Distribution',
        xaxis_title='Data Type',
        yaxis_title='Count'
    )
    
    # Correlation matrix (if we have sample data)
    if session.get('df_history') and not session['df_history'][-1].empty:
        df_sample = session['df_history'][-1]
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
                yaxis_title='Mean log2 intensity'
            )

        # Boxplot with all points
        numeric_df_for_boxplot = df_sample.select_dtypes(include=[np.number])
        if not numeric_df_for_boxplot.empty:
            plots['boxplot_distribution'] = create_boxplot(
                numeric_df_for_boxplot,
                title='Distribution of Sample Data'
            )
    else:
        # Missing values heatmap
        plots['missing_heatmap'] = create_heatmap_BW(
            df.isnull().astype(int),
            title='Missing Values Distribution'
        )

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
        assignments = request.json
        
        # Process assignments
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
        
        return jsonify({'success': True, 'message': 'Metadata assignments saved successfully'})
    
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
                         existing_sample=existing_sample)

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
    
    # Apply thresholding
    num_columns = len(df_sample.columns)
    threshold_count = max(1, int((threshold_percent / 100.0) * num_columns)) if num_columns > 0 else 0
    
    df_thresholded = df_sample.dropna(thresh=threshold_count)
    
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
        session['processing_steps'].append({'icon': 'fa-magic', 'color': 'text-primary', 'message': f'Applied {method} imputation. New shape: {imputed_df.shape[0]} rows, {imputed_df.shape[1]} columns.'})
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
        logging.error(f"Imputation failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Imputation failed: {str(e)}'})

@app.route('/normalization')
def normalization():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    # df_sample = session['df_history'][-1]
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    df_current = session['df_history'][-1]
    df_before = session['df_history'][0] # Assuming the first entry is the original sample data

    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_normalization_plots(df_before, df_current)

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
        plots = get_normalization_plots(df_before_normalization, normalized_df)

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

    plots = get_transformation_plots(df_before, df_current)

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
        plots = get_transformation_plots(df_before_transformation, transformed_df, plot_type)

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} transformation applied successfully.',
            'new_shape': transformed_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        logging.error(f"Transformation failed for method {method}: {e}", exc_info=True)
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

    plots = get_scaling_plots(df_before, df_current)

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
        session.modified = True

        df_before_scaling = session['df_history'][0]
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_scaling_plots(df_before_scaling, scaled_df, plot_type)

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} scaling applied successfully.',
            'new_shape': scaled_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        logging.error(f"Scaling failed for method {method}: {e}", exc_info=True)
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
            plot = create_boxplot(numeric_df, title='Current Data Distribution')
        elif plot_type == 'violinplot':
            plot = create_violinplot(numeric_df, title='Current Data Distribution')
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
    session['df_sample'] = None
    session['df_history'] = []
    session['imputed_mask'] = None
    session['current_column'] = ''
    session['processing_steps'] = [] # Clear processing steps
    session['imputation_performed'] = False # Reset imputation flag
    
    flash('Data reset to original state')
    return redirect(url_for('summary'))

def get_normalization_plots(df_before, df_after, plot_type='boxplot'):
    """Helper function to generate all plots for the normalization page."""
    plots = {}
    # Before normalization
    plots['dist_before'] = create_boxplot(df_before, 'Before Normalization') if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Normalization')
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Normalization')
    
    # After normalization
    plots['dist_after'] = create_boxplot(df_after, 'After Normalization') if plot_type == 'boxplot' else create_violinplot(df_after, 'After Normalization')
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Normalization')
    return plots

def get_transformation_plots(df_before, df_after, plot_type='boxplot'):
    """Helper function to generate all plots for the transformation page."""
    plots = {}
    plots['dist_before'] = create_boxplot(df_before, 'Before Transformation') if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Transformation')
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Transformation')
    
    plots['dist_after'] = create_boxplot(df_after, 'After Transformation') if plot_type == 'boxplot' else create_violinplot(df_after, 'After Transformation')
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Transformation')
    return plots

def get_scaling_plots(df_before, df_after, plot_type='boxplot'):
    """Helper function to generate all plots for the scaling page."""
    plots = {}
    plots['dist_before'] = create_boxplot(df_before, 'Before Scaling') if plot_type == 'boxplot' else create_violinplot(df_before, 'Before Scaling')
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Scaling')
    
    plots['dist_after'] = create_boxplot(df_after, 'After Scaling') if plot_type == 'boxplot' else create_violinplot(df_after, 'After Scaling')
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Scaling')
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
        response_data['plots'] = get_normalization_plots(df_before, df_current)
    elif context == 'transformation':
        df_before = session['df_history'][0]
        response_data['plots'] = get_transformation_plots(df_before, df_current)
    elif context == 'scaling':
        df_before = session['df_history'][0]
        response_data['plots'] = get_scaling_plots(df_before, df_current)

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

@app.route('/comparison')
def comparison():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_original = session['df_sample']
    df_processed = session['df_history'][-1]
    
    original_html = df_original.to_html(classes='table table-striped table-sm')
    processed_html = df_processed.to_html(classes='table table-striped table-sm')
    
    return render_template('comparison.html',
                         original_html=original_html,
                         processed_html=processed_html,
                         original_shape=df_original.shape,
                         processed_shape=df_processed.shape)

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

if __name__ == '__main__':
    app.run(debug=True)
