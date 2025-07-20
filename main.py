import json
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
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
    if 'df_sample_thd' not in session:
        session['df_sample_thd'] = None
    if 'df_sample_imp' not in session:
        session['df_sample_imp'] = None
    if 'df_sample_pre_imp' not in session:
        session['df_sample_pre_imp'] = None
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
    if session.get('df_sample_thd') is not None and not session.get('df_sample_thd').empty:
        df_sample = session['df_sample_thd']
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
            df_sample.isnull().astype(int),
            title='Missing Values Distribution',
            imputed=session['imputation_performed']
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
    if session.get('df_sample_thd') is None:
        return jsonify({'error': 'No data loaded'})
    
    df = session['df_sample_thd']
    
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
    if session.get('df_sample_thd') is None:
        return jsonify({'error': 'No data loaded'})

    df = session['df_sample_thd']

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
        session['df_sample_thd'] = df_sample if not df_sample.empty else pd.DataFrame()
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
    if session.get('df_sample') is None:
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_sample = session['df_sample']
    df_sample_thd = session.get('df_sample_thd', session['df_sample'])
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = {}
    # Missing values heatmap for imputation tab
    plots['missing_heatmap'] = create_heatmap_BW(
        df_sample_thd.isnull().astype(int),
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session['imputation_performed']
    )

    return render_template('imputation.html',
                         original_shape=df_sample.shape,
                         current_shape=df_sample_thd.shape,
                         processing_steps=session['processing_steps'],
                         plots=plots)

@app.route('/threshold', methods=['POST'])
def threshold():
    if session.get('df_sample') is None:
        return jsonify({'error': 'No sample data available'})
    
    threshold_percent = float(request.json.get('threshold', 80))
    
    df_sample = session['df_sample']
    
    # Apply thresholding
    num_columns = len(df_sample.columns)
    threshold_count = max(1, int((threshold_percent / 100.0) * num_columns)) if num_columns > 0 else 0
    
    df_thresholded = df_sample.dropna(thresh=threshold_count)
    
    # Store result
    session['df_sample_thd'] = df_thresholded
    
    # Update processing steps
    session['processing_steps'].append({'icon': 'fa-filter', 'color': 'text-info', 'message': f'Applied thresholding: {threshold_percent}% non-null values. New shape: {df_thresholded.shape[0]} rows, {df_thresholded.shape[1]} columns.'})
    session.modified = True # Mark session as modified

    # Generate and return the updated heatmap data
    updated_heatmap = create_heatmap_BW(
        df_thresholded.isnull().astype(int),
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session['imputation_performed']
    )

    return jsonify({
        'success': True,
        'original_shape': df_sample.shape,
        'new_shape': df_thresholded.shape,
        'message': f'Thresholding applied with {threshold_percent}%',
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap
    })

@app.route('/apply_imputation', methods=['POST'])
def apply_imputation():
    if session.get('df_sample_thd') is None:
        return jsonify({'error': 'No thresholded sample data available'})
    
    method = request.json.get('method')
    params = request.json.get('params', {})
    
    df_sample_thd = session['df_sample_thd']
    
    # Store the state before imputation to identify imputed values
    session['df_sample_pre_imp'] = df_sample_thd.copy()

    try:
        # Apply scaling for advanced methods
        df_scaled = (df_sample_thd - df_sample_thd.mean()) / df_sample_thd.std()
        
        if method == 'n_imputation':
            n_val = params.get('n_value', 0)
            imputed_df = nImputed(df_sample_thd, n=n_val)
        elif method == 'half_minimum':
            imputed_df = halfMinimumImputed(df_sample_thd)
        elif method == 'mean':
            imputed_df = meanImputed(df_sample_thd)
        elif method == 'median':
            imputed_df = medianImputed(df_sample_thd)
        elif method == 'miss_forest':
            max_iter = params.get('max_iter', 10)
            n_estimators = params.get('n_estimators', 100)
            imputed_df = missForestImputed(df_scaled, max_iter=max_iter, n_estimators=n_estimators)
            imputed_df = postprocess_imputation(imputed_df, df_sample_thd)
        elif method == 'svd':
            n_components = params.get('n_components', 5)
            imputed_df = svdImputed(df_scaled, n_components=n_components)
            imputed_df = postprocess_imputation(imputed_df, df_sample_thd)
        elif method == 'knn':
            n_neighbors = params.get('n_neighbors', 2)
            imputed_df = knnImputed(df_scaled, n_neighbors=n_neighbors)
            imputed_df = postprocess_imputation(imputed_df, df_sample_thd)
        elif method == 'mice_bayesian':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceBayesianRidgeImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df_sample_thd)
        elif method == 'mice_linear':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceLinearRegressionImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df_sample_thd)
        else:
            return jsonify({'error': 'Unknown imputation method'})
        
        # Store results
        session['df_sample_imp'] = imputed_df
        session['df_sample_thd'] = imputed_df
        
        # Calculate imputed mask: where was it null before and now it's not null
        df_sample_pre_imp = session['df_sample_pre_imp']
        imputed_mask = df_sample_pre_imp.isnull() & ~imputed_df.isnull()
        session['imputed_mask'] = imputed_mask

        # Update processing steps
        session['processing_steps'].append({'icon': 'fa-magic', 'color': 'text-primary', 'message': f'Applied {method} imputation. New shape: {imputed_df.shape[0]} rows, {imputed_df.shape[1]} columns.'})
        session.modified = True # Mark session as modified
        session['imputation_performed'] = True # Set flag that imputation has occurred
        
        # Generate and return the updated heatmap data
        updated_heatmap = create_heatmap_BW(
            df_sample_thd,
            title='Missing Values Distribution (Imputed Highlighted)',
            imputed=session['imputation_performed']
        )

        return jsonify({
            'success': True,
            'message': f'Imputation with {method} completed successfully',
            'steps': session['processing_steps'],
            'new_shape': imputed_df.shape,
            'missing_heatmap': updated_heatmap
        })
        
    except Exception as e:
        return jsonify({'error': f'Imputation failed: {str(e)}'})

@app.route('/normalization')
def normalization():
    if session.get('df_sample') is None:
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_sample = session['df_sample']
    df_sample_thd = session.get('df_sample_thd', session['df_sample'])
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = {}
    # Missing values heatmap for imputation tab
    plots['missing_heatmap'] = create_heatmap_BW(
        df_sample_thd.isnull().astype(int),
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session['imputation_performed']
    )

    return render_template('normalization.html',
                         original_shape=df_sample.shape,
                         current_shape=df_sample_thd.shape,
                         processing_steps=session['processing_steps'],
                         plots=plots)

@app.route('/reset')
def reset():
    if session.get('df_original') is None:
        flash('No original data to reset to')
        return redirect(url_for('index'))
    
    # Reset all dataframes to original state
    session['df_main'] = session['df_original']
    session['df_metadata'] = None
    session['df_sample'] = None
    session['df_sample_thd'] = None
    session['df_sample_imp'] = None
    session['df_sample_pre_imp'] = None
    session['imputed_mask'] = None
    session['current_column'] = ''
    session['processing_steps'] = [] # Clear processing steps
    session['imputation_performed'] = False # Reset imputation flag
    
    flash('Data reset to original state')
    return redirect(url_for('summary'))

@app.route('/reset_imputation', methods=['POST'])
def reset_imputation():
    if session.get('df_sample') is None:
        return jsonify({'error': 'No sample data available to reset imputation'})

    df_sample = session['df_sample']
    session['df_sample_thd'] = df_sample.copy() # Reset to original sample data
    session['df_sample_imp'] = None
    session['df_sample_pre_imp'] = None
    session['imputed_mask'] = None

    # Reset processing steps for imputation tab
    session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]
    session.modified = True
    session['imputation_performed'] = False # Reset imputation flag

    return jsonify({
        'success': True,
        'message': 'Imputation data reset to original sample data.',
        'new_shape': df_sample.shape,
        'steps': session['processing_steps']
    })

@app.route('/analysis')
def analysis():
    if session.get('df_sample_thd') is None:
        flash('Please process sample data first')
        return redirect(url_for('imputation'))
    
    df = session['df_sample_thd']
    df_html = df.to_html(classes='table table-striped table-hover', table_id='analysis-table')
    
    return render_template('analysis.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@app.route('/comparison')
def comparison():
    if session.get('df_sample') is None:
        flash('Please define sample data first')
        return redirect(url_for('metadata'))
    
    df_original = session['df_sample']
    df_processed = session.get('df_sample_thd', session['df_sample'])
    
    original_html = df_original.to_html(classes='table table-striped table-sm')
    processed_html = df_processed.to_html(classes='table table-striped table-sm')
    
    return render_template('comparison.html',
                         original_html=original_html,
                         processed_html=processed_html,
                         original_shape=df_original.shape,
                         processed_shape=df_processed.shape)

@app.route('/distribution/<column_name>')
def distribution(column_name):
    if session.get('df_sample_thd') is None:
        return jsonify({'error': 'No sample data available'})
    
    df = session['df_sample_thd']
    
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    
    data_series = df[column_name].dropna()
    
    if data_series.empty:
        return jsonify({'error': 'No data available for this column'})
    
    # Create distribution plot
    plot = create_distribution_plot(data_series, column_name)
    
    return jsonify({'plot': plot})

# Helper functions for creating plots
def create_bar_plot(x, y, title, xaxis_title, yaxis_title):
    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color='#440154')])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template='plotly_white'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_heatmap(data, title):
    z_data = data.values.tolist()
    x_data = data.columns.tolist() if hasattr(data, 'columns') else None
    y_data = data.index.tolist() if hasattr(data, 'index') else None

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale='Viridis'
    ))
    fig.update_layout(title=title, template='plotly_white')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_heatmap_BW(data, title, imputed=False):
    z_data = data.values.tolist()
    x_data = data.columns.tolist() if hasattr(data, 'columns') else None
    y_data = data.index.tolist() if hasattr(data, 'index') else None

    if not imputed:
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale='Purples_r',
            showscale=False
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale='Cividis',
            showscale=False
        ))
    
    fig.update_layout(title=title, template='plotly_white')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_pie_chart(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title, template='plotly_white')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_distribution_plot(data_series, column_name):
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data_series,
        nbinsx=30,
        name='Histogram',
        opacity=0.7
    ))
    
    # Add statistics text
    stats_text = f"n = {len(data_series)}<br>"
    stats_text += f"μ = {data_series.mean():.4f}<br>"
    stats_text += f"σ = {data_series.std():.4f}<br>"
    stats_text += f"Skew = {data_series.skew():.4f}<br>"
    stats_text += f"Kurt = {data_series.kurtosis():.4f}"
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'Distribution of {column_name}',
        xaxis_title='Value',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_density_plot(data_series, column_name):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data_series.tolist(), # Convert pandas Series to a list
        histnorm='probability density',
        name='Density',
        marker_color='#440154',
        opacity=0.7
    ))

    fig.update_layout(
        title=f'Density Plot of {column_name}',
        xaxis_title='Value',
        yaxis_title='Density',
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_boxplot(df, title):
    data = []
    for col in df.columns:
        # Boxplot for the column
        data.append(go.Box(
            y=df[col].tolist(), # Convert pandas Series to a list
            name=col,
            #boxpoints='all', # Display all points
            jitter=0.3,      # Add jitter to points
            pointpos=-1.8,   # Position points relative to box
            marker_color='#440154', # Box color
            line_color='#440154',   # Line color
            hoverinfo='y',   # Show y-value on hover
            showlegend=False
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        yaxis_title='Value',
        template='plotly_white',
        showlegend=True, # Show legend for column names
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True)
