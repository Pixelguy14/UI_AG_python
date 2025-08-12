from flask import Blueprint, render_template, request, session, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import io
import pandas as pd
import numpy as np

# Import functions from other modules within the src package
from ..functions.exploratory_data import loadFile, preprocessing_summary_perVariable, preprocessing_general_dataset_statistics # type: ignore
from ..functions.plot_definitions import create_bar_plot, create_heatmap, create_heatmap_BW, create_distribution_plot, create_boxplot # type: ignore

# 1. Create a Blueprint object
core_bp = Blueprint('core', __name__, template_folder='../../templates', static_folder='../../static')

# 2. Change the route decorator from @app.route to @core_bp.route
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
            # Reset session state for a new file upload
            session.clear()
            
            filename = secure_filename(file.filename)
            # Correct the path for app factory structure
            upload_folder = os.path.join(os.path.dirname(__file__), '../../uploads')
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            try:
                # Load the file
                df = loadFile(filepath)
                
                if df is None or df.empty:
                    flash('Failed to load data or the file is empty.', 'warning')
                    return redirect(request.url)
                # Handle orientation
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
                
                session['df_main'] = df
                session['df_original'] = df
                session['orientation'] = orientation
                
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
    
    return render_template('upload.html', df_preview_html=None, shape=None)

@core_bp.route('/summary')
def summary():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('core.upload_file'))
    
    df = session['df_main']
    
    plots = {}
    if session.get('df_history') and not session['df_history'][-1].empty:
        df_sample = session['df_history'][-1]
        general_stats = preprocessing_general_dataset_statistics(df_sample)
        numeric_df = df_sample.select_dtypes(include=[np.number])
        
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            corr_matrix = corr_matrix.fillna(0)
            plots['correlation'] = create_heatmap(
                corr_matrix,
                title='Correlation Matrix'
            )
        
        plots['missing_heatmap'] = create_heatmap_BW(
            df_sample,
            title='Missing Values Distribution',
            imputed=session.get('imputation_performed', False),
            null_mask=session.get('imputed_mask')
        )
        
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

        numeric_df_for_boxplot = df_sample.select_dtypes(include=[np.number])
        if not numeric_df_for_boxplot.empty:
            plots['boxplot_distribution'] = create_boxplot(
                numeric_df_for_boxplot,
                title='Distribution of Sample Data',
                group_vector=session.get('group_vector'),
                group_names=session.get('group_names')
            )
    else:
        plots['missing_heatmap'] = create_heatmap_BW(
            df.isnull().astype(int),
            title='Missing Values Distribution'
        )
        general_stats = preprocessing_general_dataset_statistics(df)

    return render_template('summary.html', 
                         general_stats=general_stats.to_html(classes='table table-striped'),
                         plots=plots)

@core_bp.route('/dataframe')
def dataframe_view():
    if session.get('df_main') is None:
        flash('Please upload a file first')
        return redirect(url_for('core.upload_file'))
    
    df = session['df_main']
    df_html = df.to_html(classes='table table-striped table-hover', table_id='dataframe-table')
    
    return render_template('dataframe.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@core_bp.route('/reset')
def reset():
    if session.get('df_original') is None:
        flash('No original data to reset to')
        return redirect(url_for('core.index'))
    
    session.clear()
    flash('Data reset to original state')
    return redirect(url_for('core.upload_file'))

@core_bp.route('/export_dataframe/<string:format>/<string:context>')
def export_dataframe(format, context):
    df = None
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

    df = df.copy()

    if context in contexts_with_metadata:
        metadata_df = None
        if session.get('df_meta_thd') is not None and not session.get('df_meta_thd').empty:
            metadata_df = session.get('df_meta_thd')
        elif session.get('df_metadata') is not None and not session.get('df_metadata').empty:
            metadata_df = session.get('df_metadata')

        if metadata_df is not None and not metadata_df.empty:
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