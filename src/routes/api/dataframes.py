from flask import Blueprint, render_template, session, flash, Response, redirect, url_for, jsonify, request
import pandas as pd
import io

from ...functions.exploratory_data import preprocessing_summary_perVariable
from ...functions.plot_definitions import create_pie_chart
from ... import data_manager

dataframes_bp = Blueprint('dataframes', __name__, url_prefix='/dataframes')

@dataframes_bp.route('/')
def dataframe_view():
    df = data_manager.load_dataframe('df_main_path')
    if df is None:
        flash('Please upload a file first','warning')
        return redirect(url_for('core.upload_file'))
    
    df_html = df.to_html(classes='table table-striped table-hover', table_id='dataframe-table')
    shape = df.shape
    columns = df.columns.tolist()
    del df
    
    return render_template('dataframe.html', 
                         df_html=df_html,
                         shape=shape,
                         columns=columns)
    

@dataframes_bp.route('/analysis')
def analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please process sample data first', 'warning')
        return redirect(url_for('preprocessing.imputation'))
    
    df = data_manager.load_dataframe(history_paths[-1])
    df_html = df.to_html(classes='table table-striped table-hover', table_id='analysis-table')
    
    return render_template('analysis.html', 
                         df_html=df_html,
                         shape=df.shape,
                         columns=df.columns.tolist())

@dataframes_bp.route('/export/<string:format>/<string:context>')
def export_dataframe(format, context):
    def get_dataframe_by_context(context):
        history_paths = session.get('df_history_paths', [])
        
        if context == 'main':
            return data_manager.load_dataframe('df_main_path')
        elif context == 'analysis':
            if history_paths:
                return data_manager.load_dataframe(history_paths[-1])
        elif context == 'differential_analysis_results':
            return data_manager.load_dataframe('differential_analysis_results_path')
        elif context == 'comparison_original':
            if history_paths:
                return data_manager.load_dataframe(history_paths[0])
        elif context == 'comparison_processed':
            if history_paths:
                return data_manager.load_dataframe(history_paths[-1])
        
        return None

    df = get_dataframe_by_context(context)

    if df is None:
        flash('No data available to export for the selected context.', 'warning')
        return redirect(request.referrer or url_for('core.index'))

    if context in ['analysis', 'comparison_original', 'comparison_processed']:
        metadata_df = data_manager.load_dataframe('df_metadata_path')
        if metadata_df is not None and not metadata_df.empty:
            if df.index.name != metadata_df.index.name:
                 metadata_df.index = df.index
            df = pd.concat([metadata_df, df], axis=1)

    output = io.BytesIO()
    
    format_details = {
        'csv': ('text/csv', f'{context}_data.csv'),
        'tsv': ('text/tab-separated-values', f'{context}_data.tsv'),
        'excel': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', f'{context}_data.xlsx')
    }

    if format not in format_details:
        return "Invalid format", 400

    mimetype, filename = format_details[format]

    if format == 'csv':
        df.to_csv(output, index=True)
    elif format == 'tsv':
        df.to_csv(output, sep='\t', index=True)
    elif format == 'excel':
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=True)
    
    output.seek(0)

    if 'df' in locals() and df is not None:
        del df
    if 'metadata_df' in locals() and metadata_df is not None:
        del metadata_df

    return Response(
        output.getvalue(),
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

def _get_column_info(df, column_name):
    if df is None:
        return jsonify({'error': 'No data loaded'})
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    col_data = df[[column_name]]
    stats = preprocessing_summary_perVariable(col_data)
    return jsonify({
        'stats': stats.T.to_html(classes='table table-sm')
    })

@dataframes_bp.route('/column_info/<column_name>')
def column_info(column_name):
    df = data_manager.load_dataframe('df_main_path')
    result = _get_column_info(df, column_name)
    del df
    return result

@dataframes_bp.route('/column_info_analysis/<column_name>')
def column_info_analysis(column_name):
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No data loaded'})
    df = data_manager.load_dataframe(history_paths[-1])
    return _get_column_info(df, column_name)

@dataframes_bp.route('/metadata_columns')
def get_metadata_columns():
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    if df_metadata is not None:
        return jsonify({'columns': df_metadata.columns.tolist()})
    return jsonify({'columns': []})