from flask import Blueprint, render_template, session, flash, redirect, url_for, jsonify
import pandas as pd
import plotly.graph_objs as go
import json

from src.functions.exploratory_data import preprocessing_summary_perVariable

distribution_bp = Blueprint('distribution', __name__)

@distribution_bp.route('/distribution')
def distribution_view():
    if session.get('df_sample_thd') is None:
        flash('Please process sample data first')
        return redirect(url_for('imputation.imputation_view'))
    
    df = session['df_sample_thd']
    
    return render_template('distribution.html', 
                         columns=df.columns.tolist())

@distribution_bp.route('/distribution/plot/<column_name>')
def get_distribution_plot(column_name):
    if session.get('df_sample_thd') is None:
        return jsonify({'error': 'No sample data available'}), 404
    
    df = session['df_sample_thd']
    
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'}), 404
    
    data_series = df[column_name].dropna()
    
    if data_series.empty:
        return jsonify({'error': 'No data available for this column'}), 404
    
    # Create distribution plot
    plot = create_distribution_plot(data_series, column_name)
    
    # Get column stats
    stats_df = preprocessing_summary_perVariable(df[[column_name]])
    stats_html = stats_df.to_html(classes='table table-striped table-sm', index=False)

    return jsonify({'plot': plot, 'stats': stats_html})


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
