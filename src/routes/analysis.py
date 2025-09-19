from flask import Blueprint, render_template, session, flash, redirect, url_for

from ..functions.plot_definitions import create_volcano_plot # type: ignore
from .. import data_manager

analysis_bp = Blueprint('analysis', __name__, template_folder='../../templates')

@analysis_bp.route('/multivariate_analysis')
def multivariate_analysis():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please process sample data first', 'warning')
        return redirect(url_for('preprocessing.imputation'))

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
        flash('Please define sample data first', 'warning')
        return redirect(url_for('core.metadata'))
    
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
        flash('Please process sample data first', 'warning')
        return redirect(url_for('preprocessing.imputation'))
    
    group_names = session.get('group_names', {})
    group_vector = session.get('group_vector', {})
    
    results_html = None
    differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
    if differential_analysis_results is not None and not differential_analysis_results.empty:
        results_html = differential_analysis_results.to_html(classes='table table-striped table-sm', table_id='resultsTable', escape=False)
    
    if differential_analysis_results is not None:
        del differential_analysis_results

    any_log_transformed = 1 in session.get('step_transformation', [])
    any_scaled = 1 in session.get('step_scaling', [])
    any_normalized = 1 in session.get('step_normalization', [])

    return render_template('differential_analysis.html',
                           group_names=group_names,
                           group_vector=group_vector,
                           results_html=results_html,
                           latest_differential_analysis_method=session.get('latest_differential_analysis_method'),
                           is_log_transformed=any_log_transformed,
                           is_scaled=any_scaled,
                           is_normalized=any_normalized,
                           paired_data=session.get('paired_data', {}))

@analysis_bp.route('/result_visualization')
def result_visualization():
    differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
    if differential_analysis_results is None or differential_analysis_results.empty:
        flash('Please run a differential analysis first.', 'warning')
        return redirect(url_for('analysis.differential_analysis'))

    results_df = differential_analysis_results
    
    p_value_col = 'p_adj' if 'p_adj' in results_df.columns else 'p_value'
    
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
