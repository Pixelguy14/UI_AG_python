from flask import Blueprint, jsonify, request, session, g
import numpy as np
import pandas as pd

from ...functions.plot_definitions import (
    create_density_plot, create_boxplot, create_violinplot,
    create_pca_plot, create_hca_plot, create_plsda_plot, create_oplsda_plot,
    create_feature_clustergram, create_pie_chart, create_heatmap_BW,
    create_heatmap, create_bar_plot, create_distribution_plot,
    create_volcano_plot, create_intensity_comparison_plot
)
from ...functions.exploratory_data import preprocessing_summary_perVariable
from ... import data_manager

plots_bp = Blueprint('plots', __name__, url_prefix='/plots')

# Helper to get the dataframe for summary plots, caching it in the request context `g`
def _get_df_for_summary():
    if 'summary_df' not in g:
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            g.summary_df = data_manager.load_dataframe('df_main_path')
        else:
            g.summary_df = data_manager.load_dataframe(history_paths[-1])
    return g.summary_df

@plots_bp.route('/summary/missing_heatmap')
def missing_heatmap():
    with data_manager.processing_lock():
        df_sample = _get_df_for_summary()
        if df_sample is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        plot = create_heatmap_BW(
            df_sample,
            title='Missing Values Distribution',
            imputed=session.get('imputation_performed', False),
            null_mask=session.get('imputed_mask')
        )
        return jsonify({'plot': plot})

@plots_bp.route('/summary/correlation_matrix')
def correlation_matrix():
    with data_manager.processing_lock():
        df_sample = _get_df_for_summary()
        if df_sample is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        numeric_df = df_sample.select_dtypes(include=['number'])
        if numeric_df.empty or numeric_df.shape[1] <= 1:
            return jsonify({'error': 'Not enough numeric data for correlation matrix.'}), 400
        corr_matrix = numeric_df.corr().fillna(0)
        plot = create_heatmap(corr_matrix, title='Correlation Matrix')
        return jsonify({'plot': plot})

@plots_bp.route('/summary/mean_intensity')
def mean_intensity():
    with data_manager.processing_lock():
        df_sample = _get_df_for_summary()
        if df_sample is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        numeric_df = df_sample.select_dtypes(include=['number'])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric data for mean intensity plot.'}), 400
        mean_values = numeric_df.mean()
        plot = create_bar_plot(
            x=mean_values.index.tolist(),
            y=mean_values.values.tolist(),
            title=f'Mean Intensity ({len(mean_values)} samples)',
            xaxis_title='Samples',
            yaxis_title='Mean log2 intensity',
            group_vector=session.get('group_vector'),
            group_names=session.get('group_names')
        )
        return jsonify({'plot': plot})

@plots_bp.route('/summary/all_columns_density')
def all_columns_density():
    with data_manager.processing_lock():
        df_sample = _get_df_for_summary()
        if df_sample is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        numeric_df = df_sample.select_dtypes(include=['number'])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric data for density plot.'}), 400
        plot = create_distribution_plot(
            numeric_df,
            'Distribution of each group',
            group_vector=session.get('group_vector'),
            group_names=session.get('group_names')
        )
        return jsonify({'plot': plot})

@plots_bp.route('/summary/boxplot_distribution')
def boxplot_distribution():
    with data_manager.processing_lock():
        df_sample = _get_df_for_summary()
        if df_sample is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        numeric_df = df_sample.select_dtypes(include=['number'])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric data for boxplot.'}), 400
        plot = create_boxplot(
            numeric_df,
            title='Distribution of Sample Data',
            group_vector=session.get('group_vector'),
            group_names=session.get('group_names')
        )
        return jsonify({'plot': plot})

# New route for data types plot from core.py
@plots_bp.route('/data_types_plot')
def data_types_plot():
    with data_manager.processing_lock():
        df = data_manager.load_dataframe('df_main_path')
        if df is None:
            return jsonify({'error': 'No data available to plot.'}), 404
        
        summary_stats = preprocessing_summary_perVariable(df)
        type_counts = summary_stats['type'].value_counts()
        plot = create_bar_plot(
            x=type_counts.index.tolist(),
            y=type_counts.values.tolist(),
            title='Data Types Distribution',
            xaxis_title='Data Type',
            yaxis_title='Count'
        )
        return jsonify({'plot': plot})

# New route for volcano plot from analysis.py
@plots_bp.route('/volcano_plot')
def volcano_plot():
    with data_manager.processing_lock():
        differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
        if differential_analysis_results is None or differential_analysis_results.empty:
            return jsonify({'error': 'Please run a differential analysis first.'}), 404

        metadata_df = data_manager.load_dataframe('df_meta_thd_path')
        if metadata_df is None or metadata_df.empty:
            metadata_df = data_manager.load_dataframe('df_metadata_path')
        
        plot = create_volcano_plot(
            results_df=differential_analysis_results,
            metadata_df=metadata_df
        )
        return jsonify({'plot': plot})

def _get_column_pie_plot(df, column_name):
    if df is None:
        return jsonify({'error': 'No data loaded'})
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    col_data = df[[column_name]]
    total_elements = len(col_data)
    total_nulls = col_data.isnull().sum().sum()
    total_non_nulls = total_elements - total_nulls
    null_plot = create_pie_chart(
        labels=['Null Values', 'Non-Null Values'],
        values=[total_nulls, total_non_nulls],
        title=f'Null Distribution - {column_name}'
    )
    return jsonify({
        'plot': null_plot
    })

@plots_bp.route('/column_pie_plot/<column_name>')
def column_pie_plot(column_name):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            return jsonify({'error': 'No data loaded'})
        df = data_manager.load_dataframe(history_paths[-1])
        return _get_column_pie_plot(df, column_name)

@plots_bp.route('/column_pie_plot_main/<column_name>')
def column_pie_plot_main(column_name):
    with data_manager.processing_lock():
        df = data_manager.load_dataframe('df_main_path')
        return _get_column_pie_plot(df, column_name)

def _get_column_density_plot(df, column_name):
    if df is None:
        return jsonify({'error': 'No data loaded'})
    if column_name not in df.columns:
        return jsonify({'error': 'Column not found'})
    col_plot_data = df[column_name].dropna()
    if col_plot_data.empty:
        return jsonify({'error': 'No data available for this column'})
    density_plot = create_density_plot(col_plot_data, column_name)
    return jsonify({
        'plot': density_plot
    })

@plots_bp.route('/column_density_plot/<column_name>')
def column_density_plot(column_name):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            return jsonify({'error': 'No data loaded'})
        df = data_manager.load_dataframe(history_paths[-1])
        return _get_column_density_plot(df, column_name)

@plots_bp.route('/column_density_plot_main/<column_name>')
def column_density_plot_main(column_name):
    with data_manager.processing_lock():
        df = data_manager.load_dataframe('df_main_path')
        return _get_column_density_plot(df, column_name)

@plots_bp.route('/distribution_plot/<plot_type>/<context>', methods=['GET'])
def get_distribution_plot(plot_type, context):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            df = data_manager.load_dataframe('df_main_path')
        else:
            if context == 'before':
                df = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])
            else:
                df = data_manager.load_dataframe(history_paths[-1])
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric data to plot'})

        try:
            if plot_type == 'boxplot':
                plot = create_boxplot(numeric_df, title='Current Data Distribution', group_vector=session.get('group_vector'), group_names=session.get('group_names'))
            elif plot_type == 'violinplot':
                plot = create_violinplot(numeric_df, title='Current Data Distribution', group_vector=session.get('group_vector'), group_names=session.get('group_names'))
            else:
                return jsonify({'error': 'Invalid plot type'})
            
            return jsonify({'plot': plot})
            
        except Exception as e:
            return jsonify({'error': f'Plot generation failed: {str(e)}'})

@plots_bp.route('/pca_plot/<int:history_index>')
def get_pca_plot(history_index):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths or history_index >= len(history_paths):
            return jsonify({'error': 'Invalid history index'})
        df = data_manager.load_dataframe(history_paths[history_index])
        if df is None:
            return jsonify({'error': 'Dataframe not found for this history step.'})
        plot = create_pca_plot(df, 'PCA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
        return jsonify({'plot': plot})

@plots_bp.route('/hca_plot', methods=['POST'])
def get_hca_plot():
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            return jsonify({'error': 'Invalid history index'})
        data = request.json
        distance_metric = data.get('distance_metric', 'euclidean')
        linkage_method = data.get('linkage_method', 'average')
        df = data_manager.load_dataframe(history_paths[-1])
        plot = create_hca_plot(df, 'HCA Plot',
                               group_vector=session.get('group_vector'),
                               group_names=session.get('group_names'),
                               distance_metric=distance_metric,
                               linkage_method=linkage_method)
        return jsonify({'plot': plot})

@plots_bp.route('/plsda_plot/<int:history_index>')
def get_plsda_plot(history_index):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths or history_index >= len(history_paths):
            return jsonify({'error': 'Invalid history index'})
        df = data_manager.load_dataframe(history_paths[history_index])
        if df is None:
            return jsonify({'error': 'Dataframe not found for this history step.'})
        plot = create_plsda_plot(df, 'PLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
        return jsonify({'plot': plot})

@plots_bp.route('/oplsda_plot/<int:history_index>')
def get_oplsda_plot(history_index):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths or history_index >= len(history_paths):
            return jsonify({'error': 'Invalid history index'})
        df = data_manager.load_dataframe(history_paths[history_index])
        if df is None:
            return jsonify({'error': 'Dataframe not found for this history step.'})
        plot = create_oplsda_plot(df, 'OPLS-DA Plot', group_vector=session.get('group_vector'),group_names=session.get('group_names'))
        return jsonify({'plot': plot})

@plots_bp.route('/clustergram_data', methods=['POST'])
def clustergram_data():
    with data_manager.processing_lock():
        data = request.json
        differential_analysis_results = data_manager.load_dataframe('differential_analysis_results_path')
        if differential_analysis_results is None or differential_analysis_results.empty:
            return jsonify({'error': 'No analysis results found'}), 404

        history_paths = session.get('df_history_paths', [])
        data_df = data_manager.load_dataframe(history_paths[-1])
        feature_metadata_df = data_manager.load_dataframe('df_meta_thd_path')
        if feature_metadata_df is None or feature_metadata_df.empty:
            feature_metadata_df = data_manager.load_dataframe('df_metadata_path')

        result = create_feature_clustergram(
            results_df=differential_analysis_results,
            data_df=data_df,
            feature_metadata_df=feature_metadata_df,
            group_vector=session.get('group_vector', {}),
            group_names=session.get('group_names', {}),
            is_scaled=1 in session.get('step_scaling', []),
            top_n=int(data.get('top_n', 50)),
            distance_metric=data.get('distance_metric', 'euclidean'),
            linkage_method=data.get('linkage_method', 'average'),
            y_axis_label=data.get('y_axis_label'),
            color_palette=data.get('color_palette', 'RdBu')
        )
        return jsonify(result)

# Helper function from preprocessing.py
def get_comparison_plots(df_before, df_after, plot_type, group_vector, group_names, processing_step_name):
    plots = {}
    plot_func = create_boxplot if plot_type == 'boxplot' else create_violinplot
    plots['dist_before'] = plot_func(df_before, f'Before {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, f'PCA Before {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['dist_after'] = plot_func(df_after, f'After {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, f'PCA After {processing_step_name}', group_vector=group_vector, group_names=group_names)
    return plots

# New route for imputation plots from preprocessing.py
@plots_bp.route('/preprocessing/imputation_plots')
def get_imputation_plots():
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            return jsonify({'error': 'No data available.'}), 404
        
        df_current = data_manager.load_dataframe(history_paths[-1])
        if df_current is None:
            return jsonify({'error': 'Current dataframe not found.'}), 404

        df_before = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])
        if df_before is None:
            return jsonify({'error': 'Previous dataframe not found.'}), 404

        missing_heatmap = create_heatmap_BW(
            df_current,
            title='Missing Values Distribution (Imputed Highlighted)',
            imputed=session.get('imputation_performed', False),
            null_mask=session.get('imputed_mask')
        )
        intensity_comparison_plot = create_intensity_comparison_plot(
            original_df=df_before,
            imputed_df=df_current,
            apply_log_transform=not (1 in session.get('step_transformation', []))
        )
        return jsonify({
            'missing_heatmap': missing_heatmap,
            'intensity_comparison_plot': intensity_comparison_plot
        })

# New route for preprocessing comparison plots
@plots_bp.route('/preprocessing/comparison_plots/<plot_type>/<context_name>')
def get_preprocessing_comparison_plots(plot_type, context_name):
    with data_manager.processing_lock():
        history_paths = session.get('df_history_paths', [])
        if not history_paths:
            return jsonify({'error': 'No data available.'}), 404
        
        df_current = data_manager.load_dataframe(history_paths[-1])
        if df_current is None:
            return jsonify({'error': 'Current dataframe not found.'}), 404

        df_before = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])
        if df_before is None:
            return jsonify({'error': 'Previous dataframe not found.'}), 404

        plots = get_comparison_plots(df_before, df_current, plot_type, session.get('group_vector'), session.get('group_names'), context_name)
        return jsonify({'plots': plots})
