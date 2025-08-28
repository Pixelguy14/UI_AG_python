from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import numpy as np

# Import processing functions
from ..functions.imputation_methods import ( # type: ignore
    halfMinimumImputed, knnImputed, meanImputed, medianImputed, 
    miceBayesianRidgeImputed, miceLinearRegressionImputed, missForestImputed, 
    nImputed, postprocess_imputation, svdImputed
)
from ..functions.normalization_methods import ( # type: ignore
    tic_normalization, mtic_normalization, median_normalization, 
    quantile_normalization, pqn_normalization
)
from ..functions.log_transfomation_methods import ( # type: ignore
    log2_transform, log10_transform, sqrt_transform, cube_root_transform, 
    arcsinh_transform, glog_transform, yeo_johnson_transform
)
from ..functions.scaling_methods import ( # type: ignore
    standard_scaling, minmax_scaling, pareto_scaling, 
    range_scaling, robust_scaling, vast_scaling
)
from ..functions.plot_definitions import ( # type: ignore
    create_heatmap_BW, create_intensity_comparison_plot, create_boxplot, 
    create_violinplot, create_pca_plot
)
from .. import data_manager

processing_bp = Blueprint('processing', __name__, template_folder='../../templates')

# Helper functions for plotting (could be moved to a helper module)
def get_comparison_plots(df_before, df_after, plot_type, group_vector, group_names, processing_step_name):
    plots = {}
    plot_func = create_boxplot if plot_type == 'boxplot' else create_violinplot
    plots['dist_before'] = plot_func(df_before, f'Before {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, f'PCA Before {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['dist_after'] = plot_func(df_after, f'After {processing_step_name}', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, f'PCA After {processing_step_name}', group_vector=group_vector, group_names=group_names)
    return plots

def _apply_processing_method(method_name, method_function, df, params):
    """Helper function to apply a processing method and return the result."""
    try:
        return method_function(df, **params)
    except Exception as e:
        raise ValueError(f"Error applying {method_name}: {e}")

@processing_bp.route('/imputation')
def imputation():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_sample = data_manager.load_dataframe(history_paths[-1])
    
    # Initialize steps if they are empty
    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = {}
    # Missing values heatmap for imputation tab
    plots['missing_heatmap'] = create_heatmap_BW(
        df_sample,
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session.get('imputation_performed', False),
        null_mask=session.get('imputed_mask')
    )
    df_before=None
    if len(history_paths) > 1:
        df_before = data_manager.load_dataframe(history_paths[-2])
    else:
        df_before = data_manager.load_dataframe(history_paths[0])

    plots['intensity_comparison_plot'] = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_sample,
        log_transform=True
    )
    original_df = data_manager.load_dataframe(history_paths[0])
    return render_template('imputation.html',
                         original_shape=original_df.shape,
                         current_shape=df_sample.shape,
                         processing_steps=session['processing_steps'],
                         plots=plots)

@processing_bp.route('/threshold', methods=['POST'])
def threshold():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})
    
    threshold_percent = float(request.json.get('threshold', 80))
    
    df_sample = data_manager.load_dataframe(history_paths[-1])
    df_metadata = data_manager.load_dataframe('df_metadata_path')
    
    # Apply thresholding
    num_columns = len(df_sample.columns)
    threshold_count = max(1, int((threshold_percent / 100.0) * num_columns)) if num_columns > 0 else 0
    
    df_thresholded = df_sample.dropna(thresh=threshold_count)
    
    # Cut metadata if it exists
    if df_metadata is not None and not df_metadata.empty:
        df_meta_thd = df_metadata.loc[df_thresholded.index]
        data_manager.save_dataframe(df_meta_thd, 'df_meta_thd_path', 'df_meta_thd')
    else:
        session['df_meta_thd_path'] = None
    
    # Store result
    history_key = f'df_history_{len(history_paths)}'
    data_manager.save_dataframe(df_thresholded, history_key, 'df_history')
    history_paths.append(history_key)
    session['df_history_paths'] = history_paths
    
    # Update processing steps
    session['processing_steps'].append({'icon': 'fa-filter', 'color': 'text-info', 'message': f'Applied thresholding: {threshold_percent}% non-null values. New shape: {df_thresholded.shape[0]} rows, {df_thresholded.shape[1]} columns.'})
    session.modified = True # Mark session as modified

    # Generate and return the updated heatmap data
    updated_heatmap = create_heatmap_BW(
        df_thresholded,
        title='Missing Values Distribution (Imputed Highlighted)',
        imputed=session.get('imputation_performed', False),
        null_mask=session.get('imputed_mask')
    )
    df_before=None
    if len(history_paths) > 1:
        df_before = data_manager.load_dataframe(history_paths[-2])
    else:
        df_before = data_manager.load_dataframe(history_paths[0])
    intensityComparisonPlot = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_thresholded,
        log_transform=True
    )
    original_df = data_manager.load_dataframe(history_paths[0])
    return jsonify({
        'success': True,
        'original_shape': original_df.shape,
        'new_shape': df_thresholded.shape,
        'message': f'Thresholding applied with {threshold_percent}%',
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap,
        "intensity_comparison_plot":intensityComparisonPlot
    })

@processing_bp.route('/apply_imputation', methods=['POST'])
def apply_imputation():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No thresholded sample data available'})
    
    method = request.json.get('method')
    params = request.json.get('params', {})
    
    df = data_manager.load_dataframe(history_paths[-1])

    try:
        # Apply scaling for advanced methods
        df_scaled = (df - df.mean()) / df.std()
        
        imputation_methods = {
            'n_imputation': (nImputed, {'n': params.get('n_value', 0)}),
            'half_minimum': (halfMinimumImputed, {}),
            'mean': (meanImputed, {}),
            'median': (medianImputed, {}),
            'miss_forest': (missForestImputed, {'max_iter': params.get('max_iter', 10), 'n_estimators': params.get('n_estimators', 100)}),
            'svd': (svdImputed, {'n_components': params.get('n_components', 5)}),
            'knn': (knnImputed, {'n_neighbors': params.get('n_neighbors', 2)}),
            'mice_bayesian': (miceBayesianRidgeImputed, {'max_iter': params.get('max_iter', 20)}),
            'mice_linear': (miceLinearRegressionImputed, {'max_iter': params.get('max_iter', 20)})
        }

        if method not in imputation_methods:
            return jsonify({'error': 'Unknown imputation method'})

        method_func, method_params = imputation_methods[method]
        
        df_to_impute = df_scaled if method in ['miss_forest', 'svd', 'knn', 'mice_bayesian', 'mice_linear'] else df

        imputed_df = _apply_processing_method(method, method_func, df_to_impute, method_params)

        if method in ['miss_forest', 'svd', 'knn', 'mice_bayesian', 'mice_linear']:
            imputed_df = postprocess_imputation(imputed_df, df)
        
        # Store results
        history_key = f'df_history_{len(history_paths)}'
        data_manager.save_dataframe(imputed_df, history_key, 'df_history')
        history_paths.append(history_key)
        session['df_history_paths'] = history_paths
        
        # Calculate imputed mask: where was it null before and now it's not null
        imputed_mask = df.isnull() & ~imputed_df.isnull()
        session['imputed_mask'] = imputed_mask

        # Update processing steps
        session['processing_steps'].append({'icon': 'fa-magic', 'color': 'text-primary', 'message': f'Applied {method} imputation.'})
        session.modified = True # Mark session as modified
        session['imputation_performed'] = True # Set flag that imputation has occurred
        
        # Generate and return the updated heatmap data
        updated_heatmap = create_heatmap_BW(
            imputed_df,
            title='Missing Values Distribution (Imputed Highlighted)',
            imputed=session.get('imputation_performed', False),
            null_mask=session.get('imputed_mask')
        )
        df_before=None
        if len(history_paths) > 1:
            df_before = data_manager.load_dataframe(history_paths[-2])
        else:
            df_before = data_manager.load_dataframe(history_paths[0])
        
        intensityComparisonPlot = create_intensity_comparison_plot(
            original_df=df_before,
            imputed_df=imputed_df,
            log_transform=True
        )

        new_shape = imputed_df.shape

        # Clean up DataFrames from memory
        del df
        del imputed_df
        if 'df_scaled' in locals():
            del df_scaled
        if df_before is not None:
            del df_before

        return jsonify({
            'success': True,
            'message': f'Imputation with {method} completed successfully',
            'steps': session['processing_steps'],
            'new_shape': new_shape,
            'missing_heatmap': updated_heatmap,
            'intensity_comparison_plot': intensityComparisonPlot
        })
        
    except Exception as e:
        # logging.error(f"Imputation failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Imputation failed: {str(e)}'})

@processing_bp.route('/replace_zeros', methods=['POST'])
def replace_zeros():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No data available to process.'})

    df_current = data_manager.load_dataframe(history_paths[-1])
    df_cleaned = df_current.replace(0, np.nan)

    history_key = f'df_history_{len(history_paths)}'
    data_manager.save_dataframe(df_cleaned, history_key, 'df_history')
    history_paths.append(history_key)
    session['df_history_paths'] = history_paths
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
    df_before=None
    if len(history_paths) > 1:
        df_before = data_manager.load_dataframe(history_paths[-2])
    else:
        df_before = data_manager.load_dataframe(history_paths[0])
    intensityComparisonPlot = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_cleaned,
        log_transform=True
    )

    # Get shape before deleting
    new_shape = df_cleaned.shape

    # Clean up DataFrames from memory
    del df_current
    del df_cleaned
    if df_before is not None:
        del df_before

    return jsonify({
        'success': True,
        'message': 'All zero values have been replaced with NaN.',
        'new_shape': new_shape,
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap,
        "intensity_comparison_plot":intensityComparisonPlot
    })

@processing_bp.route('/normalization')
def normalization():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = data_manager.load_dataframe(history_paths[-1])
    df_before = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'), 'Normalization')
    original_df = data_manager.load_dataframe(history_paths[0])
    return render_template('normalization.html',
                           original_shape=original_df.shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_normalization', methods=['POST'])
def apply_normalization():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    df = data_manager.load_dataframe(history_paths[-1])

    try:
        normalization_methods = {
            'tic': tic_normalization,
            'mtic': mtic_normalization,
            'median': median_normalization,
            'quantile': quantile_normalization,
            'pqn': pqn_normalization
        }

        if method not in normalization_methods:
            return jsonify({'error': 'Unknown normalization method'})

        normalized_df = _apply_processing_method(method, normalization_methods[method], df, {})

        history_key = f'df_history_{len(history_paths)}'
        data_manager.save_dataframe(normalized_df, history_key, 'df_history')
        history_paths.append(history_key)
        session['df_history_paths'] = history_paths
        session['processing_steps'].append({
            'icon': 'fa-chart-bar',
            'color': 'text-success',
            'message': f'Applied {method.upper()} normalization.'
        })
        session.modified = True

        df_before = data_manager.load_dataframe(history_paths[-2])
        plots = get_comparison_plots(df_before, normalized_df, 'boxplot', session.get('group_vector'), session.get('group_names'), 'Normalization')

        # Get shape before deleting
        new_shape = normalized_df.shape

        # Clean up DataFrames from memory
        del df
        del normalized_df
        del df_before

        return jsonify({
            'success': True,
            'message': f'{method.upper()} normalization applied successfully.',
            'new_shape': new_shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Normalization failed: {str(e)}'})

@processing_bp.route('/transformation')
def transformation():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = data_manager.load_dataframe(history_paths[-1])
    df_before = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'), 'Transformation')
    original_df = data_manager.load_dataframe(history_paths[0])
    return render_template('transformation.html',
                           original_shape=original_df.shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_transformation', methods=['POST'])
def apply_transformation():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = data_manager.load_dataframe(history_paths[-1])

    try:
        transformation_methods = {
            'log2': (log2_transform, {'pseudo_count': params.get('pseudo_count')}),
            'log10': (log10_transform, {'pseudo_count': params.get('pseudo_count')}),
            'sqrt': (sqrt_transform, {}),
            'cube_root': (cube_root_transform, {}),
            'arcsinh': (arcsinh_transform, {'cofactor': params.get('cofactor', 5)}),
            'glog': (glog_transform, {'lamb': params.get('lamb')}),
            'yeo_johnson': (yeo_johnson_transform, {})
        }

        if method not in transformation_methods:
            return jsonify({'error': 'Unknown transformation method'})

        method_func, method_params = transformation_methods[method]
        transformed_df = _apply_processing_method(method, method_func, df, method_params)

        history_key = f'df_history_{len(history_paths)}'
        data_manager.save_dataframe(transformed_df, history_key, 'df_history')
        history_paths.append(history_key)
        session['df_history_paths'] = history_paths
        session['processing_steps'].append({
            'icon': 'fa-exchange-alt',
            'color': 'text-info',
            'message': f'Applied {method.replace("_", " ").title()} transformation.'
        })
        session.modified = True
        
        df_before = data_manager.load_dataframe(history_paths[-2])
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_comparison_plots(df_before, transformed_df, plot_type, session.get('group_vector'), session.get('group_names'), 'Transformation')

        # Get shape before deleting
        new_shape = transformed_df.shape

        # Clean up DataFrames from memory
        del df
        del transformed_df
        del df_before

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} transformation applied successfully.',
            'new_shape': new_shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Transformation failed: {str(e)}'})

@processing_bp.route('/scaling')
def scaling():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = data_manager.load_dataframe(history_paths[-1])
    df_before = data_manager.load_dataframe(history_paths[-2]) if len(history_paths) > 1 else data_manager.load_dataframe(history_paths[0])

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'), 'Scaling')
    original_df = data_manager.load_dataframe(history_paths[0])
    return render_template('scaling.html',
                           original_shape=original_df.shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_scaling', methods=['POST'])
def apply_scaling():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = data_manager.load_dataframe(history_paths[-1])

    try:
        scaling_methods = {
            'standard': (standard_scaling, {'with_mean': params.get('with_mean', True), 'with_std': params.get('with_std', True)}),
            'minmax': (minmax_scaling, {'feature_range': tuple(params.get('feature_range', [0, 1]))}),
            'pareto': (pareto_scaling, {}),
            'range': (range_scaling, {}),
            'robust': (robust_scaling, {}),
            'vast': (vast_scaling, {})
        }

        if method not in scaling_methods:
            return jsonify({'error': 'Unknown scaling method'})

        method_func, method_params = scaling_methods[method]
        scaled_df = _apply_processing_method(method, method_func, df, method_params)

        history_key = f'df_history_{len(history_paths)}'
        data_manager.save_dataframe(scaled_df, history_key, 'df_history')
        history_paths.append(history_key)
        session['df_history_paths'] = history_paths
        session['processing_steps'].append({
            'icon': 'fa-compress-arrows-alt',
            'color': 'text-warning',
            'message': f'Applied {method.replace("_", " ").title()} scaling.'
        })
        session.modified = True
        
        df_before = data_manager.load_dataframe(history_paths[-2])
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_comparison_plots(df_before, scaled_df, plot_type, session.get('group_vector'), session.get('group_names'), 'Scaling')

        # Get shape before deleting
        new_shape = scaled_df.shape

        # Clean up DataFrames from memory
        del df
        del scaled_df
        del df_before

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} scaling applied successfully.',
            'new_shape': new_shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Scaling failed: {str(e)}'})

@processing_bp.route('/reset_sample_step', methods=['POST'])
def reset_sample_step():
    history_paths = session.get('df_history_paths', [])
    if not history_paths:
        return jsonify({'error': 'No sample data available to reset'})

    context = request.json.get('context', 'imputation') # Default to imputation for safety

    if len(history_paths) > 1:
        key_to_delete = history_paths.pop()
        data_manager.delete_dataframe(key_to_delete)
        if session.get('processing_steps'):
            session['processing_steps'].pop()

    session.modified = True

    df_current = data_manager.load_dataframe(history_paths[-1])
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
        df_before=None
        if len(history_paths) > 1:
            df_before = data_manager.load_dataframe(history_paths[-2])
        else:
            df_before = data_manager.load_dataframe(history_paths[0])
        response_data['intensity_comparison_plot'] = create_intensity_comparison_plot(
            original_df=df_before,
            imputed_df=df_current,
            log_transform=True
        )
    elif context in ['normalization', 'transformation', 'scaling']:
        df_before=None
        if len(history_paths) > 1:
            df_before = data_manager.load_dataframe(history_paths[-2])
        else:
            df_before = data_manager.load_dataframe(history_paths[0])
        response_data['plots'] = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'), context.title())

    return jsonify(response_data)