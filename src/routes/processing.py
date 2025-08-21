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

processing_bp = Blueprint('processing', __name__, template_folder='../../templates')

# Helper functions for plotting (could be moved to a helper module)
def get_comparison_plots(df_before, df_after, plot_type, group_vector, group_names):
    plots = {}
    plot_func = create_boxplot if plot_type == 'boxplot' else create_violinplot
    plots['dist_before'] = plot_func(df_before, 'Before Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_before'] = create_pca_plot(df_before, 'PCA Before Processing', group_vector=group_vector, group_names=group_names)
    plots['dist_after'] = plot_func(df_after, 'After Processing', group_vector=group_vector, group_names=group_names)
    plots['pca_after'] = create_pca_plot(df_after, 'PCA After Processing', group_vector=group_vector, group_names=group_names)
    return plots

@processing_bp.route('/imputation')
def imputation():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
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
    df_before=None
    if len(session.get('df_history', [])) > 1:
        df_before = session['df_history'][-2]
    else:
        df_before = session['df_history'][0]

    plots['intensity_comparison_plot'] = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_sample,
        log_transform=True
    )

    return render_template('imputation.html',
                         original_shape=session['df_sample'].shape,
                         current_shape=df_sample.shape,
                         processing_steps=session['processing_steps'],
                         plots=plots)

@processing_bp.route('/threshold', methods=['POST'])
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
    df_before=None
    if len(session.get('df_history', [])) > 1:
        df_before = session['df_history'][-2]
    else:
        df_before = session['df_history'][0]
    intensityComparisonPlot = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_thresholded,
        log_transform=True
    )

    return jsonify({
        'success': True,
        'original_shape': session['df_sample'].shape,
        'new_shape': df_thresholded.shape,
        'message': f'Thresholding applied with {threshold_percent}%',
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap,
        "intensity_comparison_plot":intensityComparisonPlot
    })

@processing_bp.route('/apply_imputation', methods=['POST'])
def apply_imputation():
    if not session.get('df_history'):
        return jsonify({'error': 'No thresholded sample data available'})
    
    method = request.json.get('method')
    params = request.json.get('params', {})
    
    df = session['df_history'][-1]

    try:
        # Apply scaling for advanced methods
        df_scaled = (df - df.mean()) / df.std()
        
        if method == 'n_imputation':
            n_val = params.get('n_value', 0)
            imputed_df = nImputed(df, n=n_val)
        elif method == 'half_minimum':
            imputed_df = halfMinimumImputed(df)
        elif method == 'mean':
            imputed_df = meanImputed(df)
        elif method == 'median':
            imputed_df = medianImputed(df)
        elif method == 'miss_forest':
            max_iter = params.get('max_iter', 10)
            n_estimators = params.get('n_estimators', 100)
            imputed_df = missForestImputed(df_scaled, max_iter=max_iter, n_estimators=n_estimators)
            imputed_df = postprocess_imputation(imputed_df, df)
        elif method == 'svd':
            n_components = params.get('n_components', 5)
            imputed_df = svdImputed(df_scaled, n_components=n_components)
            imputed_df = postprocess_imputation(imputed_df, df)
        elif method == 'knn':
            n_neighbors = params.get('n_neighbors', 2)
            imputed_df = knnImputed(df_scaled, n_neighbors=n_neighbors)
            imputed_df = postprocess_imputation(imputed_df, df)
        elif method == 'mice_bayesian':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceBayesianRidgeImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df)
        elif method == 'mice_linear':
            max_iter = params.get('max_iter', 20)
            imputed_df = miceLinearRegressionImputed(df_scaled, max_iter=max_iter)
            imputed_df = postprocess_imputation(imputed_df, df)
        else:
            return jsonify({'error': 'Unknown imputation method'})
        
        # Store results
        session['df_history'].append(imputed_df)
        
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
            imputed=session['imputation_performed'],
            null_mask=session.get('imputed_mask')
        )
        df_before=None
        if len(session.get('df_history', [])) > 1:
            df_before = session['df_history'][-2]
        else:
            df_before = session['df_history'][0]
        
        intensityComparisonPlot = create_intensity_comparison_plot(
            original_df=df_before,
            imputed_df=imputed_df,
            log_transform=True
        )

        return jsonify({
            'success': True,
            'message': f'Imputation with {method} completed successfully',
            'steps': session['processing_steps'],
            'new_shape': imputed_df.shape,
            'missing_heatmap': updated_heatmap,
            'intensity_comparison_plot': intensityComparisonPlot
        })
        
    except Exception as e:
        # logging.error(f"Imputation failed for method {method}: {e}", exc_info=True)
        return jsonify({'error': f'Imputation failed: {str(e)}'})

@processing_bp.route('/replace_zeros', methods=['POST'])
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
    df_before=None
    if len(session.get('df_history', [])) > 1:
        df_before = session['df_history'][-2]
    else:
        df_before = session['df_history'][0]
    intensityComparisonPlot = create_intensity_comparison_plot(
        original_df=df_before,
        imputed_df=df_cleaned,
        log_transform=True
    )

    return jsonify({
        'success': True,
        'message': 'All zero values have been replaced with NaN.',
        'new_shape': df_cleaned.shape,
        'steps': session['processing_steps'],
        'missing_heatmap': updated_heatmap,
        "intensity_comparison_plot":intensityComparisonPlot
    })

@processing_bp.route('/normalization')
def normalization():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][-2] if len(session.get('df_history', [])) > 1 else session['df_history'][0]

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'))

    return render_template('normalization.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_normalization', methods=['POST'])
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

        df_before = session['df_history'][-2]
        plots = get_comparison_plots(df_before, normalized_df, 'boxplot', session.get('group_vector'), session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.upper()} normalization applied successfully.',
            'new_shape': normalized_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Normalization failed: {str(e)}'})

@processing_bp.route('/transformation')
def transformation():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][-2] if len(session.get('df_history', [])) > 1 else session['df_history'][0]

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'))

    return render_template('transformation.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_transformation', methods=['POST'])
def apply_transformation():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = session['df_history'][-1]

    try:
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
        
        df_before = session['df_history'][-2]
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_comparison_plots(df_before, transformed_df, plot_type, session.get('group_vector'), session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} transformation applied successfully.',
            'new_shape': transformed_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Transformation failed: {str(e)}'})

@processing_bp.route('/scaling')
def scaling():
    if not session.get('df_history'):
        flash('Please define sample data first')
        return redirect(url_for('analysis.metadata'))
    
    df_current = session['df_history'][-1]
    df_before = session['df_history'][-2] if len(session.get('df_history', [])) > 1 else session['df_history'][0]

    if 'processing_steps' not in session or not session['processing_steps']:
        session['processing_steps'] = [{'icon': 'fa-check-circle', 'color': 'text-success', 'message': 'Sample data loaded, ready for processing.'}]

    plots = get_comparison_plots(df_before, df_current, 'boxplot', session.get('group_vector'), session.get('group_names'))

    return render_template('scaling.html',
                           original_shape=session['df_sample'].shape,
                           current_shape=df_current.shape,
                           processing_steps=session.get('processing_steps', []),
                           plots=plots
                          )

@processing_bp.route('/apply_scaling', methods=['POST'])
def apply_scaling():
    if not session.get('df_history'):
        return jsonify({'error': 'No sample data available'})

    method = request.json.get('method')
    params = request.json.get('params', {})
    df = session['df_history'][-1]

    try:
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
        
        df_before = session['df_history'][-2]
        plot_type = request.json.get('plot_type', 'boxplot')
        plots = get_comparison_plots(df_before, scaled_df, plot_type, session.get('group_vector'), session.get('group_names'))

        return jsonify({
            'success': True,
            'message': f'{method.replace("_", " ").title()} scaling applied successfully.',
            'new_shape': scaled_df.shape,
            'steps': session['processing_steps'],
            'plots': plots
        })
    except Exception as e:
        return jsonify({'error': f'Scaling failed: {str(e)}'})

@processing_bp.route('/reset_sample_step', methods=['POST'])
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
        df_before=None
        if len(session.get('df_history', [])) > 1:
            df_before = session['df_history'][-2]
        else:
            df_before = session['df_history'][0]
        response_data['intensity_comparison_plot'] = create_intensity_comparison_plot(
            original_df=df_before,
            imputed_df=df_current,
            log_transform=True
        )
    elif context == 'normalization':
        df_before=None
        if len(session.get('df_history', [])) > 1:
            df_before = session['df_history'][-2]
        else:
            df_before = session['df_history'][0]
        response_data['plots'] = get_normalization_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))
    elif context == 'transformation':
        df_before=None
        if len(session.get('df_history', [])) > 1:
            df_before = session['df_history'][-2]
        else:
            df_before = session['df_history'][0]
        response_data['plots'] = get_transformation_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))
    elif context == 'scaling':
        df_before=None
        if len(session.get('df_history', [])) > 1:
            df_before = session['df_history'][-2]
        else:
            df_before = session['df_history'][0]
        response_data['plots'] = get_scaling_plots(df_before, df_current, group_vector=session.get('group_vector'), group_names=session.get('group_names'))

    return jsonify(response_data)

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