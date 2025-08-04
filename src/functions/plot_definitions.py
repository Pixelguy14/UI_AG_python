import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go # type: ignore
import plotly.utils # type: ignore
from sklearn.decomposition import PCA # type: ignore
import plotly.express as px # type: ignore
from scipy.cluster.hierarchy import linkage, dendrogram # type: ignore
from sklearn.cross_decomposition import PLSRegression # type: ignore
from plotly.subplots import make_subplots # type: ignore

def _create_empty_plot_with_message(title, message="No data available for plotting."):
    fig = go.Figure()
    fig.update_layout(
        title=f"{title}<br><sup>{message}</sup>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text=message,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20, color="gray")
            )
        ],
        template='plotly_white'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_bar_plot(x, y, title, xaxis_title, yaxis_title, group_vector=None, group_names=None):
    if not x or not y:
        return _create_empty_plot_with_message(title)

    colors = px.colors.qualitative.Vivid
    group_color_map = {}
    bar_colors = []

    if group_vector and group_names and isinstance(x, list) and len(x) > 0:
        unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
        for i, group_id in enumerate(unique_group_ids):
            group_color_map[group_id] = colors[i % len(colors)]
        
        for col_name in x:
            color = '#808080' # default grey
            if col_name in group_vector and group_vector[col_name]['groups']:
                first_group_id = str(group_vector[col_name]['groups'][0])
                if first_group_id in group_color_map:
                    color = group_color_map[first_group_id]
            bar_colors.append(color)
    else:
        bar_colors = '#440154' # Original color

    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color=bar_colors if bar_colors else '#440154')])
    
    # Add custom legend
    if group_names and group_color_map:
        for group_id, group_name in group_names.items():
            if group_id != '0' and group_id in group_color_map:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=group_color_map[group_id]),
                    legendgroup=group_name,
                    showlegend=True,
                    name=group_name
                ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template='plotly_white'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_heatmap(data, title):
    if data.empty:
        return _create_empty_plot_with_message(title)

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

def create_heatmap_BW(data, title, imputed=False, null_mask=None):
    if data.empty:
        return _create_empty_plot_with_message(title)

    x_data = data.columns.tolist() if hasattr(data, 'columns') else None
    y_data = data.index.tolist() if hasattr(data, 'index') else None

    if imputed and null_mask is not None and not null_mask.empty:
        # New behavior for imputed data
        # 0 for original, 1 for imputed, 2 for still-null
        z_data = pd.DataFrame(0, index=data.index, columns=data.columns)
        z_data[null_mask] = 1 # Imputed values
        z_data[data.isnull()] = 2 # Still null values
        
        # Colorscale: 0 -> deep blue, 1 -> yellow, 2 -> white
        colorscale = [
            [0.0, 'rgb(68, 1, 84)'],   # Original data (maps to value 0)
            [0.5, 'rgb(253, 231, 37)'], # Imputed data (maps to value 1)
            [1.0, 'white']             # Still null (maps to value 2)
        ]
        zmin, zmax = 0, 2
        
    else:
        # Original behavior: 0 for non-null, 1 for null
        z_data = data.isnull().astype(int)
        colorscale = [[0, 'rgb(68, 1, 84)'], [1, 'white']] # Deep blue for data, white for null
        zmin, zmax = 0, 1


    fig = go.Figure(data=go.Heatmap(
        z=z_data.values.tolist(),
        x=x_data,
        y=y_data,
        colorscale=colorscale,
        showscale=False,
        zmin=zmin,
        zmax=zmax
    ))
    
    fig.update_layout(title=title, template='plotly_white')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_pie_chart(labels, values, title):
    if not labels or not values or sum(values) == 0:
        return _create_empty_plot_with_message(title)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title, template='plotly_white')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_distribution_plot(data_series, column_name):
    if data_series.empty:
        return _create_empty_plot_with_message(f'Distribution of {column_name}')

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
    if data_series.empty:
        return _create_empty_plot_with_message(f'Density Plot of {column_name}')

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

def create_boxplot(df, title, group_vector=None, group_names=None):
    if df.empty:
        return _create_empty_plot_with_message(title)

    data = []
    colors = px.colors.qualitative.Vivid
    group_color_map = {}

    if group_vector and group_names:
        unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
        for i, group_id in enumerate(unique_group_ids):
            group_color_map[group_id] = colors[i % len(colors)]

    for col in df.columns:
        color = '#808080'  # Default grey for no group
        if group_vector and col in group_vector and group_vector[col]['groups']:
            first_group_id = str(group_vector[col]['groups'][0])
            if first_group_id in group_color_map:
                color = group_color_map[first_group_id]

        data.append(go.Box(
            y=df[col].tolist(),
            name=col,
            marker_color=color,
            line_color=color,
            hoverinfo='y',
            showlegend=False
        ))

    fig = go.Figure(data=data)

    # Add custom legend
    if group_names and group_color_map:
        for group_id, group_name in group_names.items():
            if group_id != '0' and group_id in group_color_map:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=group_color_map[group_id]),
                    legendgroup=group_name,
                    showlegend=True,
                    name=group_name
                ))

    fig.update_layout(
        title=title,
        yaxis_title='Value',
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_violinplot(df, title, group_vector=None, group_names=None):
    if df.empty:
        return _create_empty_plot_with_message(title)

    data = []
    colors = px.colors.qualitative.Vivid
    group_color_map = {}

    if group_vector and group_names:
        unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
        for i, group_id in enumerate(unique_group_ids):
            group_color_map[group_id] = colors[i % len(colors)]

    for col in df.columns:
        color = '#808080'  # Default grey for no group
        if group_vector and col in group_vector and group_vector[col]['groups']:
            first_group_id = str(group_vector[col]['groups'][0])
            if first_group_id in group_color_map:
                color = group_color_map[first_group_id]

        data.append(go.Violin(
            y=df[col].tolist(),
            name=col,
            box_visible=True,
            meanline_visible=True,
            #points='all',
            #jitter=0.3,
            #pointpos=-1.8,
            marker_color=color,
            line_color=color,
            hoverinfo='y',
            showlegend=False
        ))

    fig = go.Figure(data=data)

    # Add custom legend
    if group_names and group_color_map:
        for group_id, group_name in group_names.items():
            if group_id != '0' and group_id in group_color_map:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=group_color_map[group_id]),
                    legendgroup=group_name,
                    showlegend=True,
                    name=group_name
                ))

    fig.update_layout(
        title=title,
        yaxis_title='Value',
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_pca_plot(df, title, group_vector=None, group_names=None):
    # Handle potential NaN values before PCA
    if df.empty or df.isnull().values.all():
        return _create_empty_plot_with_message(title, "Not enough data for PCA")

    # Select only numeric columns for PCA
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return _create_empty_plot_with_message(title, "No numeric data available for PCA.")

    if numeric_df.isnull().values.any():
        # Impute with mean for visualization purposes
        df_for_pca = numeric_df.T.fillna(numeric_df.T.mean())
        if df_for_pca.isnull().values.any(): # If NaNs still exist
            return _create_empty_plot_with_message(title, "Not enough data for PCA after imputation")
    else:
        df_for_pca = numeric_df.T

    # Ensure there are enough components for PCA
    n_samples, n_features = df_for_pca.shape
    if n_samples < 2 or n_features < 2:
        return _create_empty_plot_with_message(title, "Not enough data for PCA")

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_for_pca)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=numeric_df.columns)

    fig = go.Figure()

    if group_vector and group_names:
        colors = px.colors.qualitative.Vivid
        unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
        group_color_map = {gid: colors[i % len(colors)] for i, gid in enumerate(unique_group_ids)}
        
        sample_to_group_id = {}
        for sample_name in numeric_df.columns:
            if sample_name in group_vector and group_vector.get(sample_name, {}).get('groups'):
                group_id = str(group_vector[sample_name]['groups'][0])
                sample_to_group_id[sample_name] = group_id
            else:
                sample_to_group_id[sample_name] = '0'
        
        pca_df['group_id'] = pca_df.index.map(sample_to_group_id)
        
        for group_id in sorted(pca_df['group_id'].unique()):
            group_df = pca_df[pca_df['group_id'] == group_id]
            group_name = group_names.get(str(group_id), 'No Group' if str(group_id) == '0' else f'Group {group_id}')

            fig.add_trace(go.Scatter(
                x=group_df['PC1'].tolist(),
                y=group_df['PC2'].tolist(),
                mode='markers',
                text=group_df.index.tolist(),
                name=group_name,
                marker=dict(
                    size=8,
                    color=group_color_map.get(str(group_id), '#808080')
                )
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pca_df['PC1'].tolist(),
            y=pca_df['PC2'].tolist(),
            mode='markers',
            text=pca_df.index.tolist(),
            marker=dict(size=8, color='#440154', showscale=False)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Hierarchical Clustering Analysis 
def create_hca_plot(df, title, group_vector=None, group_names=None, distance_metric='euclidean', linkage_method='average'):
    try:
        # Select only numeric columns and transpose for clustering
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return _create_empty_plot_with_message(title, "Not enough data for HCA")

        # Impute missing values for clustering
        if numeric_df.isnull().values.any():
            df_for_hca = numeric_df.T.fillna(numeric_df.T.mean())
            if df_for_hca.isnull().values.any():
                return _create_empty_plot_with_message(title, "Not enough data for HCA after imputation")
        else:
            df_for_hca = numeric_df.T
        
        # Perform hierarchical clustering
        linked = linkage(df_for_hca.values, method=linkage_method, metric=distance_metric)
        
        # Generate dendrogram data
        dendro_data = dendrogram(linked, no_plot=True, labels=df_for_hca.index.tolist())
        
        # Create color mapping for groups
        group_color_map = {}
        if group_vector and group_names:
            colors = px.colors.qualitative.Vivid
            unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
            group_color_map = {gid: colors[i % len(colors)] for i, gid in enumerate(unique_group_ids)}
        
        # Create mapping from leaf label to color
        leaf_color_map = {}
        for sample_label in dendro_data['ivl']:
            color = '#808080'  # default grey
            if group_vector and sample_label in group_vector and group_vector[sample_label].get('groups'):
                first_group_id = str(group_vector[sample_label]['groups'][0])
                if first_group_id in group_color_map:
                    color = group_color_map[first_group_id]
            leaf_color_map[sample_label] = color

        fig = go.Figure()

        # Create mapping from leaf label to its position
        leaf_positions = {leaf: idx * 10 + 5 for idx, leaf in enumerate(dendro_data['ivl'])}

        # Add dendrogram traces with coloring
        for i, (icoord, dcoord) in enumerate(zip(dendro_data['icoord'], dendro_data['dcoord'])):
            # Determine color for this segment
            segment_color = '#808080'  # default gray
            
            # Check if this segment is connected to a leaf
            leaf_positions_in_segment = set()
            for coord in icoord:
                for leaf, pos in leaf_positions.items():
                    if abs(coord - pos) < 1e-5:  # floating point comparison tolerance
                        leaf_positions_in_segment.add(leaf)
            
            # If connected to a leaf, use its color
            if leaf_positions_in_segment:
                leaf_label = next(iter(leaf_positions_in_segment))
                segment_color = leaf_color_map.get(leaf_label, '#808080')

            fig.add_trace(go.Scatter(
                x=icoord,
                y=dcoord,
                mode='lines',
                line=dict(color=segment_color, width=2),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Prepare colored x-axis labels
        tickvals = []
        ticktext = []
        for i, sample_label in enumerate(dendro_data['ivl']):
            tickvals.append(i * 10 + 5)
            color = leaf_color_map.get(sample_label, '#808080')
            ticktext.append(f'<span style="color:{color};">{sample_label}</span>')

        # Add custom legend for groups
        if group_vector and group_names:
            for group_id, group_name in group_names.items():
                if group_id != '0' and group_id in group_color_map:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode='markers',
                        marker=dict(size=10, color=group_color_map[group_id]),
                        legendgroup=group_name,
                        showlegend=True,
                        name=group_name
                    ))

        # Format distance metric for display
        metric_display = distance_metric.capitalize()
        if metric_display == 'Correlation':
            metric_display = '1 - Correlation'
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            autosize=True,
            # showlegend=bool(group_vector and group_names),
            showlegend=True,
            hovermode='closest',
            # margin=dict(l=50, r=20, t=60, b=150),
            xaxis=dict(
                title="Samples",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title=f"Distance ({metric_display})"
            )
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error in create_hca_plot: {e}")
        return _create_empty_plot_with_message(title, f"Error generating HCA plot: {e}")

def create_plsda_plot(df, title, group_vector=None, group_names=None):
    if df.empty or df.isnull().values.all():
        return _create_empty_plot_with_message(title, "Not enough data for PLS-DA")

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return _create_empty_plot_with_message(title, "No numeric data available for PLS-DA.")

    if numeric_df.isnull().values.any():
        df_for_plsda = numeric_df.T.fillna(numeric_df.T.mean())
        if df_for_plsda.isnull().values.any():
            return _create_empty_plot_with_message(title, "Not enough data for PLS-DA after imputation")
    else:
        df_for_plsda = numeric_df.T

    if not group_vector or not group_names:
        return _create_empty_plot_with_message(title, "PLS-DA requires group information.")

    sample_to_group_id = {}
    for sample_name in numeric_df.columns:
        if sample_name in group_vector and group_vector.get(sample_name, {}).get('groups'):
            group_id = str(group_vector[sample_name]['groups'][0])
            sample_to_group_id[sample_name] = group_id
        else:
            sample_to_group_id[sample_name] = '0'

    y = np.array([int(sample_to_group_id.get(s, 0)) for s in df_for_plsda.index])
    
    if len(np.unique(y)) < 2:
        return _create_empty_plot_with_message(title, "PLS-DA requires at least two groups.")

    plsda = PLSRegression(n_components=2)
    plsda.fit(df_for_plsda.values, y)
    scores = plsda.transform(df_for_plsda.values)
    
    plsda_df = pd.DataFrame(data=scores, columns=['PC1', 'PC2'], index=numeric_df.columns)

    fig = go.Figure()

    colors = px.colors.qualitative.Vivid
    unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
    group_color_map = {gid: colors[i % len(colors)] for i, gid in enumerate(unique_group_ids)}
    
    plsda_df['group_id'] = plsda_df.index.map(sample_to_group_id)
    
    for group_id in sorted(plsda_df['group_id'].unique()):
        group_df = plsda_df[plsda_df['group_id'] == group_id]
        group_name = group_names.get(str(group_id), 'No Group' if str(group_id) == '0' else f'Group {group_id}')

        fig.add_trace(go.Scatter(
            x=group_df['PC1'].tolist(),
            y=group_df['PC2'].tolist(),
            mode='markers',
            text=group_df.index.tolist(),
            name=group_name,
            marker=dict(
                size=8,
                color=group_color_map.get(str(group_id), '#808080')
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_oplsda_plot(df, title, group_vector=None, group_names=None):
    if df.empty or df.isnull().values.all():
        return _create_empty_plot_with_message(title, "Not enough data for OPLS-DA")

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return _create_empty_plot_with_message(title, "No numeric data available for OPLS-DA.")

    if numeric_df.isnull().values.any():
        df_for_oplsda = numeric_df.T.fillna(numeric_df.T.mean())
        if df_for_oplsda.isnull().values.any():
            return _create_empty_plot_with_message(title, "Not enough data for OPLS-DA after imputation")
    else:
        df_for_oplsda = numeric_df.T

    if not group_vector or not group_names:
        return _create_empty_plot_with_message(title, "OPLS-DA requires group information.")

    sample_to_group_id = {}
    for sample_name in numeric_df.columns:
        if sample_name in group_vector and group_vector.get(sample_name, {}).get('groups'):
            group_id = str(group_vector[sample_name]['groups'][0])
            sample_to_group_id[sample_name] = group_id
        else:
            sample_to_group_id[sample_name] = '0'

    y = np.array([int(sample_to_group_id.get(s, 0)) for s in df_for_oplsda.index])

    if len(np.unique(y)) < 2:
        return _create_empty_plot_with_message(title, "OPLS-DA requires at least two groups.")

    X = df_for_oplsda.values
    X_centered = X - X.mean(axis=0)

    # 1. Predictive component from PLS
    pls = PLSRegression(n_components=1)
    pls.fit(X, y) 

    t_pred = pls.x_scores_
    p_pred = pls.x_loadings_
    
    # 2. Orthogonal components from PCA on residuals
    X_pred_recon = t_pred @ p_pred.T
    X_ortho = X_centered - X_pred_recon
    
    pca = PCA(n_components=1)
    t_ortho = pca.fit_transform(X_ortho)

    oplsda_df = pd.DataFrame(data={'t_pred': t_pred.flatten(), 't_ortho': t_ortho.flatten()}, index=numeric_df.columns)

    fig = go.Figure()

    colors = px.colors.qualitative.Vivid
    unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
    group_color_map = {gid: colors[i % len(colors)] for i, gid in enumerate(unique_group_ids)}
    
    oplsda_df['group_id'] = oplsda_df.index.map(sample_to_group_id)
    
    for group_id in sorted(oplsda_df['group_id'].unique()):
        group_df = oplsda_df[oplsda_df['group_id'] == group_id]
        group_name = group_names.get(str(group_id), 'No Group' if str(group_id) == '0' else f'Group {group_id}')

        fig.add_trace(go.Scatter(
            x=group_df['t_pred'].tolist(),
            y=group_df['t_ortho'].tolist(),
            mode='markers',
            text=group_df.index.tolist(),
            name=group_name,
            marker=dict(
                size=8,
                color=group_color_map.get(str(group_id), '#808080')
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Predictive Component",
        yaxis_title="Orthogonal Component",
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_volcano_plot(results_df, p_value_col='p_adj', log2fc_col='log2FC', p_thresh=0.05, fc_thresh=1.0):
    """
    Generates an interactive volcano plot from differential analysis results using plotly.graph_objects.
    """
    # Use p_value_col if it exists, otherwise default to 'p_value'
    if p_value_col not in results_df.columns:
        p_value_col = 'p_value'
    
    if p_value_col not in results_df.columns or log2fc_col not in results_df.columns:
        return _create_empty_plot_with_message('Volcano Plot', "Required columns (p_value/p_adj, log2FC) not found.")

    # Ensure numeric types and handle potential non-numeric values
    results_df[p_value_col] = pd.to_numeric(results_df[p_value_col], errors='coerce')
    results_df[log2fc_col] = pd.to_numeric(results_df[log2fc_col], errors='coerce')
    results_df = results_df.dropna(subset=[p_value_col, log2fc_col])

    if results_df.empty:
        return _create_empty_plot_with_message('Volcano Plot', "No valid data points after cleaning.")

    results_df['log_p'] = -np.log10(results_df[p_value_col])

    # Determine significance
    results_df['significant'] = (results_df[p_value_col] < p_thresh) & (abs(results_df[log2fc_col]) > fc_thresh)
    
    fig = go.Figure()

    # Add non-significant points
    fig.add_trace(go.Scatter(
        x=results_df[~results_df['significant']][log2fc_col].tolist(),
        y=results_df[~results_df['significant']]['log_p'].tolist(),
        mode='markers',
        marker=dict(color='gray', size=8),
        name='Not Significant',
        text=results_df[~results_df['significant']].index.tolist(),
        hoverinfo='text'
    ))

    # Add significant points
    fig.add_trace(go.Scatter(
        x=results_df[results_df['significant']][log2fc_col].tolist(),
        y=results_df[results_df['significant']]['log_p'].tolist(),
        mode='markers',
        marker=dict(color='red', size=8),
        name='Significant',
        text=results_df[results_df['significant']].index.tolist(),
        hoverinfo='text'
    ))

    # Add threshold lines
    fig.add_shape(
        type="line",
        x0=results_df[log2fc_col].min(),
        y0=-np.log10(p_thresh),
        x1=results_df[log2fc_col].max(),
        y1=-np.log10(p_thresh),
        line=dict(color="red", width=1, dash="dash"),
        name=f"P-value threshold ({p_thresh})"
    )
    fig.add_shape(
        type="line",
        x0=fc_thresh,
        y0=results_df['log_p'].min(),
        x1=fc_thresh,
        y1=results_df['log_p'].max(),
        line=dict(color="blue", width=1, dash="dash"),
        name=f"FC threshold ({fc_thresh})"
    )
    fig.add_shape(
        type="line",
        x0=-fc_thresh,
        y0=results_df['log_p'].min(),
        x1=-fc_thresh,
        y1=results_df['log_p'].max(),
        line=dict(color="blue", width=1, dash="dash"),
        name=f"FC threshold ({-fc_thresh})"
    )

    fig.update_layout(
        title='Volcano Plot',
        xaxis_title='Log2 Fold Change',
        yaxis_title='-Log10 P-value',
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_clustergram(data_df, results_df, top_n=50, p_value_col='p_adj', group_vector=None, group_names=None, distance_metric='euclidean', linkage_method='average'):
    """
    Generates a clustered heatmap (clustergram) of the top N significant features.
    """
    # Use p_value_col if it exists, otherwise default to 'p_value'
    if p_value_col not in results_df.columns:
        p_value_col = 'p_value'

    if p_value_col not in results_df.columns:
        return _create_empty_plot_with_message('Clustergram', "Required p_value column not found.")

    # Filter for significant features
    significant_features_df = results_df[results_df[p_value_col] < 0.05]
    if significant_features_df.empty:
        return _create_empty_plot_with_message('Clustergram', "No significant features found to generate a clustergram.")

    # Select top N features by p-value
    significant_features = significant_features_df.nsmallest(top_n, p_value_col).index
    
    # Prepare data for clustergram: samples as rows, features as columns
    # data_df has features as rows, samples as columns. So, transpose it.
    plot_data = data_df.loc[significant_features].T

    # Z-score normalization (per feature, which are now columns)
    # Avoid division by zero for features with zero standard deviation
    plot_data_zscored = plot_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0, axis=0)
    
    # Handle potential NaN values after z-scoring (e.g., if a feature was all zeros)
    plot_data_zscored = plot_data_zscored.fillna(0)

    if plot_data_zscored.empty:
        return _create_empty_plot_with_message('Clustergram', "No valid data after processing for clustergram.")

    # --- Hierarchical Clustering ---
    # Cluster rows (samples)
    row_linkage = linkage(plot_data_zscored.values, method=linkage_method, metric=distance_metric)
    row_dendro_data = dendrogram(row_linkage, no_plot=True, labels=plot_data_zscored.index.tolist())

    # Cluster columns (features)
    col_linkage = linkage(plot_data_zscored.T.values, method=linkage_method, metric=distance_metric)
    col_dendro_data = dendrogram(col_linkage, no_plot=True, labels=plot_data_zscored.columns.tolist())

    # Reorder data based on dendrograms
    plot_data_reordered = plot_data_zscored.iloc[row_dendro_data['leaves'], col_dendro_data['leaves']]

    print(f"\n--- Debugging Clustergram Data ---")
    print(f"plot_data_zscored shape: {plot_data_zscored.shape}")
    print(f"plot_data_reordered shape: {plot_data_reordered.shape}")
    print(f"row_dendro_data icoord min/max: {np.min(row_dendro_data['icoord'])}, {np.max(row_dendro_data['icoord'])}")
    print(f"row_dendro_data dcoord min/max: {np.min(row_dendro_data['dcoord'])}, {np.max(row_dendro_data['dcoord'])}")
    print(f"col_dendro_data icoord min/max: {np.min(col_dendro_data['icoord'])}, {np.max(col_dendro_data['icoord'])}")
    print(f"col_dendro_data dcoord min/max: {np.min(col_dendro_data['dcoord'])}, {np.max(col_dendro_data['dcoord'])}")
    print(f"--- End Debugging Clustergram Data ---\n")

    # --- Create Color Mapping for Row Dendrogram (Samples) ---
    group_color_map = {}
    if group_vector and group_names:
        colors = px.colors.qualitative.Vivid
        unique_group_ids = sorted([gid for gid in group_names.keys() if gid != '0'])
        group_color_map = {gid: colors[i % len(colors)] for i, gid in enumerate(unique_group_ids)}
    
    leaf_color_map = {}
    for sample_label in row_dendro_data['ivl']:
        color = '#808080'  # default grey
        if group_vector and sample_label in group_vector and group_vector[sample_label].get('groups'):
            first_group_id = str(group_vector[sample_label]['groups'][0])
            if first_group_id in group_color_map:
                color = group_color_map[first_group_id]
        leaf_color_map[sample_label] = color

    # Define subplot grid: 2 rows, 2 columns.
    # Top-left is empty, top-right is column dendrogram.
    # Bottom-left is row dendrogram, bottom-right is heatmap.
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.2, 0.8], # Row dendrogram width, Heatmap width
        row_heights=[0.2, 0.8],   # Column dendrogram height, Heatmap height
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        shared_xaxes=False,
        shared_yaxes=False,
        # Specs define the type of subplot and if it spans rows/cols
        specs=[
            [{"type": "xy", "rowspan": 1, "colspan": 1}, {"type": "xy", "rowspan": 1, "colspan": 1}],
            [{"type": "xy", "rowspan": 1, "colspan": 1}, {"type": "heatmap", "rowspan": 1, "colspan": 1}]
        ]
    )

    # Add Column Dendrogram
    for i, (icoord, dcoord) in enumerate(zip(col_dendro_data['icoord'], col_dendro_data['dcoord'])):
        fig.add_trace(go.Scatter(
            x=icoord,
            y=dcoord,
            mode='lines',
            line=dict(color='rgb(50,50,50)', width=2),
            hoverinfo='none',
            showlegend=False,
            xaxis='x2',  # Explicitly assign to top-right x-axis
            yaxis='y2'   # Explicitly assign to top-right y-axis
        ), row=1, col=2) # Top-right subplot

    # Add Row Dendrogram
    for i, (icoord, dcoord) in enumerate(zip(row_dendro_data['icoord'], row_dendro_data['dcoord'])):
        # Use a default color for all dendrogram lines for now
        segment_color = 'rgb(50,50,50)' 

        fig.add_trace(go.Scatter(
            x=dcoord, # x and y are swapped for horizontal dendrogram
            y=icoord,
            mode='lines',
            line=dict(color=segment_color, width=2),
            hoverinfo='none',
            showlegend=False,
            xaxis='x',  # Explicitly assign to bottom-left x-axis
            yaxis='y3'   # Explicitly assign to bottom-left y-axis
        ), row=2, col=1) # Bottom-left subplot

    # Add Heatmap
    fig.add_trace(go.Heatmap(
        z=plot_data_reordered.values,
        x=plot_data_reordered.columns.tolist(),
        y=plot_data_reordered.index.tolist(),
        colorscale='Viridis',
        colorbar=dict(title='Z-score'),
        hoverinfo='x+y+z'
    ), row=2, col=2) # Bottom-right subplot

    # --- Layout and Customization ---
    fig.update_layout(
        title=f'Clustergram of Top {len(significant_features)} Significant Features',
        template='plotly_white',
        autosize=True,
        height=800,
        width=1000,
        hovermode='closest',
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=100),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0, 0.2]), # Row dendrogram x-axis (distance)
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0.8, 1]), # Column dendrogram y-axis (distance)
        xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0.2, 1]), # Column dendrogram x-axis (feature labels)
        yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0.8, 1]), # Column dendrogram y-axis (distance)
        xaxis3=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0, 0.2]), # Row dendrogram x-axis (distance)
        yaxis3=dict(showgrid=False, zeroline=False, showticklabels=False, domain=[0, 0.8])  # Row dendrogram y-axis (sample labels)
    )

    # Update axes for dendrograms and heatmap
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=2) # Column dendrogram x-axis
    fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2) # Column dendrogram y-axis (distance)

    fig.update_xaxes(showgrid=False, zeroline=False, row=2, col=1) # Row dendrogram x-axis (distance)
    fig.update_yaxes(showgrid=False, zeroline=False, row=2, col=1) # Row dendrogram y-axis

    # Heatmap axes
    fig.update_xaxes(title_text="Features", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Samples", row=2, col=2)

    # Prepare colored y-axis labels for row dendrogram (samples)
    # This needs to be done after the heatmap is added to ensure correct axis mapping
    tickvals_row = []
    ticktext_row = []
    for i, sample_label in enumerate(row_dendro_data['ivl']):
        tickvals_row.append(i) # Use index as tick value
        color = leaf_color_map.get(sample_label, '#808080')
        ticktext_row.append(f'<span style="color:{color};">{sample_label}</span>')

    fig.update_yaxes(
        tickvals=[i for i in range(len(row_dendro_data['ivl']))],
        ticktext=[f'<span style="color:{leaf_color_map.get(label, "#808080")};">{label}</span>' for label in plot_data_reordered.index],
        row=2, col=2
    )

    # Prepare colored x-axis labels for column dendrogram (features)
    fig.update_xaxes(
        tickvals=[i for i in range(len(col_dendro_data['ivl']))],
        ticktext=plot_data_reordered.columns.tolist(),
        row=2, col=2
    )

    # Add custom legend for groups
    if group_names and group_color_map:
        for group_id, group_name in group_names.items():
            if group_id != '0' and group_id in group_color_map:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=group_color_map[group_id]),
                    legendgroup=group_name,
                    showlegend=True,
                    name=group_name
                ), row=1, col=1) # Add to an empty subplot or a corner to display legend

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)