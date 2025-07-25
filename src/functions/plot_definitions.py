import json
# import numpy as np
import pandas as pd
import plotly.graph_objs as go # type: ignore
import plotly.utils # type: ignore
from sklearn.decomposition import PCA # type: ignore


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

def create_bar_plot(x, y, title, xaxis_title, yaxis_title):
    if not x or not y:
        return _create_empty_plot_with_message(title)

    fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color='#440154')])
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

def create_boxplot(df, title):
    if df.empty:
        return _create_empty_plot_with_message(title)

    data = []
    for col in df.columns:
        # Boxplot for the column
        data.append(go.Box(
            y=df[col].tolist(), # Convert pandas Series to a list
            name=col,
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
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_violinplot(df, title):
    if df.empty:
        return _create_empty_plot_with_message(title)

    data = []
    for col in df.columns:
        data.append(go.Violin(
            y=df[col].tolist(),
            name=col,
            box_visible=True,
            meanline_visible=True,
            #points='all',
            #jitter=0.3,
            #pointpos=-1.8,
            marker_color='#440154',
            line_color='#440154',
            hoverinfo='y',
            showlegend=False
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        yaxis_title='Value',
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_pca_plot(df, title):
    # Handle potential NaN values before PCA
    if df.empty or df.isnull().values.all():
        return _create_empty_plot_with_message(title, "Not enough data for PCA")

    if df.isnull().values.any():
        # Impute with mean for visualization purposes
        df_for_pca = df.T.fillna(df.T.mean())
        if df_for_pca.isnull().values.any(): # If NaNs still exist (e.g., all-NaN columns)
            return _create_empty_plot_with_message(title, "Not enough data for PCA after imputation")
    else:
        df_for_pca = df.T

    # Ensure there are enough components for PCA
    n_samples, n_features = df_for_pca.shape
    if n_samples < 2 or n_features < 2:
        return _create_empty_plot_with_message(title, "Not enough data for PCA")

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_for_pca)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.columns)

    fig = go.Figure(data=go.Scatter(
        x=pca_df['PC1'],
        y=pca_df['PC2'],
        mode='markers',
        text=pca_df.index,
        marker=dict(
            size=8,
            color='#440154',
            showscale=False
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
        template='plotly_white'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
