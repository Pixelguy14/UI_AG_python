from tkinter.font import names
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import plotly.express as px

def random_forest_classification(df_log1p, params):
    X = df_log1p.drop(columns=['Sample', 'Group']).values
    y = df_log1p['Group'].values

    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=params.get('test_size', 0.3), stratify=y, random_state=42)
    class_weight = params.get('class_weight', 'balanced')
    if class_weight == 'none' or class_weight == 'None':
        class_weight = None

    pipe = Pipeline([
        ("rf", RandomForestClassifier(random_state=42, class_weight=class_weight))
    ])
    
    max_depth_default = params.get('max_depth_default', 'None')
    max_depth_list = []
    if max_depth_default == 'None' or max_depth_default is None:
        max_depth_list.append(None)
    else:
        try:
            max_depth_list.append(int(max_depth_default))
        except Exception:
            pass
    # Agrega los otros valores
    max_depth_list += [params.get('max_depth_min', 5), params.get('max_depth_max', 20)]
    # Filtra None y duplicados
    max_depth_list = list({d for d in max_depth_list if d is None or isinstance(d, int)})

    param_grid = {
        "rf__n_estimators": list({params.get('n_estimators_start', 100), params.get('n_estimators_end', 300)}),
        "rf__max_depth": max_depth_list,
        "rf__min_samples_leaf": list({params.get('min_samples_leaf_min', 1), params.get('min_samples_leaf_max', 3)}),
    }
    
    import numpy as np
    min_class_size = np.min(pd.Series(y_train).value_counts())
    n_splits = min(5, min_class_size)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    ############################################################
    print("Better AUC CV:", grid.best_score_)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("AUC Test:", roc_auc_score(y_test, y_proba))
    
    feat_importances = best_model.named_steps["rf"].feature_importances_
    feat_names = df_log1p.drop(columns=['Sample', 'Group']).columns
    top = pd.Series(feat_importances, index=feat_names).sort_values(ascending=False).head(20)
    print(top)
    
   
    est = grid.best_estimator_
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    score, perm_scores, pvalue = permutation_test_score(
        est,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_permutations=200,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print(f"AUC real (cv): {score:.3f}")
    print(f"P-value (perm): {pvalue:.3f}")
    
    
    import matplotlib.pyplot as plt
    # Change to Plotly histogram
    auc_plot = auc_plotly(perm_scores, score)
    

    import shap
    feature_cols = df_log1p.columns.drop(['Sample', 'Group'])
    names = feature_cols.tolist()
    
    print("X_test: ", X_test.shape)
    
    explainer = shap.TreeExplainer(best_model.named_steps["rf"])
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]  # clase "Enferma"
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    print("shap_values selected shape:", sv.shape)  # debe ser (20, 1757)
    shap_plot = summary_plot(df_log1p, sv, names, X_test)

    # 6. Dependence plot para el metabolito top (por ejemplo el primero en 'top')
    top_feature = top.index[4]
    top_feature_idx = names.index(top_feature)
    shap_dependence_plot = dependence_plot(sv, X_test, names, top_feature_idx, top_feature)


    # 1. Generar el explainer y obtener explicaciones
    explainer = shap.Explainer(best_model.named_steps["rf"], X_train)
    shap_exp = explainer(X_test, check_additivity=False)  # Desactivar la verificación de aditividad
    
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////
    expected_value = explainer.expected_value
    phi = shap_exp.values  # shape: (n_samples, n_features, n_classes)
    model_output = best_model.predict_proba(X_test)

    import numpy as np

    # Para cada clase (ejemplo: clase 1)
    class_idx = 1  # ajusta según tu problema
    shap_sum = expected_value[class_idx] + phi[:, :, class_idx].sum(axis=1)
    model_pred = model_output[:, class_idx]

    # Imprime las diferencias
    for i, (s, m) in enumerate(zip(shap_sum, model_pred)):
        print(f"Muestra {i}: SHAP sum = {s:.6f}, Model output = {m:.6f}, Diferencia = {abs(s-m):.6f}")
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Verifica la forma para entender su estructura
    # print(shap_exp.shape)               # → (20, 1757, 2)
    # print(shap_exp.values.shape)        # confirmación de dimensiones

    feature_names = names
    expected_value = explainer.expected_value[1]
    sample_idx = 0
    if shap_exp.values.ndim == 3 and shap_exp.values.shape[2] > 1:
        class_idx = 1  # o 0 si tu clase positiva es la primera
        explanation = shap_exp.values[sample_idx, :, class_idx]
    else:
        explanation = shap_exp.values[sample_idx, :]
    
    model_output = best_model.predict_proba(X_test)[sample_idx, 1]
    wf_plot = waterfall_plot(explanation, feature_names, expected_value, model_output, max_display=10)
    
    auc_plot_html = pio.to_html(auc_plot, full_html=False)
    shap_plot_html = pio.to_html(shap_plot, full_html=False)
    shap_dependence_plot_html = pio.to_html(shap_dependence_plot, full_html=False)
    wf_plot_html = pio.to_html(wf_plot, full_html=False)
    
    return {'auc_plot': auc_plot_html, 'shap_plot': shap_plot_html, 'shap_dependence_plot': shap_dependence_plot_html, 'shap_waterfall_plot': wf_plot_html}

def auc_plotly(permutation_scores, real_score):
    
    hist = go.Histogram(
        x=permutation_scores,
        nbinsx=20,
        histnorm='probability density',
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        name='Permutations'
    )
    
    counts, bins = np.histogram(permutation_scores, bins=20, density=True)
    max_height = max([0.1, counts.max()])

    line = go.Scatter(
        x=[real_score, real_score],
        y=[0, max_height],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Real AUC'
    )
    layout = go.Layout(
        xaxis_title='ROC AUC (perm)',
        yaxis_title='Density',
        title='AUC Permutation Test',
        template='plotly_white',
        legend=dict(x=0.7, y=0.95)
    )
    fig = go.Figure(data=[hist, line], layout=layout)
    return fig

def summary_plot(df_log1p, sv, names, X_test, top_n=20, jitter=0.2):
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [names[i] for i in top_idx]

    data = []
    rng = np.random.default_rng(42)
    for i, feat_idx in enumerate(top_idx):
        for sample_idx in range(sv.shape[0]):
            y_jitter = i + rng.uniform(-jitter, jitter)
            data.append({
                'Feature': top_features[i],
                'Feature_jitter': y_jitter,
                'Shap_value': sv[sample_idx, feat_idx],
                'Feature_val': X_test[sample_idx, feat_idx]
            })
    df_plot = pd.DataFrame(data)

    fig = px.scatter(
        df_plot,
        x='Shap_value',
        y='Feature_jitter',
        color='Feature_val',
        color_continuous_scale='RdBu',
        title=f"SHAP Summary Dot Plot – group '{df_log1p['Group'].unique()[1]}'",
        template='plotly_white',
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        coloraxis_colorbar=dict(
            thickness=10,
            title='Feature value'
        ),
        margin=dict(l=140, r=30, t=40, b=40),
        yaxis=dict(
            tickmode='array',
            tickvals=list(reversed(range(top_n))),
            ticktext=top_features,
            categoryorder='total descending'
        )
    )
    return fig
def dependence_plot(sv, X_test, names, top_feature_idx, top_feature_name):
    df = pd.DataFrame({
        'Feature_value': X_test[:, top_feature_idx],
        'SHAP_value': sv[:, top_feature_idx]
    })
    fig = px.scatter(
        df,
        x='Feature_value',
        y='SHAP_value',
        color='Feature_value',
        color_continuous_scale='RdBu',
        title=f"SHAP Dependence Plot – {top_feature_name}",
        template='plotly_white'
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        coloraxis_colorbar=dict(
            thickness=10,
            title='Feature value'
        ),
        margin=dict(l=80, r=30, t=40, b=40),
        xaxis_title=f"{top_feature_name} value",
        yaxis_title="SHAP value"
    )
    return fig
def waterfall_plot(shap_values, feature_names, expected_value, model_output, max_display=10):
    # Selecciona las características más importantes
    abs_vals = np.abs(shap_values)
    top_idx = np.argsort(abs_vals)[::-1][:max_display]
    top_features = [feature_names[i] for i in top_idx]
    top_shap = shap_values[top_idx]

    # Construye los valores para el waterfall
    measure = ['relative'] * len(top_shap) + ['total']
    x = top_features + ['Model Output']
    y = list(top_shap) + [model_output - expected_value - top_shap.sum()]

    # Valor inicial
    base = expected_value

    fig = go.Figure(go.Waterfall(
        name = "SHAP",
        orientation = "v",
        measure = measure,
        x = x,
        text = [f"{v:.3f}" for v in y],
        y = y,
        base = base,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color":"#3D9970"}},
        decreasing = {"marker":{"color":"#FF4136"}},
        totals = {"marker":{"color":"#0074D9"}}
    ))

    fig.update_layout(
        title=f"SHAP Waterfall Plot – Model Output: {model_output:.3f}",
        template="plotly_white",
        yaxis_title="Contribution to output",
        xaxis_title="Feature",
        showlegend=False
    )
    return fig