from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, QDialogButtonBox, 
                            QSpinBox, QLabel, QFrame, QMessageBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
import pandas as pd

from src.functions.imputation_methods import *

class DialogImputationModel(QDialog):
    def __init__(self, df_sample, df_sample_umb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Imputation Methods")
        self.setMinimumSize(300, 250)

        # Store the dataframes
        self.df_sample = df_sample
        self.df_sample_umb = df_sample_umb
        self.imputed_df = None # To store the result

        # --- UI Elements ---
        self.layout = QVBoxLayout(self)
        self.formLayout = QFormLayout()

        # Imputation method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "N Imputation",
            "Half Minimum Imputation",
            "Mean Imputation",
            "Median Imputation",
            "Miss Forest Imputation",
            "SVD Imputation",
            "KNN Imputation",
            "MICE (Bayesian Ridge)",
            "MICE (Linear Regression)"
        ])
        self.method_combo.currentIndexChanged.connect(self.update_options)

        # Description label
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)

        # Options widgets container
        self.options_frame = QFrame()
        self.options_layout = QFormLayout(self.options_frame)

        # --- Parameter Widgets ---
        # N Imputation
        self.n_spinbox = QDoubleSpinBox()
        self.n_spinbox.setRange(-1000000, 1000000)
        self.n_spinbox.setValue(0)

        # Miss Forest
        self.miss_forest_iter_spinbox = QSpinBox()
        self.miss_forest_iter_spinbox.setRange(1, 100)
        self.miss_forest_iter_spinbox.setValue(10)
        self.miss_forest_estimators_spinbox = QSpinBox()
        self.miss_forest_estimators_spinbox.setRange(1, 1000)
        self.miss_forest_estimators_spinbox.setValue(100)

        # SVD
        self.svd_components_spinbox = QSpinBox()
        self.svd_components_spinbox.setRange(1, 100)
        self.svd_components_spinbox.setValue(5)

        # KNN
        self.knn_neighbors_spinbox = QSpinBox()
        self.knn_neighbors_spinbox.setRange(1, 100)
        self.knn_neighbors_spinbox.setValue(2)

        # MICE
        self.mice_iter_spinbox = QSpinBox()
        self.mice_iter_spinbox.setRange(1, 100)
        self.mice_iter_spinbox.setValue(10)

        # Dialog buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.apply_imputation)
        self.buttonBox.rejected.connect(self.reject)

        # Assemble layout
        self.formLayout.addRow(QLabel("Imputation Method:"), self.method_combo)
        self.layout.addLayout(self.formLayout)
        self.layout.addWidget(self.description_label)
        self.layout.addWidget(self.options_frame)
        self.layout.addWidget(self.buttonBox)

        self.update_options() # Initial call to set the correct options

    def update_options(self):
        # A more robust way to clear the layout
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        method = self.method_combo.currentText()

        descriptions = {
            "N Imputation": "Replaces missing values with a fixed low constant (e.g., zero, a specific limit of detection, or the lowest possible instrument reading). Primarily considered for MNAR data where true values are indeed below detection. It explicitly assumes that the absence of a signal means a very low or non-existent abundance.",
            "Half Minimum Imputation": "Replaces missing values with half of the minimum observed value across the entire dataset or within a specific feature. A common heuristic specifically designed for MNAR data, assuming values are below the detection It largely falls into the category of single-value replacement.",
            "Mean Imputation": "Replaces missing values with the mean (for normally distributed data) of the observed values for that specific feature. Conceptually suitable for MCAR data, where missingness is truly random and doesn't depend on the value itself.",
            "Median Imputation": "Replaces missing values with the median (for skewed data) of the observed values for that specific feature. Conceptually suitable for MCAR data, where missingness is truly random and doesn't depend on the value itself.",
            "Miss Forest Imputation": "A non-parametric, iterative method that uses random forests to predict and impute missing values. For each feature with missing values, it builds a random forest model using other features as predictors. This process is repeated until the imputed values converge. Highly effective for MAR and can also handle complex MNAR scenarios reasonably well, especially when the missingness pattern can be inferred from other observed data. ",
            "SVD Imputation": "Uses matrix factorization (Singular Value Decomposition) to approximate the data matrix and impute missing values. It identifies underlying linear relationships and latent components in the data to estimate missing entries. Particularly well-suited for MAR data, especially after data normalization and variance-stabilizing transformations (e.g., logarithmization), which can linearize relationships. Modern iterative implementations are more robust and can handle MNAR data better by treating low-intensity missing values differently or by incorporating implicit assumptions about left-censoring.",
            "KNN Imputation": "Imputes missing values by finding the 'k' nearest neighbors (based on a distance metric) to a sample with missing values and then averaging (or taking the median) the observed values from those neighbors for the missing feature. Suitable for MAR data, especially at low to moderate missingness (typically <15-20%). It assumes that similar samples have similar molecular profiles.",
            "MICE (Bayesian Ridge)": "A powerful, flexible framework that generates multiple imputed datasets. It works by performing a series of regression models, where each missing variable is imputed conditional on other variables in the dataset. Theoretically superior for statistical inference as it accounts for the uncertainty introduced by imputation by generating multiple plausible imputed datasets. This is valuable for downstream statistical analysis where proper variance estimation is critical.",
            "MICE (Linear Regression)": "A powerful, flexible framework that generates multiple imputed datasets. It works by performing a series of regression models, where each missing variable is imputed conditional on other variables in the dataset. Theoretically superior for statistical inference as it accounts for the uncertainty introduced by imputation by generating multiple plausible imputed datasets. This is valuable for downstream statistical analysis where proper variance estimation is critical.y"
        }

        self.description_label.setText(descriptions.get(method, ""))

        if method == "N Imputation":
            self.options_layout.addRow(QLabel("N Value:"), self.n_spinbox)
        elif method == "Miss Forest Imputation":
            self.options_layout.addRow(QLabel("Max Iterations:"), self.miss_forest_iter_spinbox)
            self.options_layout.addRow(QLabel("N Estimators:"), self.miss_forest_estimators_spinbox)
        elif method == "SVD Imputation":
            self.options_layout.addRow(QLabel("N Components:"), self.svd_components_spinbox)
        elif method == "KNN Imputation":
            self.options_layout.addRow(QLabel("N Neighbors:"), self.knn_neighbors_spinbox)
        elif "MICE" in method:
            self.options_layout.addRow(QLabel("Max Iterations:"), self.mice_iter_spinbox)

    def apply_imputation(self):
        method = self.method_combo.currentText()
        ###df_to_impute = self.df_sample_umb.copy() # WIP
        
        try:
            # Always scale before imputation for advanced methods
            df_scaled = (df_to_impute - df_to_impute.mean()) / df_to_impute.std()
            if method == "N Imputation":
                n_val = self.n_spinbox.value()
                self.imputed_df = nImputed(df_to_impute, n=n_val)
            elif method == "Half Minimum Imputation":
                self.imputed_df = halfMinimumImputed(df_to_impute)
            elif method == "Mean Imputation":
                self.imputed_df = meanImputed(df_to_impute)
            elif method == "Median Imputation":
                self.imputed_df = medianImputed(df_to_impute)
            elif method == "Miss Forest Imputation":
                max_iter = self.miss_forest_iter_spinbox.value()
                n_est = self.miss_forest_estimators_spinbox.value()
                #self.imputed_df = missForestImputed(df_to_impute, max_iter=max_iter, n_estimators=n_est)
                self.imputed_df = missForestImputed(df_scaled, max_iter=max_iter, n_estimators=n_est).pipe(postprocess_imputation, df_to_impute)
            elif method == "SVD Imputation":
                n_comp = self.svd_components_spinbox.value()
                #self.imputed_df = svdImputed(df_to_impute, n_components=n_comp)
                self.imputed_df = svdImputed(df_scaled, n_components=n_comp).pipe(postprocess_imputation, df_to_impute)
            elif method == "KNN Imputation":
                n_neigh = self.knn_neighbors_spinbox.value()
                #self.imputed_df = knnImputed(df_to_impute, n_neighbors=n_neigh)
                self.imputed_df = knnImputed(df_scaled, n_neighbors=n_neigh).pipe(postprocess_imputation, df_to_impute)
            elif method == "MICE (Bayesian Ridge)":
                max_iter = self.mice_iter_spinbox.value()
                #self.imputed_df = miceBayesianRidgeImputed(df_to_impute, max_iter=max_iter)
                self.imputed_df = miceBayesianRidgeImputed(df_to_impute, max_iter=max_iter).pipe(postprocess_imputation, df_to_impute)
            elif method == "MICE (Linear Regression)":
                max_iter = self.mice_iter_spinbox.value()
                #self.imputed_df = miceLinearRegressionImputed(df_to_impute, max_iter=max_iter)
                self.imputed_df = miceLinearRegressionImputed(df_to_impute, max_iter=max_iter).pipe(postprocess_imputation, df_to_impute)
            
            QMessageBox.information(self, "Success", f"Imputation with {method} completed successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during imputation: {e}")
            self.imputed_df = None

    def getResults(self):
        return self.imputed_df