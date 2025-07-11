import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from scipy.stats import norm
import numpy as np
from src.functions.exploratory_data import *

# Plot prototype for Quality Control models
class PlotWidgetQC(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#F9F9F9")
        
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def clear_plot(self):
        self.plot_widget.clear()
        self.plot_widget.setTitle(None) # Also clear title
        self.plot_widget.setLabel('left', '')
        self.plot_widget.setLabel('bottom', '')
        # Remove legend if it exists
        if hasattr(self.plot_widget.plotItem, 'legend') and self.plot_widget.plotItem.legend:
            self.plot_widget.plotItem.legend.scene().removeItem(self.plot_widget.plotItem.legend)
            self.plot_widget.plotItem.legend = None


    def plot_missing_heatmap(self, df):
        self.clear_plot()
        
        # Invert data so null is 1 (colored) and not-null is 0 (background)
        data = df.isnull().to_numpy().astype(np.uint8)
        
        img = pg.ImageItem(data)
        self.plot_widget.addItem(img)
        
        # Colormap: 0 is background, 1 is the "missing" color
        colors = [[31, 56, 104, 255], [249, 249, 249, 255]] # #1f3868 and #F9F9F9
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 2), color=colors)
        img.setColorMap(cmap)
        
        self.plot_widget.setTitle('Missing Values Distribution')
        self.plot_widget.getViewBox().invertY(True)
        
        # Hide axes
        self.plot_widget.getAxis('bottom').hide()
        self.plot_widget.getAxis('left').hide()

    def plot_data_types_distribution(self, summary):
        self.clear_plot()
        
        type_counts = summary['type'].value_counts()
        x = np.arange(len(type_counts))
        
        bargraph = pg.BarGraphItem(x=x, height=type_counts.values, width=0.6)
        self.plot_widget.addItem(bargraph)
        
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Data Type')
        self.plot_widget.setTitle('Data Types Distribution')
        
        ticks = list(zip(x, type_counts.index))
        self.plot_widget.getAxis('bottom').setTicks([ticks])
        self.plot_widget.getAxis('bottom').setTextPen('black')
        self.plot_widget.getAxis('left').setTextPen('black')

    def plot_correlation_matrix(self, df):
        self.clear_plot()
        
        numerical_df = df.select_dtypes(include=[np.number])
        
        if not numerical_df.empty and numerical_df.shape[1] > 1:
            corr_matrix = numerical_df.corr()
            corr_data = corr_matrix.to_numpy()
            
            img = pg.ImageItem(corr_data)
            self.plot_widget.addItem(img)
            
            colormap = pg.colormap.get('viridis')
            img.setColorMap(colormap)
            
            self.plot_widget.setTitle('Correlation Matrix')
            
            # Set axis ticks
            x_ticks = list(enumerate(corr_matrix.columns))
            y_ticks = list(enumerate(corr_matrix.index))
            self.plot_widget.getAxis('bottom').setTicks([x_ticks])
            self.plot_widget.getAxis('left').setTicks([y_ticks])
            self.plot_widget.getAxis('bottom').setTextPen('black')
            self.plot_widget.getAxis('left').setTextPen('black')
        else:
            text = pg.TextItem("Not enough numerical columns", anchor=(0.5, 0.5), color='black')
            self.plot_widget.addItem(text)
            self.plot_widget.setTitle('Correlation Matrix')

    def plot_null_pie(self, df):
        self.clear_plot()
        
        total_elements = df.size
        if total_elements > 0:
            total_nulls = df.isnull().sum().sum()
            total_non_nulls = total_elements - total_nulls
            
            bargraph = pg.BarGraphItem(
                x=[0, 1], 
                height=[total_nulls, total_non_nulls], 
                width=0.6, 
                brushes=['#ff9999', '#66b3ff']
            )
            self.plot_widget.addItem(bargraph)
            
            self.plot_widget.setTitle('Count of Null vs Non-Null Values')
            self.plot_widget.setLabel('left', 'Total Count')
            
            ticks = [[(0, 'Null Values'), (1, 'Non-Null Values')]]
            self.plot_widget.getAxis('bottom').setTicks(ticks)
            self.plot_widget.getAxis('bottom').setTextPen('black')
            self.plot_widget.getAxis('left').setTextPen('black')
        else:
            text = pg.TextItem("DataFrame is empty", anchor=(0.5, 0.5), color='black')
            self.plot_widget.addItem(text)
            self.plot_widget.setTitle('Null Distribution')
    
    def plot_bar_chart(self, mean_tic_data, column_names, N):
        self.clear_plot()

        ind = np.arange(N)
        plot_values = mean_tic_data[:N]
        plot_labels = column_names[:N]

        bargraph = pg.BarGraphItem(x=ind, height=plot_values, width=0.6, brush="#3156A1")
        self.plot_widget.addItem(bargraph)

        self.plot_widget.setLabel("left", "Mean log2 intensity")
        self.plot_widget.setTitle(f"{N} samples")
        
        ticks = list(zip(ind, plot_labels))
        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([ticks])
        axis.setTextPen('black')
        self.plot_widget.getAxis('left').setTextPen('black')
        
        # Rotate labels if needed
        if not plot_labels.empty:
            max_label_len = max(len(str(label)) for label in plot_labels)
            if max_label_len > 15 or N > 20:
                axis.setRotation(90)
            elif N > 7:
                axis.setRotation(45)
            else:
                axis.setRotation(0)
        else:
            axis.setRotation(0)


    def plot_distribution(self, data_series):
        self.clear_plot()
        
        if data_series.empty or data_series.isna().all():
            text = pg.TextItem("No data available", anchor=(0.5, 0.5), color='black')
            self.plot_widget.addItem(text)
            self.plot_widget.setTitle('Data Distribution')
            return
        
        clean_data = data_series.dropna()
        n = len(clean_data)
        bins = max(5, min(50, int(np.ceil(np.log2(n)) + 1)))
        
        # Histogram
        y, x = np.histogram(clean_data, bins=bins, density=True)
        hist = pg.BarGraphItem(x=x[:-1], height=y, width=(x[1]-x[0]), brush='skyblue', pen='k', name='Histogram')
        self.plot_widget.addItem(hist)
        
        # Add legend
        self.plot_widget.addLegend()

        # Normal distribution curve
        mu, std = clean_data.mean(), clean_data.std()
        x_curve = np.linspace(clean_data.min(), clean_data.max(), 500)
        normal_curve = norm.pdf(x_curve, mu, std)
        self.plot_widget.plot(x_curve, normal_curve, pen=pg.mkPen('g', style=QtCore.Qt.DashLine, width=1.5), name='Normal Dist')
        
        # Stats text
        stats_text = (f"n = {n}\nμ = {mu:.4f}\nσ = {std:.4f}\n"
                    f"Skew = {clean_data.skew():.4f}\nKurt = {clean_data.kurtosis():.4f}")
        text_item = pg.TextItem(stats_text, color='black', anchor=(1, 1))
        text_item.setPos(self.plot_widget.getViewBox().viewRange()[0][1], self.plot_widget.getViewBox().viewRange()[1][1])
        self.plot_widget.addItem(text_item)

        # Formatting
        self.plot_widget.setTitle(f'Distribution of {data_series.name}')
        self.plot_widget.setLabel('bottom', 'Value')
        self.plot_widget.setLabel('left', 'Density')
        self.plot_widget.getAxis('bottom').setTextPen('black')
        self.plot_widget.getAxis('left').setTextPen('black')
    
    def plot_line_data(self, data):
        self.clear_plot()
        self.plot_widget.plot(data, pen='b', symbol='*')
