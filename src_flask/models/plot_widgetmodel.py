from PyQt5.QtWidgets import QVBoxLayout, QWidget
from scipy.stats import norm, gaussian_kde
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import seaborn as sns
from src.functions.exploratory_data import *

# Plot prototype for Quality Control models
class PlotWidgetQC(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(layout='constrained', facecolor="#F9F9F9")
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.last_width = 0
        self.last_height = 0

    def calculate_font_sizes(self, num_items=0):
        # Get current figure dimensions in inches
        fig_width, fig_height = self.figure.get_size_inches()
        
        # Calculate base font size based on figure area
        area = fig_width * fig_height
        base_size = max(8, min(12, 8 * (area / 30)))  # Constrain between 6-12
        
        # Adjust for number of items
        if num_items > 0:
            density_factor = min(1.0, 30 / num_items)  # Reduce font for dense data
            base_size *= density_factor
        
        return {
            'title': base_size * 1.2,
            'axis': base_size,
            'ticks': base_size * 0.8,
            'annotations': base_size * 0.7
        }

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_missing_heatmap(self, df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        custom_cmap = ListedColormap(["#1f3868", "#F9F9F9"])
        
        # Create heatmap
        #sns.heatmap(df.isnull(), cbar=False, cmap='binary_r', ax=ax)
        sns.heatmap(df.isnull(), cbar=False, cmap=custom_cmap, ax=ax)
        ax.set_title('Missing Values Distribution', fontsize=10)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.invert_yaxis()  # Invert Y-axis
        self.canvas.draw()

    def plot_data_types_distribution(self, summary):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Prepare data
        type_counts = summary['type'].value_counts()
        # Get dynamic font sizes
        num_categories = len(type_counts)
        font_sizes = self.calculate_font_sizes(num_categories)
        
        # Create bar plot
        sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, legend=False, palette='mako', ax=ax)
        ax.set_ylim(0, type_counts.max() * 1.1)
        ax.set_yticks([0, type_counts.max()])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_title('Data Types Distribution', fontsize=font_sizes['title'])
        ax.set_xlabel('Data Type', fontsize=font_sizes['axis'])
        ax.set_ylabel('Count', fontsize=font_sizes['axis'])

        ax.set_xticks(range(len(type_counts.index)))
        ax.set_xticklabels(type_counts.index, ha='right', fontsize=6)
        ax.set_xticklabels(
            type_counts.index, 
            ha='right', 
            fontsize=font_sizes['ticks'],
            rotation=45 if any(len(str(x)) > 10 for x in type_counts.index) else 0
        )
        self.canvas.draw()

    def plot_correlation_matrix(self, df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        num_cols = numerical_df.shape[1] if not numerical_df.empty else 0

        # Get dynamic font sizes
        font_sizes = self.calculate_font_sizes(num_cols)
        
        if not numerical_df.empty and numerical_df.shape[1] > 1:
            corr_matrix = numerical_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                        fmt=".2f", linewidths=.5, ax=ax, annot_kws={"fontsize": font_sizes["ticks"]})
            ax.set_title('Correlation Matrix', fontsize=font_sizes["title"])
            ax.tick_params(axis='x', labelsize=font_sizes["axis"])
            ax.tick_params(axis='y', labelsize=font_sizes["axis"])
        else:
            ax.text(0.5, 0.5, "Not enough numerical columns", 
                    ha='center', va='center', fontsize=font_sizes["axis"])
            ax.set_title('Correlation Matrix', fontsize=font_sizes["axis"])
        self.canvas.draw()

    def plot_null_pie(self, df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Calculate values
        total_elements = df.size
        total_nulls = df.isnull().sum().sum()
        font_sizes = self.calculate_font_sizes(2)
        
        if total_elements > 0:
            total_non_nulls = total_elements - total_nulls
            labels = ['Null Values', 'Non-Null Values']
            sizes = [total_nulls, total_non_nulls]
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0)  # Explode null slice
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90,
                   textprops={'fontsize': font_sizes["axis"]}, # Reduced label font size
                   pctdistance=0.8) # Distance of percentage from center
            ax.set_title('Percentaje of Null Values',fontsize=font_sizes["title"])
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, "DataFrame is empty", 
                    ha='center', va='center', fontsize=font_sizes["ticks"])
            ax.set_title('Null Distribution', fontsize=font_sizes["ticks"])
        self.canvas.draw()
    
    def plot_bar_chart(self, mean_tic_data, column_names, N):
        ax = self.figure.add_subplot(111)
        ax.clear()

        # Get dynamic font sizes
        font_sizes = self.calculate_font_sizes(N)

        ind = np.arange(N)
        plot_values = mean_tic_data[:N]
        plot_labels = column_names[:N]

        ax.bar(ind, plot_values, color="#3156A1")
        ax.set_ylabel("Mean log2 intensity", fontsize=font_sizes['axis'])
        ax.set_title(f"{N} samples", fontsize=font_sizes['title'])
        ax.set_xticks(ind)
        
        # Dynamic label rotation based on label length
        max_label_len = max(len(str(label)) for label in plot_labels)
        rotation = 90 if max_label_len > 15 or N > 20 else 45 if N > 7 else 0
        
        ax.set_xticklabels(
            plot_labels, 
            rotation=rotation, 
            ha='right' if rotation else 'center',
            fontsize=font_sizes['ticks']
        )
        self.canvas.draw()

    def plot_distribution(self, data_series):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Handle empty data
        if data_series.empty or data_series.isna().all():
            ax.text(0.5, 0.5, "No data available", 
                    ha='center', va='center', fontsize=12)
            ax.set_title('Data Distribution', fontsize=12)
            self.canvas.draw()
            return
        
        # Clean data
        clean_data = data_series.dropna()
        
        # Calculate optimal bin count using Sturges' formula
        n = len(clean_data)
        bins = max(5, min(50, int(np.ceil(np.log2(n)) + 1)))  # 5-50 bins
        
        # Plot histogram with density normalization
        ax.hist(clean_data, bins=bins, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', label='Histogram')
        
        # Plot KDE
        kde = gaussian_kde(clean_data)
        xmin, xmax = clean_data.min(), clean_data.max()
        x = np.linspace(xmin, xmax, 500)
        #ax.plot(x, kde(x), 'r-', linewidth=2, label='Kernel Density')
        
        # Overlay normal distribution
        mu, std = clean_data.mean(), clean_data.std()
        normal_curve = norm.pdf(x, mu, std)
        ax.plot(x, normal_curve, 'g--', linewidth=1.5, label='Normal Dist')
        
        # Add statistical information
        stats_text = (f"n = {n}\nμ = {mu:.4f}\nσ = {std:.4f}\n"
                    f"Skew = {clean_data.skew():.4f}\nKurt = {clean_data.kurtosis():.4f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_title(f'Distribution of {data_series.name}', fontsize=12)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas.draw()
    
    def plot_line_data(self, data):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data, '*-')
        self.figure.tight_layout()
        self.canvas.draw()
