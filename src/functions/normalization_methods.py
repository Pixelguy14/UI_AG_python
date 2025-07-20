"""
1. Normalization (Between-Sample Adjustment)

Purpose: Correct for technical variations (e.g., batch effects, sample loading differences).
Key Methods:

    Total Ion Current (TIC):

        Divide each sample by its total intensity (sum) → Normalizes for overall abundance differences.

        Use Case: LC-MS data with varying sample concentrations.

    Quantile Normalization:

        Force all samples to have identical intensity distributions.

        Use Case: RNA-seq or when assuming all samples should follow the same distribution.

    Median/Sum Normalization:

        Scale by median or sum of each sample (robust to outliers).

    Sample-Specific Controls:

        Normalize to internal standards or housekeeping features.

User Controls Needed:

    Dropdown to select normalization method.

    Option to visualize pre/post distributions (boxplots/violin plots).

2. Transformation (Within-Sample Adjustment)

Purpose: Stabilize variance and handle skewed distributions.
Key Methods:

    Log Transformation (Log2/Log10):

        Compress dynamic range → Makes variance more homogeneous.

        Critical for: Fold-change calculations (e.g., log2(x + epsilon) to avoid log(0)).

    Square Root / Cube Root:

        Less aggressive than log → Useful for count data with zeros.

    Arcsinh Transformation:

        Handles zeros and negative values (common in proteomics).

User Controls Needed:

    Checkbox to apply transformation + method selection.

    Warning if data contains negative values (incompatible with log).

3. Scaling (Feature-Level Adjustment)

Purpose: Standardize features for downstream analysis (e.g., PCA, clustering).
Key Methods:

    Z-score (Standard Scaling):

        Center to mean=0, scale to std=1.

        Use Case: PCA, heatmaps, ML algorithms.

    Min-Max Scaling:

        Scale to [0, 1] range → Preserves sparsity.

    Pareto Scaling:

        Divide by √std → Compromise between Z-score and no scaling.

    Range Scaling:

        Scale to [-1, 1] → Preserves sign.

User Controls Needed:

    Radio buttons to select scaling method.

    Option to scale by features (columns) or samples (rows).
    

Visual Feedback:

    Generate paired plots (before/after) for each step:

        Boxplots (distribution per sample).

        Density plots (global distribution).

        PCA (check batch effect removal).
        
    implement Parallelization in the operations, and state preservation for both raw and processed data
"""
