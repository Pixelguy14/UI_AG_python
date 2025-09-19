function initImputation() {
    const imputationMethodRadios = document.querySelectorAll('input[name="imputationMethod"]');
    if (imputationMethodRadios.length === 0) {
        return; // No imputation radios on this page
    }

    const imputationMethodDescriptions = {
        'n_imputation': 'Replaces missing values with a constant (e.g., 0). Best for data missing not at random (MNAR) where values are below a detection limit.',
        'half_minimum': 'A common heuristic for MNAR data, replacing missing values with half of the minimum observed value in the feature.',
        'mean': 'Replaces missing values with the feature\'s mean. Suitable for data missing completely in random (MCAR).',
        'median': 'Replaces missing values with the feature\'s median. More robust to outliers than mean imputation, also for MCAR data.',
        'miss_forest': 'Uses Random Forest to predict and impute missing values. Effective for all missing data types (MCAR, MAR, MNAR).',
        'svd': 'Uses Singular Value Decomposition to approximate the data matrix and impute values. Best for data missing at random (MAR).',
        'knn': 'Imputes missing values using the average of the k-nearest neighbors. Suitable for MAR data.',
        'mice_bayesian': 'Uses MICE with Bayesian Ridge regression. Good for complex, high-dimensional data, providing regularization.',
        'mice_linear': 'Uses MICE with standard Linear Regression. Models each feature with missing values as a function of other features.'
    };

    function updateImputationOptions() {
        const selectedRadio = document.querySelector('input[name="imputationMethod"]:checked');
        if (!selectedRadio) {
            return;
        }
        const method = selectedRadio.value;
        const descriptionText = document.getElementById('imputationDescriptionText');
        const imputationDescription = document.getElementById('imputationDescription');
        const imputationMethodDiv = document.getElementById('imputation'); // The collapse div

        if (descriptionText) {
            descriptionText.textContent = imputationMethodDescriptions[method];
            imputationDescription.style.display = 'block';
        }

        // Dynamically add parameters if needed (similar to imputation.html)
        let paramsHtml = '';
        if (method === 'n_imputation') {
            paramsHtml = `
            <div class="mb-3">
                <label for="nValue" class="form-label">N Value:</label>
                <input type="number" class="form-control" id="nValue" value="0" step="0.01">
            </div>`;
        } else if (method === 'miss_forest') {
            paramsHtml = `
            <div class="mb-3">
                <label for="maxIter" class="form-label">Max Iterations:</label>
                <input type="number" class="form-control" id="maxIter" value="10" min="1">
            </div>
            <div class="mb-3">
                <label for="nEstimators" class="form-label">N Estimators:</label>
                <input type="number" class="form-control" id="nEstimators" value="100" min="10">
            </div>`;
        } else if (method === 'svd') {
            paramsHtml = `
            <div class="mb-3">
                <label for="nComponents" class="form-label">N Components:</label>
                <input type="number" class="form-control" id="nComponents" value="5" min="1">
            </div>`;
        } else if (method === 'knn') {
            paramsHtml = `
            <div class="mb-3">
                <label for="nNeighbors" class="form-label">N Neighbors:</label>
                <input type="number" class="form-control" id="nNeighbors" value="2" min="1">
            </div>`;
        } else if (method === 'mice_bayesian' || method === 'mice_linear') {
            paramsHtml = `
            <div class="mb-3">
                <label for="miceMaxIter" class="form-label">Max Iterations:</label>
                <input type="number" class="form-control" id="miceMaxIter" value="100" min="1">
            </div>`;
        }
        // Find the card-body within the imputation collapse and append paramsHtml
        const cardBody = imputationMethodDiv.querySelector('.card-body');
        if (cardBody) {
            // Remove existing dynamic parameters before adding new ones
            const existingParams = cardBody.querySelectorAll('.dynamic-param');
            existingParams.forEach(param => param.remove());
            
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = paramsHtml;
            Array.from(tempDiv.children).forEach(child => {
                child.classList.add('dynamic-param'); // Mark as dynamic param
                cardBody.insertBefore(child, document.getElementById('sendImputationBtn'));
            });
        }
    }

    function applyImputation() {
        const method = document.querySelector('input[name="imputationMethod"]:checked').value;
        const sendImputationBtn = document.getElementById('sendImputationBtn');
        // No progress bar for imputation in sidebar, just disable button

        sendImputationBtn.disabled = true;
        sendImputationBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';
        
        let params = {};
        if (method === 'n_imputation') params.n_value = parseFloat(document.getElementById('nValue').value);
        if (method === 'miss_forest') {
            params.max_iter = parseInt(document.getElementById('maxIter').value);
            params.n_estimators = parseInt(document.getElementById('nEstimators').value);
        }
        if (method === 'svd') params.n_components = parseInt(document.getElementById('nComponents').value);
        if (method === 'knn') params.n_neighbors = parseInt(document.getElementById('nNeighbors').value);
        if (method === 'mice_bayesian' || method === 'mice_linear') params.max_iter = parseInt(document.getElementById('miceMaxIter').value);

        fetch(window.APP_URLS.apply_imputation, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ method: method, params: params }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.showToast(data.message, 'success');
                const currentShapeSpan = document.getElementById('currentShape');
                if (currentShapeSpan) {
                    currentShapeSpan.textContent = `${data.new_shape[0]} rows, ${data.new_shape[1]} columns`;
                }
                updateProcessingSteps(data.steps);
                htmx.trigger(document.body, 'plots-changed');
            } else {
                window.showToast(data.error, 'danger');
            }
        })
        .catch(error => {
            console.error('Error applying imputation:', error);
            window.showToast('An unexpected error occurred during imputation.', 'danger');
        })
        .finally(() => {
            sendImputationBtn.disabled = false;
            sendImputationBtn.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send';
        });
    }

    // Event listeners for Imputation
    imputationMethodRadios.forEach(radio => {
        radio.onchange = updateImputationOptions;
    });
    const sendImputationBtn = document.getElementById('sendImputationBtn');
    if (sendImputationBtn) {
        sendImputationBtn.onclick = applyImputation;
    }
    
    // Initial call to set up descriptions and parameters
    if (!document.querySelector('input[name="imputationMethod"]:checked')) {
        imputationMethodRadios[0].checked = true;
    }
    updateImputationOptions();
}