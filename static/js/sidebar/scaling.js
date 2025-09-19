function initScaling() {
    const scalingMethodRadios = document.querySelectorAll('input[name="scalingMethod"]');
    if (scalingMethodRadios.length === 0) {
        return; // No scaling radios on this page
    }

    const scalingMethodDescriptions = {
        'standard': 'Applies Standard Scaling (Z-score). Centers the data to 0 and scales to unit variance.',
        'minmax': 'Applies Min-Max Scaling. Scales features to a given range, typically 0-1.',
        'pareto': 'Applies Pareto Scaling. Divides by the square root of the standard deviation. Reduces the importance of large values.',
        'range': 'Applies Range Scaling. Scales data to a range of -1 to 1.',
        'robust': 'Applies Robust Scaling. Scales features using statistics that are robust to outliers (median and interquartile range).',
        'vast': 'Applies VAST Scaling. A combination of autoscaling and Pareto scaling, often used in metabolomics.',
    };

    function updateScalingOptions() {
        const selectedRadio = document.querySelector('input[name="scalingMethod"]:checked');
        if (!selectedRadio) {
            return;
        }
        const method = selectedRadio.value;
        const descriptionText = document.getElementById('scalingDescriptionText');
        const scalingDescription = document.getElementById('scalingDescription');

        if (descriptionText) {
            descriptionText.textContent = scalingMethodDescriptions[method];
            scalingDescription.style.display = 'block';
        }
    }

    function applyScaling() {
        const method = document.querySelector('input[name="scalingMethod"]:checked').value;
        const sendScalingBtn = document.getElementById('sendScalingBtn');

        sendScalingBtn.disabled = true;
        sendScalingBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';

        fetch(window.APP_URLS.apply_scaling, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ method: method }),
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
            console.error('Error applying scaling:', error);
            window.showToast('An unexpected error occurred during scaling.', 'danger');
        })
        .finally(() => {
            sendScalingBtn.disabled = false;
            sendScalingBtn.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send';
        });
    }

    // Event listeners for Scaling
    scalingMethodRadios.forEach(radio => {
        radio.onchange = updateScalingOptions;
    });
    const sendScalingBtn = document.getElementById('sendScalingBtn');
    if (sendScalingBtn) {
        sendScalingBtn.onclick = applyScaling;
    }

    // Initial call to set up descriptions
    if (!document.querySelector('input[name="scalingMethod"]:checked')) {
        scalingMethodRadios[0].checked = true;
    }
    updateScalingOptions();
}