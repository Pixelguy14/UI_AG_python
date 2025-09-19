function initNormalization() {
    const normalizationMethodRadios = document.querySelectorAll('input[name="normalizationMethod"]');
    if (normalizationMethodRadios.length === 0) {
        return; // No normalization radios on this page
    }

    const normalizationMethodDescriptions = {
        'tic': 'Normalizes each sample by its total ion count (sum of intensities).',
        'mtic': 'Normalizes each sample by its TIC, then scales to the median TIC of all samples.',
        'median': 'Normalizes each sample (column) by its median intensity. Robust to outliers.',
        'quantile': 'Forces all samples to have identical intensity distributions.',
        'pqn': 'Corrects for dilution effects based on a reference spectrum (median).',
    };

    function updateNormalizationOptions() {
        const selectedRadio = document.querySelector('input[name="normalizationMethod"]:checked');
        if (!selectedRadio) {
            return;
        }
        const method = selectedRadio.value;
        const descriptionText = document.getElementById('normalizationDescriptionText');
        const normalizationDescription = document.getElementById('normalizationDescription');

        if (descriptionText) {
            descriptionText.textContent = normalizationMethodDescriptions[method];
            normalizationDescription.style.display = 'block';
        }
    }

    function applyNormalization() {
        const method = document.querySelector('input[name="normalizationMethod"]:checked').value;
        const sendNormalizationBtn = document.getElementById('sendNormalizationBtn');

        sendNormalizationBtn.disabled = true;
        sendNormalizationBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';

        fetch(window.APP_URLS.apply_normalization, {
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
            console.error('Error applying normalization:', error);
            window.showToast('An unexpected error occurred during normalization.', 'danger');
        })
        .finally(() => {
            sendNormalizationBtn.disabled = false;
            sendNormalizationBtn.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send';
        });
    }

    // Event listeners for Normalization
    normalizationMethodRadios.forEach(radio => {
        radio.onchange = updateNormalizationOptions;
    });
    const sendNormalizationBtn = document.getElementById('sendNormalizationBtn');
    if (sendNormalizationBtn) {
        sendNormalizationBtn.onclick = applyNormalization;
    }

    // Initial call to set up descriptions
    if (!document.querySelector('input[name="normalizationMethod"]:checked')) {
        normalizationMethodRadios[0].checked = true;
    }
    updateNormalizationOptions();
}