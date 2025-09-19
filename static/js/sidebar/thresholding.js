function initThresholding() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValueSpan = document.getElementById('thresholdValue');
    const applyThresholdBtn = document.getElementById('applyThresholdBtn');
    const thresholdProgress = document.getElementById('thresholdProgress');

    if (thresholdSlider && thresholdValueSpan) {
        thresholdSlider.addEventListener('input', function() {
            thresholdValueSpan.textContent = this.value;
        });
    }

    if (applyThresholdBtn) {
        applyThresholdBtn.onclick = function() {
            const threshold = thresholdSlider.value;
            thresholdProgress.style.display = 'block';
            applyThresholdBtn.disabled = true;

            fetch(window.APP_URLS.threshold, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ threshold: parseFloat(threshold) }),
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
                console.error('Error applying thresholding:', error);
                window.showToast('An unexpected error occurred during thresholding.', 'danger');
            })
            .finally(() => {
                thresholdProgress.style.display = 'none';
                applyThresholdBtn.disabled = false;
            });
        }
    }
}