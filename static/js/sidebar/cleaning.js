function initCleaning() {
    const replaceZerosBtn = document.getElementById('replaceZerosBtn');
    if (replaceZerosBtn) {
        replaceZerosBtn.onclick = function() {
            replaceZerosBtn.disabled = true;
            replaceZerosBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Replacing...';

            fetch(window.APP_URLS.replace_zeros, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
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
                console.error('Error replacing zeros:', error);
                window.showToast('An error occurred while replacing zeros.', 'danger');
            })
            .finally(() => {
                replaceZerosBtn.disabled = false;
                replaceZerosBtn.innerHTML = '<i class="fas fa-exchange-alt me-2"></i>Replace All Zeros with NaN';
            });
        }
    }
}