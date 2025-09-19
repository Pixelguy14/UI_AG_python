function initTransformation() {
    const transformationMethodRadios = document.querySelectorAll('input[name="transformationMethod"]');
    if (transformationMethodRadios.length === 0) {
        return; // No transformation radios on this page
    }

    const transformationMethodDescriptions = {
        'log2': 'Applies log base 2 transformation. Useful for count data or data with a wide range.',
        'log10': 'Applies log base 10 transformation. Similar to log2, but with a different base.',
        'sqrt': 'Applies square root transformation. Can help stabilize variance and make data more normal.',
        'cube_root': 'Applies cube root transformation. Less aggressive than square root, useful for skewed data.',
        'arcsinh': 'Applies arcsinh transformation. A common alternative to log transformation, especially for data that includes zero or negative values.',
        'glog': 'Applies generalized log transformation. A flexible transformation that can handle data with a wide range and zeros.',
        'yeo_johnson': 'Applies Yeo-Johnson transformation. A power transformation that works for both positive and negative values.',
    };

    function updateTransformationOptions() {
        const selectedRadio = document.querySelector('input[name="transformationMethod"]:checked');
        if (!selectedRadio) {
            return;
        }
        const method = selectedRadio.value;
        const descriptionText = document.getElementById('transformationDescriptionText');
        const transformationDescription = document.getElementById('transformationDescription');

        if (descriptionText) {
            descriptionText.textContent = transformationMethodDescriptions[method];
            transformationDescription.style.display = 'block';
        }
    }

    function applyTransformation() {
        const method = document.querySelector('input[name="transformationMethod"]:checked').value;
        const sendTransformationBtn = document.getElementById('sendTransformationBtn');

        sendTransformationBtn.disabled = true;
        sendTransformationBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';

        fetch(window.APP_URLS.apply_transformation, {
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
            console.error('Error applying transformation:', error);
            window.showToast('An unexpected error occurred during transformation.', 'danger');
        })
        .finally(() => {
            sendTransformationBtn.disabled = false;
            sendTransformationBtn.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send';
        });
    }

    // Event listeners for Transformation
    transformationMethodRadios.forEach(radio => {
        radio.onchange = updateTransformationOptions;
    });
    const sendTransformationBtn = document.getElementById('sendTransformationBtn');
    if (sendTransformationBtn) {
        sendTransformationBtn.onclick = applyTransformation;
    }

    // Initial call to set up descriptions
    if (!document.querySelector('input[name="transformationMethod"]:checked')) {
        transformationMethodRadios[0].checked = true;
    }
    updateTransformationOptions();
}