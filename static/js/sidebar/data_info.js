function updateProcessingSteps(steps) {
    const stepsList = document.getElementById('processingSteps');
    if (!stepsList) return;
    stepsList.innerHTML = ''; // Clear existing steps
    steps.forEach(function(step) {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        listItem.innerHTML = `<i class="fas ${step.icon} ${step.color} me-2"></i>${step.message}`;
        stepsList.appendChild(listItem);
    });
}

function resetData() {
    fetch(window.APP_URLS.reset_sample_step, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ context: 'sidebar' }), // Context can be adjusted if needed
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
        console.error('Error resetting data:', error);
        window.showToast('An error occurred while resetting data.', 'danger');
    });
}