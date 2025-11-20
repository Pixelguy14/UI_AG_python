function initPrediction(){
    const predictionMethodsRadios = document.querySelectorAll('input[name="predictionMethod"]');
    if (predictionMethodsRadios.length === 0){
        return;
    }

    const predictionMethodDescriptions = {
        'random_forest': 'Random Forest: An ensemble learning method that constructs multiple decision trees and outputs the mode of their predictions for classification tasks or mean prediction for regression tasks.',
    };

    function updatePredictionOptions(){
        const selectedRadio = document.querySelector('input[name="predictionMethod"]:checked');
        if (!selectedRadio){
            return;
        }
        const method = selectedRadio.value;
        const descriptionText = document.getElementById('predictionDescriptionText');
        const predictionDescription = document.getElementById('predictionDescription');
        const predictionControlsDiv = document.getElementById('prediction-controls');

        if (descriptionText){
            descriptionText.textContent = predictionMethodDescriptions[method];
            predictionDescription.style.display = 'block';
        }
        // Dynamically add parameters if needed
        let paramsHtml = '';
        if (method === 'random_forest'){
            paramsHtml = `
            <label for="trainTest-size">Test Size:</label>
                <input type="number" class="form-control" id="trainTest-size" value="0.3" min="0.1" max="0.9" step="0.1">
            <label for="classWeightPrediction">Class Weight:</label>
                <select class="form-select" id="classWeightPrediction">
                    <option value="balanced">Balanced</option>
                    <option value="balanced_subsample">Balanced subsample</option>
                    <option value="none" selected>None</option>
                </select>
            <label for="EstimatorsPrediction">Estimators:</label>
                <div class="input-group">
                    <input type="number" class="form-control" id="FirstEstimatorsPrediction" value="100" min="10" step="10">
                    <input type="number" class="form-control" id="LastEstimatorsPrediction" value="300" min="10" step="10">
                </div>
            <label for="MaxDepthPrediction">Max Depth:</label>
                <div class="input-group">
                    <input type="text" class="form-control" id="DefDepthPrediction" value="None">
                    <input type="number" class="form-control" id="MinDepthPrediction" value="5" min="1" step="1">
                    <input type="number" class="form-control" id="MaxDepthPrediction" value="20" min="1" step="1">
                </div>
            <label for="MinSamplesLeafPrediction">Min Samples Leaf:</label>
                <div class="input-group">
                    <input type="number" class="form-control" id="MinSamplesLeafPrediction" value="1" min="1" step="1">
                    <input type="number" class="form-control" id="MaxSamplesLeafPrediction" value="3" min="1" step="1">
                </div>`;
        }
        // Find the card-body within the prediction collapse and append paramsHtml
        if (predictionControlsDiv){
            // Remove existing dynamic parameters before adding new ones
            
             predictionControlsDiv.innerHTML = paramsHtml;
        }
    }

    function applyPrediction(){
        console.log(window.APP_URLS.apply_prediction);
        const method = document.querySelector('input[name="predictionMethod"]:checked').value;
        const sendPredictionBtn = document.getElementById('sendPredictionBtn');

        sendPredictionBtn.disabled = true;
        sendPredictionBtn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...`;

        let params = {};
        if (method === 'random_forest'){
            params = {
                'test_size': parseFloat(document.getElementById('trainTest-size').value),
                'class_weight': document.getElementById('classWeightPrediction').value,
                'n_estimators_start': parseInt(document.getElementById('FirstEstimatorsPrediction').value),
                'n_estimators_end': parseInt(document.getElementById('LastEstimatorsPrediction').value),
                'max_depth_default': document.getElementById('DefDepthPrediction').value,
                'max_depth_min': parseInt(document.getElementById('MinDepthPrediction').value),
                'max_depth_max': parseInt(document.getElementById('MaxDepthPrediction').value),
                'min_samples_leaf_min': parseInt(document.getElementById('MinSamplesLeafPrediction').value),
                'min_samples_leaf_max': parseInt(document.getElementById('MaxSamplesLeafPrediction').value)
            };
        }
        
        fetch(window.APP_URLS.apply_prediction, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ method: method, params: params }),
        })
        .then(async response => {
            if (!response.ok) {
            // Try to extract error message from HTML
                const text = await response.text();
            throw new Error(`Server error: ${response.status}\n${text}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success){
                window.showToast('Prediction applied successfully!', 'success');
                window.location.href = '/api/preprocessing/prediction';
            }else{
                window.showToast(data.message, 'danger');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            window.showToast('An error occurred while applying prediction.', 'danger');
        })
        .finally(() => {
            sendPredictionBtn.disabled = false;
            sendPredictionBtn.innerHTML = `<i class="fas fa-paper-plane me-2"></i>Send`;
        });
    }

    // Event listeners for Prediction method radios
    predictionMethodsRadios.forEach(radio => {
        radio.onclick = updatePredictionOptions;
    });
    const sendPredictionBtn = document.getElementById('sendPredictionBtn');
    if (sendPredictionBtn){
        sendPredictionBtn.onclick = applyPrediction;
    }

    // Initial call to set up descriptions and parameters
    if (!document.querySelector('input[name="predictionMethod"]:checked')){
        predictionMethodsRadios[0].checked = true;
    }
    updatePredictionOptions();

}
document.addEventListener('DOMContentLoaded', initPrediction);
