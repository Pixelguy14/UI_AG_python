function exportPlotAsSVG_SQR(plotId, filename) {
    const plotElement = document.getElementById(plotId);
    if (plotElement && plotElement.layout) {
        Plotly.downloadImage(plotElement, { format: 'svg', width: 1200, height: 1200, filename: filename });
    } else {
        // console.error('Plotly graph with id ' + plotId + ' not found or not ready.');
        window.showToast('Plotly graph with id ' + plotId + ' not found or not ready.','danger')
        alert('Could not export plot. Please make sure the plot is fully loaded.');
    }
}

function exportPlotAsSVG_RCT(plotId, filename) {
    const plotElement = document.getElementById(plotId);
    if (plotElement && plotElement.layout) {
        Plotly.downloadImage(plotElement, { format: 'svg', width: 1600, height: 800, filename: filename });
    } else {
        // console.error('Plotly graph with id ' + plotId + ' not found or not ready.');
        window.showToast('Plotly graph with id ' + plotId + ' not found or not ready.','danger')
        alert('Could not export plot. Please make sure the plot is fully loaded.');
    }
}

function exportPlotAsSVG_RCT_2(plotId, filename) {
    const plotElement = document.getElementById(plotId);
    if (plotElement && plotElement.layout) {
        Plotly.downloadImage(plotElement, { format: 'svg', width: 1200, height: 400, filename: filename });
    } else {
        // console.error('Plotly graph with id ' + plotId + ' not found or not ready.');
        window.showToast('Plotly graph with id ' + plotId + ' not found or not ready.','danger')
        alert('Could not export plot. Please make sure the plot is fully loaded.');
    }
}