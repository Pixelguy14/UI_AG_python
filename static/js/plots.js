let clustergramResizeHandler = null;

function fetchAndRenderClustergram() {
    const clustergramDiv = document.getElementById('clustergramPlot');
    if (!clustergramDiv) return;

    const topN = document.getElementById('topNFeatures').value;
    const distanceMetric = document.getElementById('distanceMetric').value;
    const linkageMethod = document.getElementById('linkageMethod').value;
    const yAxisLabelSelect = document.getElementById('yAxisLabel');
    const yAxisLabel = yAxisLabelSelect.value;
    const yAxisTitle = yAxisLabelSelect.options[yAxisLabelSelect.selectedIndex].text;
    const colorPalette = document.getElementById('colorPalette').value;

    fetch('/api/clustergram_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            top_n: topN,
            distance_metric: distanceMetric,
            linkage_method: linkageMethod,
            y_axis_label: yAxisLabel,
            color_palette: colorPalette
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            clustergramDiv.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
            return;
        }

        const heatmapTrace = data.heatmap;
        const colDendroTraces = data.col_dendro;
        const rowDendroTraces = data.row_dendro;
        const heatmapCustomData = data.heatmap_customdata;
        const metadataColumnNames = data.metadata_column_names;

        let hoverTemplate = 
            '<b>Feature:</b> %{y}<br>' +
            '<b>Sample:</b> %{x}<br>' +
            '<b>Z-score:</b> %{z:.2f}<br>';

        metadataColumnNames.forEach((colName, index) => {
            hoverTemplate += `<b>${colName}:</b> %{customdata[${index}]}<br>`;
        });
        hoverTemplate += '<extra></extra>';
        heatmapTrace.hovertemplate = hoverTemplate;
        heatmapTrace.customdata = heatmapCustomData;

        let maxColDist = 0;
        colDendroTraces.forEach(trace => {
            trace.y.forEach(val => {
                if (val > maxColDist) maxColDist = val;
            });
        });

        let maxRowDist = 0;
        rowDendroTraces.forEach(trace => {
            trace.x.forEach(val => {
                if (val > maxRowDist) maxRowDist = val;
            });
        });

        const xMin = Math.min(...data.heatmap_x);
        const xMax = Math.max(...data.heatmap_x);
        const yMin = Math.min(...data.heatmap_y);
        const yMax = Math.max(...data.heatmap_y);

        let displayTickVals = data.heatmap_y;
        let displayTickText = data.row_labels;

        const maxLabels = 50;
        if (data.row_labels.length > maxLabels) {
            const samplingInterval = Math.ceil(data.row_labels.length / maxLabels);
            displayTickVals = [];
            displayTickText = [];
            for (let i = 0; i < data.row_labels.length; i += samplingInterval) {
                displayTickVals.push(data.heatmap_y[i]);
                displayTickText.push(data.row_labels[i]);
            }
        }
        
        const layout = {
            title: `Clustergram of Top ${data.row_labels.length} Features`,
            height: 1000,
            autosize: true,
            showlegend: false,
            hovermode: 'closest',
            margin: { l: 150, r: 50, t: 100, b: 150 },
            
            xaxis: {
                domain: [0.15, 0.85],
                range: [xMin, xMax],
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                matches: 'x3'
            },
            yaxis: {
                domain: [0.82, 0.98],
                range: [0, maxColDist * 1.05],
                showgrid: false,
                zeroline: false,
                showticklabels: false
            },

            xaxis2: {
                domain: [0, 0.12],
                range: [maxRowDist * 1.05, 0],
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                autorange: 'reversed'
            },
            yaxis2: {
                domain: [0.02, 0.8],
                range: [yMin, yMax],
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                matches: 'y3'
            },

            xaxis3: {
                domain: [0.15, 0.85],
                range: [xMin, xMax],
                tickvals: data.heatmap_x,
                ticktext: data.column_labels,
                tickangle: -90,
                side: 'bottom',
                anchor: 'y3'
            },
            yaxis3: {
                domain: [0.02, 0.8],
                range: [yMin, yMax],
                tickvals: displayTickVals,
                ticktext: displayTickText,
                side: 'right',
                title: {
                    text: yAxisTitle,
                    font: { size: 16 }
                }
            }
        };

        heatmapTrace.xaxis = 'x3';
        heatmapTrace.yaxis = 'y3';

        colDendroTraces.forEach(trace => {
            trace.xaxis = 'x';
            trace.yaxis = 'y';
        });

        rowDendroTraces.forEach(trace => {
            trace.xaxis = 'x2';
            trace.yaxis = 'y2';
        });

        const allTraces = [...colDendroTraces, ...rowDendroTraces, heatmapTrace];
        
        Plotly.newPlot('clustergramPlot', allTraces, layout, {responsive: true});
        
        if (clustergramResizeHandler) {
            window.removeEventListener('resize', clustergramResizeHandler);
        }

        clustergramResizeHandler = () => {
            Plotly.Plots.resize('clustergramPlot');
        };

        window.addEventListener('resize', clustergramResizeHandler);
    });
}

function initClustergramPage() {
    const clustergramDiv = document.getElementById('clustergramPlot');
    if (!clustergramDiv) {
        return;
    }
    
    fetchAndRenderClustergram();

    document.getElementById('updateClustergramBtn').addEventListener('click', fetchAndRenderClustergram);
}

function cleanupClustergramPage() {
    if (clustergramResizeHandler) {
        window.removeEventListener('resize', clustergramResizeHandler);
        clustergramResizeHandler = null;
    }
}
