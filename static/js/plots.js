document.addEventListener('DOMContentLoaded', function() {
    const clustergramDiv = document.getElementById('clustergramPlot');
    if (clustergramDiv) {
        // Function to fetch and render clustergram
        function fetchAndRenderClustergram() {
            const topN = document.getElementById('topNFeatures').value;
            const distanceMetric = document.getElementById('distanceMetric').value;
            const linkageMethod = document.getElementById('linkageMethod').value;
            const yAxisLabel = document.getElementById('yAxisLabel').value;

            fetch('/api/clustergram_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    top_n: topN,
                    distance_metric: distanceMetric,
                    linkage_method: linkageMethod,
                    y_axis_label: yAxisLabel
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
                const heatmapCustomData = data.heatmap_customdata; // New: 2D array for customdata
                const metadataColumnNames = data.metadata_column_names; // New: column names for customdata

                // Update heatmap hovertemplate dynamically
                let hoverTemplate = 
                    '<b>Feature:</b> %{y}<br>' +
                    '<b>Sample:</b> %{x}<br>' +
                    '<b>Z-score:</b> %{z:.2f}<br>';

                // Add metadata fields dynamically
                metadataColumnNames.forEach((colName, index) => {
                    hoverTemplate += `<b>${colName}:</b> %{customdata[${index}]}<br>`;
                });
                hoverTemplate += '<extra></extra>';
                heatmapTrace.hovertemplate = hoverTemplate;

                // Assign customdata to heatmap trace
                heatmapTrace.customdata = heatmapCustomData;

                // Calculate max dendrogram distances
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

                // Get range boundaries
                const xMin = Math.min(...data.heatmap_x);
                const xMax = Math.max(...data.heatmap_x);
                const yMin = Math.min(...data.heatmap_y);
                const yMax = Math.max(...data.heatmap_y);
                
                // Create layout
                const layout = {
                    title: `Clustergram of Top ${data.row_labels.length} Features`,
                    height: 1000,
                    autosize: true,
                    showlegend: false,
                    hovermode: 'closest',
                    margin: { l: 150, r: 50, t: 100, b: 150 },
                    
                    // Column Dendrogram (top)
                    xaxis: {
                        domain: [0.15, 0.85],
                        range: [xMin, xMax],
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        matches: 'x3' // Link to heatmap x-axis
                    },
                    yaxis: {
                        domain: [0.82, 0.98],
                        range: [0, maxColDist * 1.05],
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false
                    },

                    // Row Dendrogram (left)
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
                        matches: 'y3' // Link to heatmap y-axis
                    },

                    // Heatmap
                    xaxis3: {
                        domain: [0.15, 0.85],
                        range: [xMin, xMax],
                        tickvals: data.heatmap_x,
                        ticktext: data.column_labels,
                        tickangle: -90,
                        side: 'bottom',
                        anchor: 'y3' // Explicitly anchor to y3
                    },
                    yaxis3: {
                        domain: [0.02, 0.8],
                        range: [yMin, yMax],
                        tickvals: data.heatmap_y,
                        ticktext: data.row_labels,
                        side: 'right'
                    }
                };

                // Assign axes to traces
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

                // Combine all traces
                const allTraces = [...colDendroTraces, ...rowDendroTraces, heatmapTrace];
                
                // Create plot
                Plotly.newPlot('clustergramPlot', allTraces, layout, {responsive: true});
                
                // Make plot responsive to window resize
                window.addEventListener('resize', () => {
                    Plotly.Plots.resize('clustergramPlot');
                });
            });
        }

        // Initial load of clustergram
        fetchAndRenderClustergram();

        // Event listener for update button
        document.getElementById('updateClustergramBtn').addEventListener('click', fetchAndRenderClustergram);
    }
});
