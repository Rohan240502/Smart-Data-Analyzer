// API Configuration
// CHANGE THIS to your actual Render backend URL after deploying (e.g., https://your-app.onrender.com)
const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:5000' 
    : 'https://smart-data-analyzer.onrender.com'; // Replace with your production URL

// Global state to track charts and data
let charts = {};
let latestAnalysis = null;

// UI Elements
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('file');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadLoader = document.getElementById('uploadLoader');
const dashboard = document.getElementById('dashboard');
const trainBtn = document.getElementById('trainBtn');
const targetSelect = document.getElementById('targetSelect');
const predictLoader = document.getElementById('predictLoader');
const predictResults = document.getElementById('predictResults');

// --- Helper: File Name Display ---
fileInput.addEventListener('change', function (e) {
    if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
        document.querySelector('.file-label i').className = 'fa-solid fa-check';
        document.querySelector('.file-label').style.color = '#38bdf8';
    }
});

// --- Main: Upload and Analyze ---
uploadForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // UI state: Loading
    uploadLoader.style.display = 'block';
    dashboard.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        latestAnalysis = data;
        
        renderDashboard(data);
        
        // UI state: Success
        uploadLoader.style.display = 'none';
        dashboard.style.display = 'block';
        dashboard.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error(err);
        alert('Error: ' + err.message);
        uploadLoader.style.display = 'none';
    }
});

// --- Renderer: Dashboard ---
function renderDashboard(data) {
    // 1. Summary Stats
    const summaryBody = document.querySelector('#summaryTable tbody');
    summaryBody.innerHTML = `
        <tr><td><i class="fa-solid fa-layer-group"></i> Rows</td><td>${data.before.rows}</td><td>${data.after.rows}</td></tr>
        <tr><td><i class="fa-solid fa-table-columns"></i> Columns</td><td>${data.before.columns}</td><td>${data.after.columns}</td></tr>
        <tr><td><i class="fa-solid fa-triangle-exclamation"></i> Missing Values</td><td>${data.before.missing_total}</td><td>${data.after.missing_total}</td></tr>
    `;

    // 2. Missing Values Table
    const missingBody = document.querySelector('#missingTable tbody');
    missingBody.innerHTML = '';
    Object.entries(data.missing_by_column).forEach(([col, count]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${col}</td>
            <td class="${count > 0 ? 'negative' : 'positive'}">
                ${count > 0 ? count : '<i class="fa-solid fa-check"></i> 0'}
            </td>
        `;
        missingBody.appendChild(row);
    });

    // 3. Smart Insights
    const insightsGrid = document.getElementById('insightsGrid');
    insightsGrid.innerHTML = '';
    if (data.insights && data.insights.length > 0) {
        data.insights.forEach(insight => {
            const div = document.createElement('div');
            div.className = 'insight-item';
            div.innerHTML = `
                <div class="insight-icon ${insight.color}">
                    <i class="fa-solid ${insight.icon}"></i>
                </div>
                <div class="insight-text">${insight.text}</div>
            `;
            insightsGrid.appendChild(div);
        });
        document.getElementById('insightsCard').style.display = 'block';
    } else {
        document.getElementById('insightsCard').style.display = 'none';
    }

    // 4. Target Selection for Predictor
    targetSelect.innerHTML = '';
    data.columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.textContent = col;
        targetSelect.appendChild(opt);
    });

    // 5. Download Link
    const downloadLink = document.getElementById('downloadLink');
    downloadLink.href = `${API_BASE_URL}/download`;

    // 6. Heatmap
    renderHeatmap(data.heatmap);

    // 7. Charts
    renderCharts(data.charts);
}

// --- Renderer: Heatmap ---
function renderHeatmap(heatmapData) {
    const table = document.getElementById('heatmapTable');
    if (!heatmapData) {
        document.getElementById('heatmapCard').style.display = 'none';
        return;
    }
    
    document.getElementById('heatmapCard').style.display = 'block';
    let html = '<thead><tr><th></th>';
    heatmapData.columns.forEach(col => html += `<th title="${col}">${col}</th>`);
    html += '</tr></thead><tbody>';

    heatmapData.data.forEach((row, i) => {
        html += `<tr><th title="${heatmapData.columns[i]}">${heatmapData.columns[i]}</th>`;
        row.forEach(val => {
            const opacity = Math.max(Math.abs(val), 0.1);
            const colorVar = val >= 0 ? '99, 102, 241' : '244, 63, 94';
            const textColor = opacity > 0.5 ? 'white' : '#e2e8f0';
            html += `<td style="background-color: rgba(${colorVar}, ${opacity}); color: ${textColor};" title="Correlation: ${val}">${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

// --- Renderer: Charts ---
function renderCharts(chartConfig) {
    const container = document.getElementById('chartsContainer');
    container.innerHTML = '';
    
    // Clear old chart instances
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};

    chartConfig.forEach(config => {
        const card = document.createElement('div');
        card.className = 'card chart-card slide-up';
        if (config.grid === 'full') card.style.gridColumn = '1 / -1';
        
        card.innerHTML = `
            <div class="card-header" style="justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <i class="fa-solid fa-chart-pie"></i>
                    <h2>${config.title}</h2>
                </div>
                <button class="download-btn" data-id="${config.id}" data-title="${config.title}">
                    <i class="fa-solid fa-download"></i>
                </button>
            </div>
            <div class="chart-container" style="height: 350px;">
                <canvas id="${config.id}"></canvas>
            </div>
        `;
        container.appendChild(card);

        const ctx = document.getElementById(config.id).getContext('2d');
        let bg, border;
        let grad = ctx.createLinearGradient(0, 0, 0, 400);

        if (config.color === 'gradient') {
            grad.addColorStop(0, 'rgba(99, 102, 241, 0.8)');
            grad.addColorStop(1, 'rgba(168, 85, 247, 0.4)');
            bg = grad; border = '#6366f1';
        } else if (config.color === 'multi') {
            bg = ['#6366f1', '#a855f7', '#ec4899', '#3b82f6', '#14b8a6', '#f59e0b', '#ef4444'];
            border = 'transparent';
        } else {
            grad.addColorStop(0, 'rgba(56, 189, 248, 0.6)');
            grad.addColorStop(1, 'rgba(14, 165, 233, 0.1)');
            bg = grad; border = '#38bdf8';
        }

        charts[config.id] = new Chart(ctx, {
            type: config.type,
            data: {
                labels: config.labels,
                datasets: [{
                    label: 'Values',
                    data: config.data,
                    backgroundColor: bg,
                    borderColor: border,
                    borderWidth: 1,
                    borderRadius: 6,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: config.type === 'doughnut',
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: config.type === 'doughnut' ? {} : {
                    y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    });
}

// --- Logic: AI Predictor ---
trainBtn.addEventListener('click', async function () {
    const target = targetSelect.value;
    
    // UI State: Loading
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<i class="fa-solid fa-hourglass"></i> Thinking...';
    predictLoader.style.display = 'inline-block';
    predictResults.style.display = 'none';

    const body = new FormData();
    body.append('target', target);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: body
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Update Results
        document.getElementById('modelAccuracy').textContent = `Model Accuracy: ${data.accuracy}`;
        
        const list = document.getElementById('featureList');
        list.innerHTML = '';
        data.features.forEach(f => {
            const width = Math.max(f.importance, 5);
            list.insertAdjacentHTML('beforeend', `
                <div class="feature-item">
                    <span class="feature-name" title="${f.name}">${f.name}</span>
                    <div class="feature-bar-container"><div class="feature-bar" style="width: ${width}%"></div></div>
                    <span class="feature-val">${f.importance}%</span>
                </div>
            `);
        });

        // Prediction Chart
        renderPredictionChart(data);
        
        predictResults.style.display = 'block';

    } catch (err) {
        alert(err.message);
    } finally {
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fa-solid fa-robot"></i> Train Model';
        predictLoader.style.display = 'none';
    }
});

function renderPredictionChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    if (window.predictionChartInstance) window.predictionChartInstance.destroy();

    window.predictionChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.predictions.map(p => p.label),
            datasets: [
                { label: 'Actual', data: data.predictions.map(p => p.actual), borderColor: '#10b981', tension: 0.3 },
                { label: 'Predicted', data: data.predictions.map(p => p.predicted), borderColor: '#0ea5e9', borderDash: [5, 5], tension: 0.3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#cbd5e1' } } },
            scales: {
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });
}

// --- Logic: Download Chart as Image ---
document.addEventListener('click', function (e) {
    const btn = e.target.closest('.download-btn');
    if (btn) {
        const canvas = document.getElementById(btn.dataset.id);
        const link = document.createElement('a');
        link.download = btn.dataset.title.replace(/ /g, '_') + '.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }
});
