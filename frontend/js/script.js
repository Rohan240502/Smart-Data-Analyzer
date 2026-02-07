const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:5000' 
    : 'https://datavisionary.onrender.com';

// Global state
let charts = {};
let latestAnalysis = null;

// UI Elements
const uploadSection = document.getElementById('uploadSection');
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('file');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadLoader = document.getElementById('uploadLoader');
const dashboard = document.getElementById('dashboard');
const trainBtn = document.getElementById('trainBtn');
const targetSelect = document.getElementById('targetSelect');
const predictLoader = document.getElementById('predictLoader');
const predictResults = document.getElementById('predictResults');
const dropZone = document.getElementById('dropZone');

// --- Drag & Drop Magic ---
['dragover', 'dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        if (evt === 'dragover') dropZone.style.borderColor = '#a855f7';
        else dropZone.style.borderColor = 'rgba(255,255,255,0.1)';
    });
});

dropZone.addEventListener('drop', (e) => {
    fileInput.files = e.dataTransfer.files;
    updateFileDisplay(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) updateFileDisplay(e.target.files[0]);
});

function updateFileDisplay(file) {
    fileNameDisplay.textContent = file.name;
    document.getElementById('fileIcon').className = 'fa-solid fa-file-circle-check';
    document.getElementById('fileIcon').style.color = '#10b981';
}

// --- Main: Upload and Analyze ---
uploadForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    // ⚡ SAFETY: Free tier servers crash on massive files
    const MAX_SIZE = 20 * 1024 * 1024; // 20MB
    if (file.size > MAX_SIZE) {
        alert('File is too large for the free server tier. Please use a file smaller than 20MB.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    uploadLoader.style.display = 'block';
    const loaderText = uploadLoader.querySelector('p');
    loaderText.textContent = "Uploading & Analyzing... (May take 30-60s)";
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.status === 504) {
            throw new Error('The server timed out. This dataset might be too complex for the free tier, or the server is waking up. Please try again in a moment.');
        }

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.details || errData.error || 'Server error');
        }

        const data = await response.json();
        latestAnalysis = data;
        
        renderDashboard(data);
        
        // Success: Hide upload, show dashboard
        uploadSection.style.display = 'none';
        dashboard.style.display = 'block';
        window.scrollTo({ top: 0, behavior: 'smooth' });

    } catch (err) {
        alert('Analysis Error: ' + err.message);
        console.error('Fetch Error:', err);
    } finally {
        uploadLoader.style.display = 'none';
    }
});

// --- Renderer: Dashboard ---
function renderDashboard(data) {
    // 1. Data Preview (Screen 4)
    renderPreviewTable(data.sample_data);

    // 2. Summary & Missing (Screen 6)
    const summaryBody = document.querySelector('#summaryTable tbody');
    summaryBody.innerHTML = `
        <tr><td><i class="fa-solid fa-rows"></i> Rows</td><td>${data.before.rows}</td><td>${data.after.rows}</td></tr>
        <tr><td><i class="fa-solid fa-columns"></i> Columns</td><td>${data.before.columns}</td><td>${data.after.columns}</td></tr>
        <tr><td><i class="fa-solid fa-circle-exclamation"></i> Missing</td><td>${data.before.missing_total}</td><td>${data.after.missing_total}</td></tr>
    `;

    const missingBody = document.querySelector('#missingTable tbody');
    missingBody.innerHTML = '';
    Object.entries(data.missing_by_column).slice(0, 10).forEach(([col, count]) => {
        missingBody.innerHTML += `<tr><td>${col}</td><td class="${count > 0 ? 'negative' : 'positive'}">${count}</td></tr>`;
    });

    // 3. Smart Insights (Screen 7)
    const insightsGrid = document.getElementById('insightsGrid');
    insightsGrid.innerHTML = '';
    (data.insights || []).forEach(insight => {
        insightsGrid.innerHTML += `
            <div class="insight-item">
                <div class="insight-icon ${insight.color || 'purple'}">
                    <i class="fa-solid ${insight.icon || 'fa-lightbulb'}"></i>
                </div>
                <div class="insight-text" style="font-size: 0.9rem;">${insight.text}</div>
            </div>
        `;
    });

    // 4. Populate Target Selector
    targetSelect.innerHTML = '';
    data.columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col; opt.textContent = col;
        targetSelect.appendChild(opt);
    });

    // 5. Download Link
    document.getElementById('downloadLink').href = `${API_BASE_URL}/download`;

    // 6. Visuals
    renderHeatmap(data.heatmap);
    renderCharts(data.charts);
}

// --- Screen 4: Preview Table ---
function renderPreviewTable(rows) {
    if (!rows || rows.length === 0) return;
    const head = document.getElementById('previewHead');
    const body = document.getElementById('previewBody');
    
    const cols = Object.keys(rows[0]);
    head.innerHTML = cols.map(c => `<th>${c}</th>`).join('');
    body.innerHTML = rows.map(r => `
        <tr>${cols.map(c => `<td>${r[c] !== null ? r[c] : '-'}</td>`).join(' ')}</tr>
    `).join('');
}

// --- Screen 12-14: Model Training ---
trainBtn.addEventListener('click', async () => {
    const target = targetSelect.value;
    predictLoader.style.display = 'inline-block';
    predictResults.style.display = 'none';

    try {
        const fd = new FormData();
        fd.append('target', target);
        const res = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.error) throw new Error(data.error);
        renderPredictionResults(data);
    } catch (err) {
        alert(err.message);
    } finally {
        predictLoader.style.display = 'none';
    }
});

function renderPredictionResults(data) {
    predictResults.style.display = 'block';
    document.getElementById('modelAccuracy').textContent = `${data.model_type}: ${data.accuracy}`;
    
    // Feature Importance Bars (Screen 14)
    const flist = document.getElementById('featureList');
    flist.innerHTML = data.features.map(f => `
        <div class="feature-row">
            <span style="font-size: 0.9rem;">${f.name}</span>
            <span style="font-size: 0.8rem; color: var(--text-secondary);">${f.importance}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width: ${f.importance}%"></div></div>
    `).join('');

    // Prediction Chart
    const ctx = document.getElementById('predictionChart').getContext('2d');
    if (charts.predict) charts.predict.destroy();

    charts.predict = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.predictions.map(p => p.label),
            datasets: [
                { label: 'Actual', data: data.predictions.map(p => p.actual), borderColor: '#a855f7', tension: 0.4 },
                { label: 'Predicted', data: data.predictions.map(p => p.predicted), borderColor: '#0ea5e9', borderDash: [5, 5], tension: 0.4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display : false } }
            }
        }
    });
}

function renderHeatmap(data) {
    const table = document.getElementById('heatmapTable');
    if (!data) return document.getElementById('heatmapCard').style.display = 'none';
    
    document.getElementById('heatmapCard').style.display = 'block';
    let html = '<thead><tr><th></th>' + data.columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    
    data.data.forEach((row, i) => {
        html += `<tr><th>${data.columns[i]}</th>`;
        row.forEach(val => {
            const alpha = Math.abs(val);
            const color = val > 0 ? `rgba(139, 92, 246, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
            html += `<td style="background: ${color}; color: ${alpha > 0.5 ? '#fff' : '#94a3b8'}">${val}</td>`;
        });
        html += '</tr>';
    });
    table.innerHTML = html + '</tbody>';
}

function renderCharts(config) {
    const container = document.getElementById('chartsContainer');
    container.innerHTML = '';
    config.forEach(c => {
        const id = `chart_${Math.random().toString(36).substr(2, 9)}`;
        container.innerHTML += `
            <div class="card chart-card">
                <div class="card-header"><i class="fa-solid fa-chart-area"></i><h2>${c.title}</h2></div>
                <div style="height: 300px;"><canvas id="${id}"></canvas></div>
            </div>
        `;
        setTimeout(() => {
            const ctx = document.getElementById(id).getContext('2d');
            new Chart(ctx, {
                type: c.type,
                data: {
                    labels: c.labels,
                    datasets: [{ data: c.data, backgroundColor: c.color === 'multi' ? ['#a855f7', '#0ea5e9', '#ec4899', '#f59e0b', '#10b981'] : '#a855f7' }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: c.type === 'doughnut', labels: { color: '#94a3b8' } } } }
            });
        }, 10);
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
