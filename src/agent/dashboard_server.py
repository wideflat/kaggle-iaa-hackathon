"""
Real-time dashboard server for agent visualization

Uses FastAPI with WebSocket for live updates.

Usage:
    # Start server only
    python -m src.agent.dashboard_server

    # Server runs on http://localhost:8765
    # Dashboard at http://localhost:8765/
"""

import os
import asyncio
import json
import webbrowser
from pathlib import Path
from typing import List, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.agent.event_emitter import get_emitter, AgentEvent


# Connected WebSocket clients
connected_clients: Set[WebSocket] = set()

# Event queue for broadcasting
event_queue: asyncio.Queue = asyncio.Queue(maxsize=100)


async def broadcast_events():
    """Background task to broadcast events to all connected clients"""
    while True:
        try:
            event: AgentEvent = await event_queue.get()
            event_json = event.to_json()

            # Broadcast to all connected clients
            disconnected = set()
            for client in connected_clients:
                try:
                    await client.send_text(event_json)
                except Exception:
                    disconnected.add(client)

            # Remove disconnected clients
            for client in disconnected:
                connected_clients.discard(client)

        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Start broadcast task
    task = asyncio.create_task(broadcast_events())

    # Set up event emitter queue
    emitter = get_emitter()
    emitter.set_queue(event_queue)

    yield

    # Cleanup
    task.cancel()


app = FastAPI(lifespan=lifespan)


# Serve static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve dashboard HTML"""
    html_path = static_dir / "dashboard.html"
    if html_path.exists():
        return FileResponse(html_path)

    # Inline fallback if file doesn't exist
    return HTMLResponse(content=get_inline_dashboard(), status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)

    # Send event history to new client
    emitter = get_emitter()
    for event in emitter.get_history():
        try:
            await websocket.send_text(event.to_json())
        except Exception:
            break

    try:
        while True:
            # Keep connection alive, handle incoming messages if needed
            data = await websocket.receive_text()
            # Could handle client commands here
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


@app.get("/api/status")
async def get_status():
    """Get current agent status (for polling fallback)"""
    emitter = get_emitter()
    history = emitter.get_history()

    if not history:
        return {"status": "idle", "events": []}

    last_event = history[-1]
    return {
        "status": last_event.event_type,
        "events": [e.to_json() for e in history[-20:]]  # Last 20 events
    }


def get_inline_dashboard() -> str:
    """Inline dashboard HTML (fallback if file doesn't exist)"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background: #16213e; border-radius: 8px; }
        .status-item { flex: 1; }
        .status-label { font-size: 12px; color: #888; }
        .status-value { font-size: 24px; font-weight: bold; }
        .status-value.running { color: #00d4ff; }
        .status-value.success { color: #00ff88; }
        .chart-container { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .features-container { background: #16213e; border-radius: 8px; padding: 20px; }
        .feature { padding: 10px; border-bottom: 1px solid #2a2a4a; font-family: monospace; font-size: 13px; }
        .feature.success { border-left: 3px solid #00ff88; }
        .feature.failed { border-left: 3px solid #ff4444; opacity: 0.7; }
        .feature-code { color: #aaa; margin-top: 5px; white-space: pre-wrap; }
        .improvement { color: #00ff88; }
        .no-improvement { color: #ff4444; }
        .current-code { background: #0f3460; border-radius: 8px; padding: 15px; margin-bottom: 20px; font-family: monospace; }
        .current-code pre { white-space: pre-wrap; color: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Feature Engineering Agent</h1>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" id="status">Waiting...</div>
            </div>
            <div class="status-item">
                <div class="status-label">Iteration</div>
                <div class="status-value" id="iteration">0 / 0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Best RMSLE</div>
                <div class="status-value success" id="best-rmsle">--</div>
            </div>
            <div class="status-item">
                <div class="status-label">Success Rate</div>
                <div class="status-value" id="success-rate">--</div>
            </div>
        </div>

        <div class="current-code" id="current-code-container" style="display: none;">
            <div class="status-label">Current Feature</div>
            <pre id="current-code"></pre>
        </div>

        <div class="chart-container">
            <canvas id="rmsle-chart" height="200"></canvas>
        </div>

        <div class="features-container">
            <h3 style="margin-bottom: 15px;">Recent Features</h3>
            <div id="features-list"></div>
        </div>
    </div>

    <script>
        let chart;
        let rmsleData = [];
        let iterations = [];
        let features = [];
        let currentIteration = 0;
        let totalIterations = 0;
        let successCount = 0;
        let totalCount = 0;
        let bestRmsle = null;

        // Initialize chart
        const ctx = document.getElementById('rmsle-chart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Actual RMSLE',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0,
                    fill: false,
                    pointRadius: 3
                }, {
                    label: 'Best RMSLE',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0,
                    fill: true,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: '#2a2a4a' },
                        ticks: { color: '#888' }
                    },
                    x: {
                        grid: { color: '#2a2a4a' },
                        ticks: { color: '#888' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#888' } }
                }
            }
        });

        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleEvent(data);
        };

        ws.onclose = () => {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').style.color = '#ff4444';
        };

        function handleEvent(event) {
            const { event_type, data } = event;

            switch (event_type) {
                case 'agent_start':
                    totalIterations = data.total_iterations || 0;
                    document.getElementById('status').textContent = 'Starting...';
                    document.getElementById('status').className = 'status-value running';
                    if (data.baseline_rmsle) {
                        bestRmsle = data.baseline_rmsle;
                        addRmslePoints(0, data.baseline_rmsle, data.baseline_rmsle);
                        document.getElementById('best-rmsle').textContent = data.baseline_rmsle.toFixed(5);
                    }
                    break;

                case 'iteration_start':
                    currentIteration = data.iteration;
                    document.getElementById('status').textContent = 'Generating...';
                    document.getElementById('status').className = 'status-value running';
                    document.getElementById('iteration').textContent = `${currentIteration} / ${totalIterations}`;
                    break;

                case 'feature_generated':
                    document.getElementById('status').textContent = 'Evaluating...';
                    document.getElementById('current-code-container').style.display = 'block';
                    document.getElementById('current-code').textContent = data.code || '';
                    break;

                case 'evaluation_complete':
                    // Wait for accept/reject
                    break;

                case 'feature_accepted':
                    successCount++;
                    totalCount++;
                    bestRmsle = data.new_rmsle;
                    addRmslePoints(currentIteration, data.new_rmsle, data.new_rmsle);
                    document.getElementById('best-rmsle').textContent = data.new_rmsle.toFixed(5);
                    addFeature(data, true);
                    updateSuccessRate();
                    document.getElementById('current-code-container').style.display = 'none';
                    break;

                case 'feature_rejected':
                    totalCount++;
                    // Add actual RMSLE (if evaluated) and current best RMSLE
                    if (data.rmsle && currentIteration > 0) {
                        addRmslePoints(currentIteration, data.rmsle, bestRmsle);
                    }
                    addFeature(data, false);
                    updateSuccessRate();
                    document.getElementById('current-code-container').style.display = 'none';
                    break;

                case 'agent_complete':
                    document.getElementById('status').textContent = 'Complete!';
                    document.getElementById('status').className = 'status-value success';
                    break;
            }
        }

        function addRmslePoints(iteration, actualRmsle, bestRmsle) {
            chart.data.labels.push(iteration.toString());
            chart.data.datasets[0].data.push(actualRmsle);  // Actual RMSLE (dotted)
            chart.data.datasets[1].data.push(bestRmsle);     // Best RMSLE (solid)
            chart.update();
        }

        function addFeature(data, success) {
            const list = document.getElementById('features-list');
            const div = document.createElement('div');
            div.className = `feature ${success ? 'success' : 'failed'}`;

            const icon = success ? '✅' : '❌';
            const delta = data.improvement ? ` (Δ ${data.improvement > 0 ? '-' : '+'}${Math.abs(data.improvement).toFixed(5)})` : '';

            div.innerHTML = `
                <div>${icon} ${data.columns ? data.columns.join(', ') : 'Unknown'}${delta}</div>
                <div class="feature-code">${(data.code || '').split('\\n')[0]}</div>
            `;

            list.insertBefore(div, list.firstChild);

            // Keep only last 10
            while (list.children.length > 10) {
                list.removeChild(list.lastChild);
            }
        }

        function updateSuccessRate() {
            const rate = totalCount > 0 ? ((successCount / totalCount) * 100).toFixed(0) : '--';
            document.getElementById('success-rate').textContent = `${rate}%`;
        }
    </script>
</body>
</html>
"""


def start_server(port: int = 8765, open_browser: bool = True):
    """Start the dashboard server"""
    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    print("Starting dashboard server at http://localhost:8765")
    start_server(port=8765, open_browser=True)
