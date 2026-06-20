/* ===========================================================
   IIoMT Agentic Security — Real-Time Dashboard Logic
   Vanilla JS + Socket.IO client
   =========================================================== */

(function () {
    'use strict';

    // ── Socket.IO Connection ──────────────────────────────────
    const socket = io ? io() : null;

    // ── State ─────────────────────────────────────────────────
    const MAX_CHART_POINTS = 60;
    const chartData = {
        labels: [],
        tauEdge: [],
        tauAgent: [],
        tTtm: []
    };
    let chartCanvas = null;
    let chartCtx = null;

    // ── Helpers ───────────────────────────────────────────────

    /**
     * Format an ISO timestamp into a short locale string.
     * @param {string} ts - ISO 8601 timestamp.
     * @returns {string} Formatted time string.
     */
    function formatTimestamp(ts) {
        const d = new Date(ts);
        return d.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }

    /**
     * Animate a numeric value from its current to a target.
     * @param {HTMLElement} el   - DOM element whose textContent to animate.
     * @param {number} target    - Target number.
     * @param {string} suffix    - Suffix to append (e.g. " ms", "%").
     * @param {number} duration  - Animation duration in ms.
     */
    function animateNumber(el, target, suffix, duration) {
        if (!el) return;
        const start = parseFloat(el.dataset.current || '0');
        const range = target - start;
        const startTime = performance.now();
        duration = duration || 600;

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // ease-out quad
            const eased = 1 - (1 - progress) * (1 - progress);
            const current = start + range * eased;
            el.textContent = (Number.isInteger(target) ? Math.round(current) : current.toFixed(1)) + (suffix || '');
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                el.dataset.current = String(target);
            }
        }
        requestAnimationFrame(step);
    }

    /**
     * Return a CSS color for a given risk score 0-1.
     */
    function riskColor(score) {
        if (score >= 0.7) return 'var(--red)';
        if (score >= 0.4) return 'var(--amber)';
        return 'var(--green)';
    }

    function severityClass(score) {
        if (score >= 0.7) return 'severity-high';
        if (score >= 0.4) return 'severity-medium';
        return 'severity-low';
    }

    function attackTypeClass(type) {
        const t = (type || '').toLowerCase().replace(/[\s-]/g, '');
        if (t.includes('ddos')) return 'ddos';
        if (t.includes('spoof')) return 'spoofing';
        if (t.includes('mitm')) return 'mitm';
        if (t.includes('recon')) return 'recon';
        if (t.includes('dos')) return 'dos';
        return '';
    }

    // ── KPI Cards ─────────────────────────────────────────────

    /**
     * Update the four KPI cards with new metrics data.
     * @param {object} metrics
     */
    function updateKPICards(metrics) {
        // Active threats
        const threatEl = document.getElementById('kpi-threats');
        const threatCard = document.getElementById('kpi-threats-card');
        if (threatEl) {
            const count = metrics.active_threats || 0;
            animateNumber(threatEl, count, '', 500);
            if (threatCard) {
                threatCard.classList.toggle('has-threats', count > 0);
            }
        }

        // Detection latency
        const latencyEl = document.getElementById('kpi-latency');
        if (latencyEl) animateNumber(latencyEl, metrics.tau_edge || 0, ' ms', 500);

        // Response latency
        const responseEl = document.getElementById('kpi-response');
        if (responseEl) animateNumber(responseEl, metrics.t_ttm || 0, ' ms', 500);

        // System health
        const healthEl = document.getElementById('kpi-health');
        if (healthEl) animateNumber(healthEl, metrics.system_health || 0, '%', 500);
    }

    // ── Topology Map (SVG) ────────────────────────────────────

    function statusColor(status) {
        switch ((status || '').toLowerCase()) {
            case 'normal':      return '#00ff88';
            case 'throttled':   return '#ffa500';
            case 'quarantined': return '#ff3366';
            default:            return '#596380';
        }
    }

    /**
     * Render SVG device-node topology.
     * @param {Array} devices
     */
    function updateTopologyMap(devices) {
        const svg = document.getElementById('topology-svg');
        if (!svg || !devices || !devices.length) return;

        const ns = 'http://www.w3.org/2000/svg';
        svg.innerHTML = '';

        const w = svg.clientWidth || 500;
        const h = svg.clientHeight || 260;
        const cx = w / 2;
        const cy = h / 2;

        // Central gateway node
        const gwGroup = document.createElementNS(ns, 'g');

        const gwGlow = document.createElementNS(ns, 'circle');
        gwGlow.setAttribute('cx', cx);
        gwGlow.setAttribute('cy', cy);
        gwGlow.setAttribute('r', 32);
        gwGlow.setAttribute('fill', 'rgba(0,212,255,0.08)');
        gwGroup.appendChild(gwGlow);

        const gwCircle = document.createElementNS(ns, 'circle');
        gwCircle.setAttribute('cx', cx);
        gwCircle.setAttribute('cy', cy);
        gwCircle.setAttribute('r', 22);
        gwCircle.setAttribute('fill', '#12172b');
        gwCircle.setAttribute('stroke', '#00d4ff');
        gwCircle.setAttribute('stroke-width', '2');
        gwGroup.appendChild(gwCircle);

        const gwLabel = document.createElementNS(ns, 'text');
        gwLabel.setAttribute('x', cx);
        gwLabel.setAttribute('y', cy + 4);
        gwLabel.setAttribute('text-anchor', 'middle');
        gwLabel.setAttribute('fill', '#00d4ff');
        gwLabel.setAttribute('font-size', '10');
        gwLabel.setAttribute('font-weight', '700');
        gwLabel.setAttribute('font-family', 'Inter, sans-serif');
        gwLabel.textContent = 'GATEWAY';
        gwGroup.appendChild(gwLabel);

        svg.appendChild(gwGroup);

        // Device nodes around perimeter
        const n = devices.length;
        const rx = Math.min(w, 500) * 0.38;
        const ry = h * 0.36;

        devices.forEach(function (dev, i) {
            const angle = (2 * Math.PI * i) / n - Math.PI / 2;
            const dx = cx + rx * Math.cos(angle);
            const dy = cy + ry * Math.sin(angle);
            const col = statusColor(dev.status);

            // Connection line
            const line = document.createElementNS(ns, 'line');
            line.setAttribute('x1', cx);
            line.setAttribute('y1', cy);
            line.setAttribute('x2', dx);
            line.setAttribute('y2', dy);
            line.setAttribute('stroke', col);
            line.setAttribute('stroke-opacity', '0.2');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,4');
            svg.appendChild(line);

            // Outer glow
            const glow = document.createElementNS(ns, 'circle');
            glow.setAttribute('cx', dx);
            glow.setAttribute('cy', dy);
            glow.setAttribute('r', 24);
            glow.setAttribute('fill', col.replace(')', ',0.08)').replace('rgb', 'rgba'));
            svg.appendChild(glow);

            // Node circle
            const circle = document.createElementNS(ns, 'circle');
            circle.setAttribute('cx', dx);
            circle.setAttribute('cy', dy);
            circle.setAttribute('r', 16);
            circle.setAttribute('fill', '#12172b');
            circle.setAttribute('stroke', col);
            circle.setAttribute('stroke-width', '2');
            circle.style.transition = 'stroke 0.4s ease';
            svg.appendChild(circle);

            // Device label
            const text = document.createElementNS(ns, 'text');
            text.setAttribute('x', dx);
            text.setAttribute('y', dy + 3);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', col);
            text.setAttribute('font-size', '8');
            text.setAttribute('font-weight', '600');
            text.setAttribute('font-family', 'Inter, sans-serif');
            // abbreviate
            const abbr = (dev.name || dev.id || '').replace(/\s+/g, ' ');
            text.textContent = abbr.length > 12 ? abbr.substring(0, 11) + '…' : abbr;
            svg.appendChild(text);

            // Criticality sub-label
            const sub = document.createElementNS(ns, 'text');
            sub.setAttribute('x', dx);
            sub.setAttribute('y', dy + 14);
            sub.setAttribute('text-anchor', 'middle');
            sub.setAttribute('fill', '#596380');
            sub.setAttribute('font-size', '7');
            sub.setAttribute('font-family', 'Inter, sans-serif');
            sub.textContent = (dev.criticality || '').replace('_', ' ');
            svg.appendChild(sub);
        });
    }

    // ── Latency Chart (Canvas) ────────────────────────────────

    function initChart() {
        chartCanvas = document.getElementById('latency-canvas');
        if (!chartCanvas) return;
        chartCtx = chartCanvas.getContext('2d');
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
    }

    function resizeCanvas() {
        if (!chartCanvas) return;
        const rect = chartCanvas.parentElement.getBoundingClientRect();
        chartCanvas.width = rect.width - 48;
        chartCanvas.height = 220;
        drawChart();
    }

    /**
     * Append a latency data point and redraw.
     * @param {object} dp - { tau_edge, tau_agent, t_ttm, timestamp }
     */
    function updateLatencyChart(dp) {
        chartData.labels.push(dp.timestamp || new Date().toISOString());
        chartData.tauEdge.push(dp.tau_edge || 0);
        chartData.tauAgent.push(dp.tau_agent || 0);
        chartData.tTtm.push(dp.t_ttm || 0);

        if (chartData.labels.length > MAX_CHART_POINTS) {
            chartData.labels.shift();
            chartData.tauEdge.shift();
            chartData.tauAgent.shift();
            chartData.tTtm.shift();
        }

        drawChart();
    }

    function drawChart() {
        if (!chartCtx || !chartCanvas) return;
        const ctx = chartCtx;
        const W = chartCanvas.width;
        const H = chartCanvas.height;
        const pad = { top: 20, right: 12, bottom: 30, left: 45 };

        ctx.clearRect(0, 0, W, H);

        const series = [
            { data: chartData.tauEdge,  color: '#00d4ff', label: 'τ_edge' },
            { data: chartData.tauAgent, color: '#6366f1', label: 'τ_agent' },
            { data: chartData.tTtm,     color: '#ffa500', label: 'T_ttm' }
        ];

        // Compute max
        let maxVal = 10;
        series.forEach(function (s) {
            s.data.forEach(function (v) { if (v > maxVal) maxVal = v; });
        });
        maxVal = Math.ceil(maxVal / 10) * 10 + 10;

        const plotW = W - pad.left - pad.right;
        const plotH = H - pad.top - pad.bottom;
        const n = chartData.labels.length;
        if (n < 2) return;

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (plotH / 5) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(W - pad.right, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#596380';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (plotH / 5) * i;
            const val = maxVal - (maxVal / 5) * i;
            ctx.fillText(val.toFixed(0), pad.left - 8, y + 3);
        }

        // X-axis labels (every 10th)
        ctx.textAlign = 'center';
        for (let i = 0; i < n; i += 10) {
            const x = pad.left + (plotW / (n - 1)) * i;
            ctx.fillText(formatTimestamp(chartData.labels[i]).substring(0, 5), x, H - 6);
        }

        // Draw lines with gradient fill
        series.forEach(function (s) {
            if (s.data.length < 2) return;

            ctx.beginPath();
            s.data.forEach(function (v, idx) {
                var x = pad.left + (plotW / (n - 1)) * idx;
                var y = pad.top + plotH - (v / maxVal) * plotH;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            // Gradient fill
            ctx.lineTo(pad.left + plotW, pad.top + plotH);
            ctx.lineTo(pad.left, pad.top + plotH);
            ctx.closePath();
            var grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
            grad.addColorStop(0, s.color.replace(')', ',0.15)').replace('#', 'rgba(').length > 30
                ? s.color + '26'   // hex with alpha
                : 'rgba(0,212,255,0.08)');
            // Simple approach: use hex alpha
            grad.addColorStop(0, hexToRGBA(s.color, 0.12));
            grad.addColorStop(1, hexToRGBA(s.color, 0.0));
            ctx.fillStyle = grad;
            ctx.fill();

            // Stroke line on top
            ctx.beginPath();
            s.data.forEach(function (v, idx) {
                var x = pad.left + (plotW / (n - 1)) * idx;
                var y = pad.top + plotH - (v / maxVal) * plotH;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Last-point dot
            var lastX = pad.left + plotW;
            var lastY = pad.top + plotH - (s.data[s.data.length - 1] / maxVal) * plotH;
            ctx.beginPath();
            ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = s.color;
            ctx.fill();
        });
    }

    function hexToRGBA(hex, alpha) {
        hex = hex.replace('#', '');
        if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
        var r = parseInt(hex.substring(0, 2), 16);
        var g = parseInt(hex.substring(2, 4), 16);
        var b = parseInt(hex.substring(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    // ── Alert Feed ────────────────────────────────────────────

    /**
     * Prepend an alert to the alert feed.
     * @param {object} alert - { id, device_name, attack_type, risk_score,
     *                           explanation, timestamp }
     */
    function addAlert(alert) {
        const feed = document.getElementById('alert-feed');
        if (!feed) return;

        const item = document.createElement('div');
        item.className = 'alert-item ' + severityClass(alert.risk_score);
        item.dataset.alertId = alert.id;

        const riskPct = Math.round((alert.risk_score || 0) * 100);
        const riskFillColor = riskGradient(alert.risk_score);

        item.innerHTML =
            '<div class="alert-item__header">' +
                '<span class="alert-item__device">' + escapeHtml(alert.device_name) + '</span>' +
                '<span class="alert-item__time">' + formatTimestamp(alert.timestamp) + '</span>' +
            '</div>' +
            '<div class="alert-item__type ' + attackTypeClass(alert.attack_type) + '">' +
                escapeHtml(alert.attack_type) +
            '</div>' +
            '<div class="risk-bar-wrapper">' +
                '<div class="risk-bar"><div class="risk-bar__fill" style="width:' + riskPct + '%;background:' + riskFillColor + '"></div></div>' +
                '<span class="risk-bar__label" style="color:' + riskColor(alert.risk_score) + '">' + riskPct + '%</span>' +
            '</div>' +
            '<div class="alert-item__explanation">' + escapeHtml(alert.explanation) + '</div>' +
            '<div class="alert-actions">' +
                '<button class="btn btn-approve" onclick="Dashboard.handleOverride(\'' + alert.id + '\',\'approve\')">Approve</button>' +
                '<button class="btn btn-reject"  onclick="Dashboard.handleOverride(\'' + alert.id + '\',\'reject\')">Reject</button>' +
                '<button class="btn btn-escalate" onclick="Dashboard.handleOverride(\'' + alert.id + '\',\'escalate\')">Escalate</button>' +
            '</div>';

        feed.insertBefore(item, feed.firstChild);

        // Cap feed length
        while (feed.children.length > 50) {
            feed.removeChild(feed.lastChild);
        }
    }

    function riskGradient(score) {
        if (score >= 0.7) return 'linear-gradient(90deg, #ffa500, #ff3366)';
        if (score >= 0.4) return 'linear-gradient(90deg, #00ff88, #ffa500)';
        return 'linear-gradient(90deg, #00d4ff, #00ff88)';
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str || ''));
        return div.innerHTML;
    }

    // ── Mitigation Log ────────────────────────────────────────

    /**
     * Add a row to the mitigation log table.
     * @param {object} entry - { timestamp, device_name, action, level, status }
     */
    function updateMitigationLog(entry) {
        var tbody = document.getElementById('mitigation-tbody');
        if (!tbody) return;

        var tr = document.createElement('tr');
        tr.style.animation = 'slide-in .3s ease-out';

        var statusCls = 'pending';
        if (entry.status === 'active') statusCls = 'active';
        else if (entry.status === 'rolled_back') statusCls = 'rolled-back';

        tr.innerHTML =
            '<td>' + formatTimestamp(entry.timestamp) + '</td>' +
            '<td>' + escapeHtml(entry.device_name) + '</td>' +
            '<td>' + escapeHtml(entry.action) + '</td>' +
            '<td>L' + (entry.level != null ? entry.level : '-') + '</td>' +
            '<td><span class="status-badge ' + statusCls + '">' + escapeHtml(entry.status) + '</span></td>';

        tbody.insertBefore(tr, tbody.firstChild);

        while (tbody.children.length > 50) {
            tbody.removeChild(tbody.lastChild);
        }
    }

    // ── Override Handler ──────────────────────────────────────

    /**
     * Send clinician override action to server.
     * @param {string} alertId
     * @param {string} action - 'approve' | 'reject' | 'escalate'
     */
    function handleOverride(alertId, action) {
        fetch('/api/override', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                alert_id: alertId,
                action: action,
                operator: 'clinician_dashboard',
                timestamp: new Date().toISOString()
            })
        })
        .then(function (res) { return res.json(); })
        .then(function (data) {
            // Flash the alert item
            var el = document.querySelector('[data-alert-id="' + alertId + '"]');
            if (el) {
                el.style.opacity = '0.5';
                setTimeout(function () { el.style.opacity = '1'; }, 400);
            }
            console.log('[HITL] Override response:', data);
        })
        .catch(function (err) {
            console.error('[HITL] Override error:', err);
        });
    }

    // ── Clock ─────────────────────────────────────────────────

    function updateClock() {
        var el = document.getElementById('header-time');
        if (el) {
            el.textContent = new Date().toLocaleString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            });
        }
    }

    // ── Initialization ────────────────────────────────────────

    function initDashboard() {
        updateClock();
        setInterval(updateClock, 1000);
        initChart();

        // Fetch initial data
        fetch('/api/metrics')
            .then(function (r) { return r.json(); })
            .then(function (data) { updateKPICards(data); })
            .catch(function (e) { console.error('[HITL] Failed to load metrics:', e); });

        fetch('/api/devices')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                updateTopologyMap(data);
                var countEl = document.getElementById('device-count');
                if (countEl) countEl.textContent = data.length;
            })
            .catch(function (e) { console.error('[HITL] Failed to load devices:', e); });

        fetch('/api/alerts')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                // add in reverse so newest is on top
                data.reverse().forEach(function (a) { addAlert(a); });
            })
            .catch(function (e) { console.error('[HITL] Failed to load alerts:', e); });

        fetch('/api/history')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                data.reverse().forEach(function (e) { updateMitigationLog(e); });
            })
            .catch(function (e) { console.error('[HITL] Failed to load history:', e); });

        // SocketIO events
        if (socket) {
            socket.on('connect', function () {
                console.log('[HITL] SocketIO connected');
            });

            socket.on('metrics_update', function (data) {
                updateKPICards(data);
                updateLatencyChart(data);
            });

            socket.on('new_alert', function (data) {
                addAlert(data);
                // update threat count
                var el = document.getElementById('kpi-threats');
                if (el) {
                    var cur = parseInt(el.dataset.current || '0', 10);
                    animateNumber(el, cur + 1, '', 300);
                }
            });

            socket.on('mitigation_update', function (data) {
                updateMitigationLog(data);
                // Refresh devices
                fetch('/api/devices')
                    .then(function (r) { return r.json(); })
                    .then(function (d) { updateTopologyMap(d); });
            });

            socket.on('device_update', function (data) {
                updateTopologyMap(data);
            });
        }
    }

    // ── Boot ──────────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDashboard);
    } else {
        initDashboard();
    }

    // Expose override handler globally for inline onclick
    window.Dashboard = {
        handleOverride: handleOverride
    };

})();
