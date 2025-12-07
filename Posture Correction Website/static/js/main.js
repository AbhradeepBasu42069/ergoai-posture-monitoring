const app = {
    socket: null,
    chart: null,
    gauge: null,
    isCameraOn: true,

    init() {
        this.initSocket();
        this.initCharts();
        this.startSessionTimer();
    },

    initSocket() {
        this.socket = io.connect('http://' + document.domain + ':' + location.port);

        this.socket.on('connect', () => {
            this.updateStatus('online');
            this.toast('System Online', 'success');
        });

        this.socket.on('disconnect', () => {
            this.updateStatus('offline');
            this.toast('Connection Lost', 'error');
        });

        this.socket.on('stats_update', (data) => this.onDataUpdate(data));
    },

    initCharts() {
        // Line Chart for Trends
        backgroundColor: ['#00ff9d', 'rgba(255,255,255,0.05)'],
            borderWidth: 0,
                circumference: 360,
                    rotation: 0
    }]
},
    options: {
        cutout: '85%',
        plugins: { tooltip: { enabled: false } },
        animation: { duration: 500 }
    }
        });
    },

onDataUpdate(data) {
    // --- DASHBOARD UPDATES ---
    const score = Math.round(data.score);

    // Update Values
    document.getElementById('score-val').innerText = score;
    document.getElementById('val-lean').innerText = data.metrics.body_lean + "°";
    document.getElementById('val-tilt').innerText = data.metrics.head_tilt + "°";
    document.getElementById('val-asym').innerText = data.metrics.shoulder_asym + "px";

    // Update Progress Bars
    this.setBar('bar-lean', data.metrics.body_lean, 30);
    this.setBar('bar-tilt', data.metrics.head_tilt, 40);
    this.setBar('bar-asym', data.metrics.shoulder_asym, 50);

    // Update Gauge Color
    let color = '#00ff9d'; // Success
    let statusText = 'EXCELLENT';
    let statusClass = 'good';

    if (score < 75) { color = '#ffaa00'; statusText = 'WARNING'; statusClass = 'warn'; }
    if (score < 50) { color = '#ff0055'; statusText = 'BAD POSTURE'; statusClass = 'bad'; }

    this.gauge.data.datasets[0].backgroundColor[0] = color;
    this.gauge.data.datasets[0].data[0] = score;
    this.gauge.data.datasets[0].data[1] = 100 - score;
    this.gauge.update();

    // Update Status Badge
    const badge = document.getElementById('status-text');
    badge.className = `status-badge ${statusClass}`;
    badge.innerHTML = `<i class="fas fa-circle"></i> ${statusText}`;

    // Update Line Chart
    const chartData = this.chart.data.datasets[0].data;
    chartData.shift();
    chartData.push(score);
    this.chart.update('none');
},

// --- NAVIGATION ---
nav(viewId) {
    // 1. Update Menu
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');

    // 2. Update View
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.getElementById('view-' + viewId).classList.add('active');

    // 3. Update Text
    document.getElementById('page-title').innerText = viewId.charAt(0).toUpperCase() + viewId.slice(1);
    document.getElementById('bread-current').innerText = viewId.charAt(0).toUpperCase() + viewId.slice(1);
},

// --- ACTIONS ---
toggleCamera() {
    this.isCameraOn = !this.isCameraOn;
    this.socket.emit('toggle_camera', { active: this.isCameraOn });
    const btn = document.getElementById('btn-cam');
    if (this.isCameraOn) {
        btn.innerHTML = '<i class="fas fa-video-slash"></i> Stop Feed';
        btn.className = 'btn danger';
    } else {
        btn.innerHTML = '<i class="fas fa-play"></i> Start Feed';
        btn.className = 'btn secondary';
    }
},

takeSnapshot() {
    const link = document.createElement('a');
    link.download = `ergo-snap-${Date.now()}.jpg`;
    link.href = document.getElementById('video-feed').src;
    // Note: For MJPEG stream, this might download the stream source. 
    // Better to use canvas if cross-origin allows, but let's stick to simple href for now or use the canvas method if needed.

    // Canvas method for reliability:
    const img = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    canvas.width = 1280; canvas.height = 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    try {
        link.href = canvas.toDataURL('image/jpeg');
        link.click();
        this.toast('Snapshot Saved', 'success');
    } catch (e) {
        console.error(e);
        this.toast('Snapshot failed (CORS)', 'error');
    }
},

// --- UTILS ---
updateStatus(state) {
    const el = document.getElementById('sys-status');
    if (state === 'online') {
        el.innerHTML = '<i class="fas fa-wifi"></i> SYSTEM ONLINE';
        el.style.color = 'var(--success)';
        el.style.borderColor = 'rgba(0, 255, 157, 0.2)';
        el.style.background = 'rgba(0, 255, 157, 0.05)';
    } else {
        el.innerHTML = '<i class="fas fa-exclamation-triangle"></i> OFFLINE';
        el.style.color = 'var(--danger)';
        el.style.borderColor = 'rgba(255, 0, 85, 0.2)';
        el.style.background = 'rgba(255, 0, 85, 0.05)';
    }
},

toast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'error') icon = 'times-circle';

    toast.innerHTML = `<i class="fas fa-${icon}"></i> ${msg}`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
},

startSessionTimer() {
    let seconds = 0;
    setInterval(() => {
        seconds++;
        const date = new Date(0);
        date.setSeconds(seconds);
        document.getElementById('session-timer').innerText = date.toISOString().substr(11, 8);
    }, 1000);
}
};

document.addEventListener('DOMContentLoaded', () => app.init());
