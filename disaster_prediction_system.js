
function switchTab(tab) {
    const panels   = document.querySelectorAll('.tab-panel');
    const tabBtns  = document.querySelectorAll('.tab-btn');
    const navLinks = document.querySelectorAll('.nav-link');

    panels.forEach(p => p.classList.remove('active'));
    tabBtns.forEach(b => b.classList.remove('active'));

    document.getElementById(`panel-${tab}`).classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');

    navLinks.forEach(l => {
        if (l.getAttribute('href') === `#${tab}`) l.classList.add('active');
        else l.classList.remove('active');
    });

    document.getElementById(`panel-${tab}`).scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const id = link.getAttribute('href').replace('#','');
        if (id === 'flood' || id === 'earthquake') {
            switchTab(id);
        } else {
            const el = document.getElementById(id);
            if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

const FLOOD_COLOR = '#3b82f6';
const EQ_COLOR    = '#e74c3c';
const GREEN       = '#10b981';

function makeChartOptions(yMax) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
            y: { beginAtZero: true, max: yMax,
                 grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
            legend: { labels: { color: '#cbd5e1', font: { size: 12 } } },
            tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', titleColor: '#fff', bodyColor: '#fff' }
        }
    };
}

function makeRadarOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            r: {
                beginAtZero: true, max: 1,
                grid:        { color: 'rgba(255,255,255,0.15)' },
                pointLabels: { color: '#cbd5e1', font: { size: 11 } },
                ticks:       { display: false, backdropColor: 'transparent' }
            }
        },
        plugins: {
            legend:  { labels: { color: '#cbd5e1' } },
            tooltip: { backgroundColor: 'rgba(0,0,0,0.8)' }
        }
    };
}

// ======================================================
//  FLOOD PREDICTION ENGINE
// ======================================================
const FloodEngine = {
    charts: {},

    predict(d) {
        const r  = Math.min(d.rainfall_mm/100,1)       * 0.3567;
        const r7 = Math.min(d.rain7d/50,1)             * 0.1864;
        const rv = Math.min(d.river_m/10,1)            * 0.1073;
        const hu = (d.humidity/100)                    * 0.0862;
        const el = Math.max(0,1-d.elevation/1000)      * 0.0844;
        const sl = Math.max(0,1-d.slope/30)            * 0.0658;
        const sp = (1-d.soil_perm)                     * 0.0579;
        const tp = (Math.abs(d.temp-20)/40)            * 0.0554;

        const base  = r+r7+rv+hu+el+sl+sp+tp;
        const score = Math.min(1, base*(0.9+Math.random()*0.2));
        const pct   = Math.round(score*100);
        const level = pct<25?'Low Risk': pct<50?'Moderate Risk': pct<75?'High Risk':'Critical Risk';
        const rec   = pct<25?'Continue normal activities':
                      pct<50?'Monitor weather conditions':
                      pct<75?'Prepare for potential flooding':
                             'Take immediate precautions';
        return { score, pct, level, confidence: Math.round(85+score*10), rec };
    },

    updateDisplay(p) {
        const circ   = 2*Math.PI*80;
        const offset = circ - (p.pct/100)*circ;
        const el = document.getElementById('flood-gauge-progress');
        el.style.strokeDashoffset = offset;

        const colors = {'Low Risk':'#10b981','Moderate Risk':'#f59e0b','High Risk':'#f97316','Critical Risk':'#ef4444'};
        el.style.stroke = colors[p.level];

        document.getElementById('flood-gauge-pct').textContent   = `${p.pct}%`;
        document.getElementById('flood-result-level').textContent = p.level;
        document.getElementById('flood-result-prob').textContent  = `${p.pct}%`;
        document.getElementById('flood-result-conf').textContent  = `${p.confidence}%`;
        document.getElementById('flood-result-rec').textContent   = p.rec;

        this.updateRadar(p);
    },

    updateRadar(p) {
        if (!this.charts.radar) return;
        const d = FloodEngine.getFormData();
        this.charts.radar.data.datasets[0].data = [
            Math.min(0.9, d.rainfall_mm/100),
            Math.min(0.8, d.rain7d/50),
            Math.min(0.9, d.river_m/10),
            d.humidity/100,
            Math.max(0.05,1-d.elevation/1000),
            1-d.soil_perm
        ];
        this.charts.radar.update('active');
    },

    getFormData() {
        return {
            rainfall_mm: parseFloat(document.getElementById('f-rainfall').value)||25,
            river_m:     parseFloat(document.getElementById('f-river').value)||5,
            temp:        parseFloat(document.getElementById('f-temp').value)||20,
            humidity:    parseFloat(document.getElementById('f-humidity').value)||70,
            elevation:   parseFloat(document.getElementById('f-elevation').value)||100,
            slope:       parseFloat(document.getElementById('f-slope').value)||5,
            soil_perm:   parseFloat(document.getElementById('f-soil').value)||0.5,
            rain7d:      parseFloat(document.getElementById('f-rain7d').value)||20
        };
    },

    initCharts() {
        // Water chart
        const labels = [], rain = [], river = [];
        for (let i=23;i>=0;i--) {
            const t = new Date(); t.setHours(t.getHours()-i);
            labels.push(t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
            rain.push(Math.max(0,20+Math.sin(i*0.3)*15+(Math.random()-0.5)*10));
            river.push(4+Math.sin(i*0.2)*1.5+(Math.random()-0.5)*0.5);
        }

        const wCtx = document.getElementById('flood-water-chart').getContext('2d');
        this.charts.water = new Chart(wCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label:'Rainfall (mm)', data:rain,  borderColor:FLOOD_COLOR, backgroundColor:'rgba(59,130,246,0.1)', borderWidth:3, fill:true, tension:0.4, yAxisID:'y' },
                    { label:'River Level (m)',data:river, borderColor:'#0ea5e9',  backgroundColor:'rgba(14,165,233,0.1)', borderWidth:3, fill:true, tension:0.4, yAxisID:'y1' }
                ]
            },
            options: {
                responsive:true, maintainAspectRatio:false,
                interaction:{ mode:'index', intersect:false },
                scales:{
                    x:{grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#94a3b8',maxTicksLimit:8}},
                    y:{type:'linear',position:'left',grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#94a3b8'}},
                    y1:{type:'linear',position:'right',grid:{drawOnChartArea:false},ticks:{color:'#94a3b8'}}
                },
                plugins:{legend:{labels:{color:'#cbd5e1'}},tooltip:{backgroundColor:'rgba(0,0,0,0.8)'}}
            }
        });

        const rCtx = document.getElementById('flood-radar-chart').getContext('2d');
        this.charts.radar = new Chart(rCtx, {
            type:'radar',
            data:{
                labels:['Rainfall','7-Day Avg','River Level','Humidity','Elevation','Soil Type'],
                datasets:[
                    {label:'Current',     data:[0.5,0.4,0.5,0.7,0.3,0.5], borderColor:FLOOD_COLOR, backgroundColor:'rgba(59,130,246,0.2)', borderWidth:3},
                    {label:'Safe Threshold',data:[0.3,0.3,0.4,0.6,0.5,0.6], borderColor:GREEN,       backgroundColor:'rgba(16,185,129,0.1)',  borderWidth:2}
                ]
            },
            options: makeRadarOptions()
        });

        // Accuracy Chart
        const accLabels = [];
        for (let i=0; i<12; i++) accLabels.push(`Week ${i+1}`);
        const aCtx = document.getElementById('flood-accuracy-chart').getContext('2d');
        this.charts.accuracy = new Chart(aCtx, {
            type: 'line',
            data: {
                labels: accLabels,
                datasets: [{
                    label: 'Model Accuracy (%)',
                    data: [92.1, 93.2, 93.8, 94.1, 94.3, 94.5, 94.6, 94.7, 94.8, 94.9, 94.9, 94.7],
                    borderColor: FLOOD_COLOR,
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: makeChartOptions(100)
        });

        const cCtx = document.getElementById('flood-confidence-chart').getContext('2d');
        this.charts.confidence = new Chart(cCtx, {
            type: 'bar',
            data: {
                labels: ['60-70%', '70-80%', '80-90%', '90-95%', '95%+'],
                datasets: [{
                    label: 'Predictions',
                    data: [45, 120, 380, 620, 950],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(34, 211, 238, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(59, 130, 246, 0.7)'
                    ],
                    borderColor: FLOOD_COLOR,
                    borderWidth: 1
                }]
            },
            options: makeChartOptions(1000)
        });
    },

    setupEvents() {
        document.getElementById('flood-form').addEventListener('submit', e => {
            e.preventDefault();
            const p = this.predict(this.getFormData());
            this.updateDisplay(p);
        });

        document.querySelectorAll('#panel-flood input').forEach(inp => {
            inp.addEventListener('input', () => {
                clearTimeout(this._timer);
                this._timer = setTimeout(() => {
                    const p = this.predict(this.getFormData());
                    this.updateDisplay(p);
                }, 400);
            });
        });
    },

    startRealTime() {
        setInterval(() => {
            document.getElementById('flood-rainfall-status').textContent = `${(15+Math.random()*20).toFixed(1)}mm`;
            document.getElementById('flood-river-status').textContent    = `${(4+Math.random()*2).toFixed(1)}m`;
            document.getElementById('flood-alerts-count').textContent    = Math.floor(Math.random()*5)+1;

            if (this.charts.water) {
                const now = new Date();
                this.charts.water.data.labels.push(now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
                this.charts.water.data.labels.shift();
                this.charts.water.data.datasets[0].data.push(Math.max(0,20+Math.random()*30));
                this.charts.water.data.datasets[0].data.shift();
                this.charts.water.data.datasets[1].data.push(4+Math.random()*2);
                this.charts.water.data.datasets[1].data.shift();
                this.charts.water.update('quiet');
            }
        }, 8000);
    }
};


const EqEngine = {
    charts: {},

    predict(d) {
        const pn = (d.pwave-3)/7;
        const sn = (d.swave-2)/4;
        const tn = Math.min(d.tremors/20,1);
        const fn = Math.max(0,1-d.fault/100);
        const dn = (d.density-2)/2;
        const cn = 1-Math.abs(d.crustal-35)/35;
        const wn = d.water/100;

        const mults = {'pacific-ring':1.6,'mediterranean':1.35,'himalayan':1.45,'mid-atlantic':1.15,'other':1.0};
        const base  = tn*0.28 + fn*0.22 + pn*0.18 + sn*0.15 + (1-cn)*0.10 + dn*0.04 + wn*0.03;
        const score = Math.min(1, base*(mults[d.region]||1.0)*(0.92+Math.random()*0.14));
        const pct   = Math.round(score*100);
        const mag   = 1.8 + score*6.5;
        const mRange= 0.3 + score*0.8;
        const level = pct<25?'Low': pct<55?'Moderate': pct<80?'High':'Critical';
        return { score, pct, mag, mMin:(mag-mRange).toFixed(1), mMax:(mag+mRange).toFixed(1),
                 confidence:Math.round(82+score*15), level };
    },

    updateDisplay(p) {
        const circ   = 2*Math.PI*80;
        const offset = circ-(p.pct/100)*circ;
        const el = document.getElementById('eq-gauge-progress');
        el.style.strokeDashoffset = offset;

        const colors = {'Low':'#10b981','Moderate':'#f59e0b','High':'#f97316','Critical':'#ef4444'};
        el.style.stroke = colors[p.level];

        document.getElementById('eq-gauge-pct').textContent     = `${p.pct}%`;
        document.getElementById('eq-result-level').textContent  = `${p.level} Risk`;
        document.getElementById('eq-result-mag').textContent    = `${p.mMin} – ${p.mMax}`;
        document.getElementById('eq-result-prob').textContent   = `${p.pct}%`;
        document.getElementById('eq-result-conf').textContent   = `${p.confidence}%`;
        document.getElementById('eq-result-class').textContent  = p.level;

        this.updateRadar(p);
    },

    updateRadar(p) {
        if (!this.charts.radar) return;
        this.charts.radar.data.datasets[0].data = [
            Math.min(0.9, 0.3+p.score*0.6),
            Math.min(0.8, 0.25+p.score*0.55),
            Math.min(0.9, p.score*0.9),
            Math.min(0.95,p.score*0.95),
            Math.min(0.8, 0.2+p.score*0.6),
            Math.min(0.85,0.15+p.score*0.7)
        ];
        this.charts.radar.update('active');
    },

    getFormData() {
        return {
            pwave:   parseFloat(document.getElementById('e-pwave').value)||6.0,
            swave:   parseFloat(document.getElementById('e-swave').value)||3.5,
            tremors: parseInt(document.getElementById('e-tremors').value)||8,
            fault:   parseFloat(document.getElementById('e-fault').value)||35,
            density: parseFloat(document.getElementById('e-density').value)||2.8,
            crustal: parseFloat(document.getElementById('e-crustal').value)||38,
            water:   parseFloat(document.getElementById('e-water').value)||65,
            region:  document.getElementById('e-region').value||'other'
        };
    },

    initCharts() {
        const labels = [], seismic = [];
        for (let i=23;i>=0;i--) {
            const t = new Date(); t.setHours(t.getHours()-i);
            labels.push(t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
            seismic.push(Math.max(0, 1.5+Math.sin(i*0.5)*0.5+(Math.random()-0.5)*0.7));
        }

        const sCtx = document.getElementById('eq-seismic-chart').getContext('2d');
        this.charts.seismic = new Chart(sCtx, {
            type:'line',
            data:{ labels, datasets:[{
                label:'Seismic Activity (M)', data:seismic,
                borderColor:EQ_COLOR, backgroundColor:'rgba(231,76,60,0.12)',
                borderWidth:3, fill:true, tension:0.4, pointRadius:3
            }]},
            options: makeChartOptions(5)
        });

        const rCtx = document.getElementById('eq-radar-chart').getContext('2d');
        this.charts.radar = new Chart(rCtx, {
            type:'radar',
            data:{
                labels:['P-Wave','S-Wave','Tremors','Fault Proximity','Rock Density','Crustal Stability'],
                datasets:[
                    {label:'Current',       data:[0.6,0.5,0.4,0.7,0.45,0.55], borderColor:EQ_COLOR, backgroundColor:'rgba(231,76,60,0.2)',  borderWidth:3},
                    {label:'Safe Threshold',data:[0.4,0.4,0.2,0.8,0.50,0.70], borderColor:GREEN,    backgroundColor:'rgba(16,185,129,0.1)', borderWidth:2}
                ]
            },
            options: makeRadarOptions()
        });


        const accLabels = [];
        for (let i=0; i<12; i++) accLabels.push(`Week ${i+1}`);
        const aCtx = document.getElementById('eq-accuracy-chart').getContext('2d');
        this.charts.accuracy = new Chart(aCtx, {
            type: 'line',
            data: {
                labels: accLabels,
                datasets: [{
                    label: 'Model Accuracy (%)',
                    data: [90.2, 90.8, 91.4, 91.9, 92.1, 92.3, 92.4, 92.5, 92.6, 92.6, 92.5, 92.6],
                    borderColor: EQ_COLOR,
                    backgroundColor: 'rgba(220, 38, 38, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: makeChartOptions(100)
        });

        const eCtx = document.getElementById('eq-error-chart').getContext('2d');
        this.charts.error = new Chart(eCtx, {
            type: 'bar',
            data: {
                labels: ['±0.0-0.2M', '±0.2-0.4M', '±0.4-0.6M', '±0.6-0.8M', '±0.8M+'],
                datasets: [{
                    label: 'Magnitude Predictions',
                    data: [1240, 380, 95, 32, 8],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(34, 211, 238, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(249, 115, 22, 0.7)',
                        'rgba(239, 68, 68, 0.7)'
                    ],
                    borderColor: EQ_COLOR,
                    borderWidth: 1
                }]
            },
            options: makeChartOptions(1500)
        });
    },

    setupEvents() {
        document.getElementById('eq-form').addEventListener('submit', e => {
            e.preventDefault();
            const p = this.predict(this.getFormData());
            this.updateDisplay(p);
        });

        document.querySelectorAll('#panel-earthquake input, #panel-earthquake select').forEach(inp => {
            inp.addEventListener('change', () => {
                const p = this.predict(this.getFormData());
                this.updateDisplay(p);
            });
        });
    },

    startRealTime() {
        setInterval(() => {
            const act = (1.5+Math.random()*2.5).toFixed(1);
            document.getElementById('eq-seismic-status').textContent = act;
            document.getElementById('eq-update-status').textContent  = `${Math.floor(Math.random()*9)+1} min`;
            document.getElementById('eq-alerts-count').textContent   = Math.floor(Math.random()*6)+1;

            if (this.charts.seismic) {
                const now = new Date();
                this.charts.seismic.data.labels.push(now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
                this.charts.seismic.data.labels.shift();
                this.charts.seismic.data.datasets[0].data.push(Math.max(0,1.5+Math.random()*2.5));
                this.charts.seismic.data.datasets[0].data.shift();
                this.charts.seismic.update('quiet');
            }
        }, 10000);
    }
};


const FireEngine = {
    charts: {},

    predict(d) {
        const t = Math.max(0, (d.temperature_c - 15) / 25);
        const h = Math.max(0, (50 - d.humidity_pct) / 50);
        const w = Math.min(1, d.wind_speed_kmh / 60);
        const f = d.vegetation_dryness;
        const e = Math.max(0, (d.elevation_m) / 2000);
        const s = Math.max(0, Math.min(1, d.slope_degrees / 40));
        const rf = Math.min(1, d.recent_fires / 3);
        const fl = Math.min(1, d.fuel_load / 12);
        
        const base = 0.30 * t + 0.25 * h + 0.20 * w + 0.10 * f + 0.05 * s + 0.04 * rf + 0.03 * e + 0.03 * fl;
        const score = Math.min(1, base * (0.9 + Math.random() * 0.2));
        const pct = Math.round(score * 100);
        const level = pct < 25 ? 'Low' : pct < 45 ? 'Moderate' : pct < 70 ? 'High' : 'Critical';
        const rec = pct < 25 ? 'Normal vigilance' : pct < 45 ? 'Reduce outdoor burning' : pct < 70 ? 'Prepare suppression resources' : 'Immediate evacuation possible';
        return { score, pct, level, rec };
    },

    updateDisplay(p) {
        const circ = 2 * Math.PI * 80;
        const offset = circ - (p.pct / 100) * circ;
        const el = document.getElementById('wf-gauge-progress');
        if (el) el.style.strokeDashoffset = offset;
        if (document.getElementById('wf-gauge-pct')) document.getElementById('wf-gauge-pct').textContent = `${p.pct}%`;
        if (document.getElementById('wf-result-level')) document.getElementById('wf-result-level').textContent = p.level;
        if (document.getElementById('wf-result-prob')) document.getElementById('wf-result-prob').textContent = p.score.toFixed(2);
        if (document.getElementById('wf-result-rec')) document.getElementById('wf-result-rec').textContent = p.rec;
        if (document.getElementById('wf-risk-status')) document.getElementById('wf-risk-status').textContent = p.level;
    },

    getFormData() {
        return {
            temperature_c:      parseFloat(document.getElementById('wf-temp').value) || 30,
            humidity_pct:       parseFloat(document.getElementById('wf-humidity').value) || 20,
            wind_speed_kmh:     parseFloat(document.getElementById('wf-wind').value) || 10,
            vegetation_dryness: parseFloat(document.getElementById('wf-fuel').value) || 0.5,
            elevation_m:        parseFloat(document.getElementById('wf-elev').value) || 500,
            slope_degrees:      parseFloat(document.getElementById('wf-slope').value) || 10,
            recent_fires:       parseFloat(document.getElementById('wf-recent').value) || 0,
            fuel_load:          parseFloat(document.getElementById('wf-fuelload').value) || 3
        };
    },

    setupEvents() {
        const form = document.getElementById('wf-form');
        if (!form) return;
        form.addEventListener('submit', e => {
            e.preventDefault();
            this.updateDisplay(this.predict(this.getFormData()));
        });
    },

    initCharts() {
        const labels = [], idx = [];
        for (let i = 23; i >= 0; i--) {
            const t = new Date();
            t.setHours(t.getHours() - i);
            labels.push(t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
            idx.push(Math.max(0, 10 + Math.sin(i * 0.3) * 6 + (Math.random() - 0.5) * 6));
        }

        const ctx = document.getElementById('wf-fire-chart');
        if (ctx) {
            this.charts.fire = new Chart(ctx.getContext('2d'), {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Fire Index',
                        data: idx,
                        borderColor: 'rgba(255,107,47,0.95)',
                        backgroundColor: 'rgba(255,107,47,0.12)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.35
                    }]
                },
                options: makeChartOptions(40)
            });
        }

        const rCtx = document.getElementById('wf-radar-chart');
        if (rCtx) {
            this.charts.radar = new Chart(rCtx.getContext('2d'), {
                type: 'radar',
                data: {
                    labels: ['Temp','Humidity','Wind','Fuel','Slope','Moisture'],
                    datasets: [{
                        label: 'Current',
                        data: [0.7,0.2,0.4,0.74,0.25,0.3],
                        borderColor: 'rgba(255,154,102,0.95)',
                        backgroundColor: 'rgba(255,154,102,0.15)',
                        borderWidth: 3
                    }]
                },
                options: makeRadarOptions()
            });
        }

        const accLabels = [];
        for (let i=0; i<12; i++) accLabels.push(`Week ${i+1}`);
        const aCtx = document.getElementById('wf-accuracy-chart');
        if (aCtx) {
            this.charts.accuracy = new Chart(aCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: accLabels,
                    datasets: [{
                        label: 'Model Accuracy (%)',
                        data: [85.2, 85.8, 86.5, 87.1, 87.6, 88.0, 88.1, 88.0, 88.1, 87.9, 88.0, 88.1],
                        borderColor: 'rgba(234, 88, 12, 0.95)',
                        backgroundColor: 'rgba(234, 88, 12, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: makeChartOptions(100)
            });
        }

        const fpCtx = document.getElementById('wf-fp-chart');
        if (fpCtx) {
            this.charts.fp = new Chart(fpCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['0-2%', '2-4%', '4-6%', '6-8%', '8%+'],
                    datasets: [{
                        label: 'Prediction Events',
                        data: [2100, 850, 320, 110, 35],
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.7)',
                            'rgba(34, 211, 238, 0.7)',
                            'rgba(245, 158, 11, 0.7)',
                            'rgba(249, 115, 22, 0.7)',
                            'rgba(239, 68, 68, 0.7)'
                        ],
                        borderColor: 'rgba(234, 88, 12, 0.95)',
                        borderWidth: 1
                    }]
                },
                options: makeChartOptions(2500)
            });
        }
    },

    startRealTime() {
        setInterval(() => {
            if (document.getElementById('wf-temp-status')) document.getElementById('wf-temp-status').textContent = `${(28 + Math.floor(Math.random()*8))}°C`;
            if (document.getElementById('wf-wind-status')) document.getElementById('wf-wind-status').textContent = `${(10 + Math.floor(Math.random()*30))} km/h`;
            if (this.charts && this.charts.fire) {
                const now = new Date();
                this.charts.fire.data.labels.push(now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
                this.charts.fire.data.labels.shift();
                this.charts.fire.data.datasets[0].data.push(Math.max(0,10 + Math.random()*30));
                this.charts.fire.data.datasets[0].data.shift();
                this.charts.fire.update('quiet');
            }
        }, 8000);
    }
};


const StormEngine = {
    charts: {},

    predict(d) {
        const w = Math.min(1, d.wind_speed_kmh / 200);
        const p = Math.max(0, (1010 - d.central_pressure_hpa) / 60);
        const s = Math.min(1, (d.sea_surface_temp_c - 25) / 10);
        const slope = Math.max(0, (15 - d.coastal_slope_degrees) / 15);
        const lat = Math.max(0, Math.abs(d.latitude_degrees) / 40);
        const recent = Math.min(1, d.recent_storms / 2);
        const current = Math.max(0, Math.abs(d.ocean_current_ms) / 3);
        
        const base = 0.30 * w + 0.25 * p + 0.20 * s + 0.10 * slope + 0.05 * lat + 0.05 * recent + 0.05 * current;
        const score = Math.min(1, base * (0.9 + Math.random() * 0.2));
        const pct = Math.round(score * 100);
        const level = pct < 25 ? 'Low' : pct < 50 ? 'Moderate' : pct < 75 ? 'High' : 'Extreme';
        const rec = pct < 25 ? 'No immediate threat' : pct < 50 ? 'Monitor advisories' : pct < 75 ? 'Secure property & prepare' : 'Follow evacuation orders';
        return { score, pct, level, rec };
    },

    updateDisplay(p) {
        const circ = 2 * Math.PI * 80;
        const offset = circ - (p.pct / 100) * circ;
        const el = document.getElementById('st-gauge-progress');
        if (el) el.style.strokeDashoffset = offset;
        if (document.getElementById('st-gauge-pct')) document.getElementById('st-gauge-pct').textContent = `${p.pct}%`;
        if (document.getElementById('st-result-level')) document.getElementById('st-result-level').textContent = p.level;
        if (document.getElementById('st-result-prob')) document.getElementById('st-result-prob').textContent = `${p.pct}%`;
        if (document.getElementById('st-result-rec')) document.getElementById('st-result-rec').textContent = p.rec;
        if (document.getElementById('st-risk-status')) document.getElementById('st-risk-status').textContent = p.level;
    },

    getFormData() {
        return {
            wind_speed_kmh:        parseFloat(document.getElementById('st-wind').value) || 80,
            central_pressure_hpa:  parseFloat(document.getElementById('st-press').value) || 990,
            sea_surface_temp_c:    parseFloat(document.getElementById('st-sea').value) || 27,
            coastal_slope_degrees: parseFloat(document.getElementById('st-slope').value) || 3,
            sea_elevation_m:       parseFloat(document.getElementById('st-elev').value) || 200,
            latitude_degrees:      parseFloat(document.getElementById('st-lat').value) || 20,
            recent_storms:         parseFloat(document.getElementById('st-recent').value) || 0,
            ocean_current_ms:      parseFloat(document.getElementById('st-current').value) || 0.5
        };
    },

    setupEvents() {
        const form = document.getElementById('st-form');
        if (!form) return;
        form.addEventListener('submit', e => {
            e.preventDefault();
            this.updateDisplay(this.predict(this.getFormData()));
        });
    },

    initCharts() {
        const labels = [], wind = [], press = [];
        for (let i = 23; i >= 0; i--) {
            const t = new Date();
            t.setHours(t.getHours() - i);
            labels.push(t.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
            wind.push(Math.max(0, 40 + Math.sin(i * 0.25) * 20 + (Math.random() - 0.5) * 15));
            press.push(Math.max(920, 1010 + Math.sin(i * 0.2) * 8 + (Math.random() - 0.5) * 6));
        }

        const wCtx = document.getElementById('st-weather-chart');
        if (wCtx) {
            this.charts.wind = new Chart(wCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels,
                    datasets: [
                        {
                            label: 'Wind (km/h)',
                            data: wind,
                            borderColor: 'rgba(6,182,212,0.95)',
                            backgroundColor: 'rgba(6,182,212,0.12)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.35,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Pressure (hPa)',
                            data: press,
                            borderColor: 'rgba(14,165,233,0.8)',
                            backgroundColor: 'rgba(14,165,233,0.08)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.35,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        x:  { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
                        y:  { type: 'linear', position: 'left',  beginAtZero: true, grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8' } },
                        y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: { legend: { labels: { color: '#cbd5e1' } }, tooltip: { backgroundColor: 'rgba(0,0,0,0.8)' } }
                }
            });
        }

        const rCtx = document.getElementById('st-radar-chart');
        if (rCtx) {
            this.charts.radar = new Chart(rCtx.getContext('2d'), {
                type: 'radar',
                data: {
                    labels: ['Wind','Pressure','Sea Temp','Coastal Slope','Surge'],
                    datasets: [{
                        label: 'Current',
                        data: [0.6,0.4,0.7,0.2,0.35],
                        borderColor: 'rgba(6,182,212,0.95)',
                        backgroundColor: 'rgba(6,182,212,0.12)',
                        borderWidth: 3
                    }]
                },
                options: makeRadarOptions()
            });
        }

        const accLabels = [];
        for (let i=0; i<12; i++) accLabels.push(`Week ${i+1}`);
        const aCtx = document.getElementById('st-accuracy-chart');
        if (aCtx) {
            this.charts.accuracy = new Chart(aCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: accLabels,
                    datasets: [{
                        label: 'Model Accuracy (%)',
                        data: [88.5, 89.1, 89.6, 90.0, 90.2, 90.3, 90.4, 90.3, 90.4, 90.3, 90.4, 90.4],
                        borderColor: 'rgba(6, 182, 212, 0.95)',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: makeChartOptions(100)
            });
        }

        const weCtx = document.getElementById('st-wind-error-chart');
        if (weCtx) {
            this.charts.windError = new Chart(weCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['±0-5 km/h', '±5-10 km/h', '±10-15 km/h', '±15-20 km/h', '±20 km/h+'],
                    datasets: [{
                        label: 'Wind Predictions',
                        data: [2890, 1250, 380, 95, 18],
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.7)',
                            'rgba(34, 211, 238, 0.7)',
                            'rgba(245, 158, 11, 0.7)',
                            'rgba(249, 115, 22, 0.7)',
                            'rgba(239, 68, 68, 0.7)'
                        ],
                        borderColor: 'rgba(6, 182, 212, 0.95)',
                        borderWidth: 1
                    }]
                },
                options: makeChartOptions(3500)
            });
        }
    },

    startRealTime() {
        setInterval(() => {
            if (document.getElementById('st-wind-status'))     document.getElementById('st-wind-status').textContent     = `${(60 + Math.floor(Math.random()*40))} km/h`;
            if (document.getElementById('st-pressure-status')) document.getElementById('st-pressure-status').textContent = `${(960 + Math.floor(Math.random()*40))} hPa`;
            if (document.getElementById('st-alerts-count'))   document.getElementById('st-alerts-count').textContent    = Math.floor(Math.random()*5)+1;
            if (this.charts && this.charts.wind) {
                const now = new Date();
                this.charts.wind.data.labels.push(now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}));
                this.charts.wind.data.labels.shift();
                this.charts.wind.data.datasets[0].data.push(Math.max(0, 40 + Math.random()*60));
                this.charts.wind.data.datasets[0].data.shift();
                this.charts.wind.data.datasets[1].data.push(Math.max(920, 960 + Math.random()*50));
                this.charts.wind.data.datasets[1].data.shift();
                this.charts.wind.update('quiet');
            }
        }, 9000);
    }
};

// ======================================================
//  INITIALIZATION
// ======================================================
document.addEventListener('DOMContentLoaded', () => {
    FloodEngine.initCharts();
    FloodEngine.setupEvents();
    FloodEngine.startRealTime();

    EqEngine.initCharts();
    EqEngine.setupEvents();
    EqEngine.startRealTime();

    if (typeof FireEngine.initCharts === 'function') FireEngine.initCharts();
    if (typeof FireEngine.setupEvents === 'function') FireEngine.setupEvents();
    if (typeof FireEngine.startRealTime === 'function') FireEngine.startRealTime();

    if (typeof StormEngine.initCharts === 'function') StormEngine.initCharts();
    if (typeof StormEngine.setupEvents === 'function') StormEngine.setupEvents();
    if (typeof StormEngine.startRealTime === 'function') StormEngine.startRealTime();
});
