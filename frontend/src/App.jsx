import { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const API = 'http://localhost:8000';

/* ─── Gauge SVG ─────────────────────────────────────────────────────────── */
function Gauge({ probability, riskLevel }) {
  const colors = { LOW: '#2ed573', MEDIUM: '#ffd93d', HIGH: '#ffa502', CRITICAL: '#ff4757' };
  const color = colors[riskLevel] || '#2ed573';
  const pct = Math.min(Math.max(probability, 0), 1);
  const angle = pct * 180;
  const r = 70, cx = 80, cy = 80;
  const toRad = (deg) => (deg * Math.PI) / 180;
  const arcX = cx + r * Math.cos(toRad(180 + angle));
  const arcY = cy + r * Math.sin(toRad(180 + angle));
  const largeArc = angle > 180 ? 1 : 0;

  return (
    <div className="gauge">
      <svg viewBox="0 0 160 90">
        {/* track */}
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" strokeLinecap="round" />
        {/* fill */}
        {pct > 0 && (
          <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 ${largeArc} 1 ${arcX} ${arcY}`}
            fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
        )}
        {/* needle */}
        <circle cx={arcX} cy={arcY} r="5" fill={color}
          style={{ filter: `drop-shadow(0 0 5px ${color})` }} />
      </svg>
      <div className="gauge-label" style={{ color }}>
        {(probability * 100).toFixed(1)}%
      </div>
    </div>
  );
}

/* ─── Stepper Input ───────────────────────────────────────────────────────── */
function StepperInput({ label, name, value, onChange, isRupee = false, step = 100 }) {
  const dec = () => onChange(name, Math.max(0, value - step));
  const inc = () => onChange(name, value + step);

  return (
    <div className="stepper-group">
      <label>{label}</label>
      <div className="stepper-row">
        <div className="stepper-display">
          {isRupee && <span className="stepper-prefix">₹</span>}
          <input
            type="number"
            name={name}
            value={value}
            min="0"
            step={step}
            onChange={(e) => onChange(name, parseFloat(e.target.value) || 0)}
          />
        </div>
        <button type="button" className="stepper-btn" onClick={dec} aria-label="decrease">−</button>
        <button type="button" className="stepper-btn" onClick={inc} aria-label="increase">+</button>
      </div>
    </div>
  );
}

/* ─── Prediction Form ─────────────────────────────────────────────────────── */
function PredictionSection() {
  const [form, setForm] = useState({
    step: 1,
    type: 'PAYMENT',
    amount: 1000,
    oldbalanceOrg: 10000,
    newbalanceOrig: 9000,
    oldbalanceDest: 0,
    newbalanceDest: 0,
    isFlaggedFraud: 0,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleType = (e) => setForm((p) => ({ ...p, type: e.target.value }));
  const handleStepper = (name, val) => setForm((p) => ({ ...p, [name]: val }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Error');
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const isFraud = result?.is_fraud;

  return (
    <section className="section" id="predict">
      <div className="section-header">
        <span className="section-tag">🔍 Live Analysis</span>
        <h2>Transaction Risk Analyzer</h2>
        <p>Enter transaction details to instantly detect fraud probability using our XGBoost model.</p>
      </div>
      <div className="form-container">
        <form onSubmit={submit}>
          <div className="glass form-card">
            <h3>Transaction Details</h3>
            <p>Fill in the transaction fields below — feature engineering runs automatically.</p>

            <div className="stepper-form">
              {/* Transaction Type */}
              <div className="stepper-group">
                <label>Transaction Type</label>
                <div className="stepper-row select-only">
                  <select name="type" id="type-select" value={form.type} onChange={handleType}>
                    {['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'].map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>

              <StepperInput label="Amount" name="amount" value={form.amount}
                onChange={handleStepper} isRupee step={500} />
              <StepperInput label="Old Balance (Sender)" name="oldbalanceOrg" value={form.oldbalanceOrg}
                onChange={handleStepper} isRupee step={1000} />
              <StepperInput label="New Balance (Sender)" name="newbalanceOrig" value={form.newbalanceOrig}
                onChange={handleStepper} isRupee step={1000} />
              <StepperInput label="Old Balance (Receiver)" name="oldbalanceDest" value={form.oldbalanceDest}
                onChange={handleStepper} isRupee step={1000} />
              <StepperInput label="New Balance (Receiver)" name="newbalanceDest" value={form.newbalanceDest}
                onChange={handleStepper} isRupee step={1000} />
            </div>

            <button id="analyze-btn" type="submit" className="btn-predict" disabled={loading}>
              {loading && <span className="spinner" />}
              {loading ? 'Analyzing…' : 'Predict'}
            </button>

            {error && (
              <div style={{ marginTop: 16, padding: '14px 18px', background: 'rgba(255,71,87,0.1)', border: '1px solid rgba(255,71,87,0.3)', borderRadius: 10, color: '#ff4757', fontSize: '0.9rem' }}>
                ⚠️ {error}. Make sure the backend is running at {API}
              </div>
            )}

            {result && (
              <div className={`glass result-card ${isFraud ? 'fraud' : 'safe'}`}>
                <div className="result-header">
                  <div className={`result-icon ${isFraud ? 'fraud' : 'safe'}`}>
                    {isFraud ? '🚨' : '✅'}
                  </div>
                  <div>
                    <div className="result-title" style={{ color: isFraud ? '#ff4757' : '#2ed573' }}>
                      {isFraud ? 'Fraudulent Transaction Detected' : 'Transaction Appears Legitimate'}
                    </div>
                    <div className="result-subtitle">
                      Risk Level: <span className={`risk-badge risk-${result.risk_level}`}>{result.risk_level}</span>
                    </div>
                  </div>
                </div>

                <div className="gauge-wrapper">
                  <div>
                    <Gauge probability={result.probability} riskLevel={result.risk_level} />
                    <div style={{ textAlign: 'center', fontSize: '0.72rem', color: 'var(--muted)', marginTop: 4 }}>
                      Fraud Probability
                    </div>
                  </div>
                  <div className="result-details">
                    <h4>Engineered Features Used</h4>
                    <div className="detail-grid">
                      {Object.entries(result.features_used).slice(0, 8).map(([k, v]) => (
                        <div key={k} className="detail-item">
                          <div className="key">{k.replace(/_/g, ' ')}</div>
                          <div className="val">{typeof v === 'number' ? v.toFixed(2) : v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}

/* ─── Metrics Section ─────────────────────────────────────────────────────── */
function MetricsSection() {
  const [metrics, setMetrics] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setErr('Backend offline — start uvicorn to see live metrics.'));
  }, []);

  const cards = metrics
    ? [
        { icon: '🎯', label: 'Accuracy', val: `${metrics.accuracy}%`, sub: `on ${(metrics.sample_size / 1000).toFixed(0)}k samples` },
        { icon: '📈', label: 'ROC-AUC', val: metrics.roc_auc, sub: 'Area under ROC curve' },
        { icon: '⚡', label: 'F1 Score', val: metrics.f1_score, sub: 'Harmonic mean P/R' },
        { icon: '🔍', label: 'Avg Precision', val: metrics.avg_precision, sub: 'Precision-recall AUC' },
        { icon: '🚨', label: 'Fraud Rate', val: `${metrics.fraud_rate}%`, sub: 'In training sample' },
      ]
    : [];

  const fiEntries = metrics
    ? Object.entries(metrics.feature_importances || {}).sort((a, b) => b[1] - a[1]).slice(0, 10)
    : [];

  const chartData = {
    labels: fiEntries.map(([k]) => k.replace(/_/g, ' ')),
    datasets: [{
      label: 'Importance',
      data: fiEntries.map(([, v]) => v),
      backgroundColor: fiEntries.map((_, i) =>
        `hsla(${190 - i * 14}, 80%, 60%, ${0.9 - i * 0.05})`
      ),
      borderRadius: 6,
      borderSkipped: false,
    }],
  };

  const chartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => ` ${(ctx.parsed.x * 100).toFixed(2)}%` } },
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892a4', font: { size: 11 } } },
      y: { grid: { display: false }, ticks: { color: '#e4e8f0', font: { size: 11, weight: '600' } } },
    },
  };

  return (
    <section className="section" id="metrics">
      <div className="section-header">
        <span className="section-tag">📊 Performance</span>
        <h2>Model Metrics</h2>
        <p>XGBoost trained on 200k PaySim transactions with class-balanced weights.</p>
      </div>

      {err && (
        <div style={{ textAlign: 'center', color: 'var(--muted)', padding: '40px', background: 'rgba(255,255,255,0.02)', borderRadius: 12, border: '1px dashed rgba(255,255,255,0.1)' }}>
          {err}
        </div>
      )}

      {metrics && (
        <>
          <div className="metrics-grid">
            {cards.map(({ icon, label, val, sub }) => (
              <div key={label} className="glass metric-card">
                <div className="metric-icon">{icon}</div>
                <div className="metric-val gradient-text">{val}</div>
                <div className="metric-label">{label}</div>
                <div className="metric-sub">{sub}</div>
              </div>
            ))}
          </div>

          <div className="two-col">
            <div className="glass chart-card">
              <h3>🔑 Feature Importances (Top 10)</h3>
              <div style={{ height: 320 }}>
                <Bar data={chartData} options={chartOptions} />
              </div>
            </div>

            <div className="glass chart-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <h3 style={{ marginBottom: 24 }}>🧮 Confusion Matrix</h3>
              {(() => {
                const cm = metrics.confusion_matrix || {};
                return (
                  <div className="cm-grid">
                    <div className="cm-cell tn">
                      <div className="cm-val">{cm.tn?.toLocaleString()}</div>
                      <div className="cm-label">True Negative</div>
                    </div>
                    <div className="cm-cell fp">
                      <div className="cm-val">{cm.fp?.toLocaleString()}</div>
                      <div className="cm-label">False Positive</div>
                    </div>
                    <div className="cm-cell fn">
                      <div className="cm-val">{cm.fn?.toLocaleString()}</div>
                      <div className="cm-label">False Negative</div>
                    </div>
                    <div className="cm-cell tp">
                      <div className="cm-val">{cm.tp?.toLocaleString()}</div>
                      <div className="cm-label">True Positive</div>
                    </div>
                  </div>
                );
              })()}
              <div style={{ marginTop: 24, fontSize: '0.8rem', color: 'var(--muted)', textAlign: 'center' }}>
                Test set: {metrics.sample_size ? (metrics.sample_size * 0.2 / 1000).toFixed(0) : '—'}k transactions
              </div>
            </div>
          </div>
        </>
      )}

      {!metrics && !err && (
        <div style={{ textAlign: 'center', padding: '60px', color: 'var(--muted)' }}>
          <span className="spinner" /> Loading metrics…
        </div>
      )}
    </section>
  );
}

/* ─── EDA Gallery ─────────────────────────────────────────────────────────── */
function EDASection() {
  const images = [
    { src: `${API}/static/01_eda_overview.png`, title: 'EDA Overview', desc: 'Transaction type distributions, amount histograms, and fraud vs. legitimate balance patterns.' },
    { src: `${API}/static/02_correlation_heatmap.png`, title: 'Correlation Heatmap', desc: 'Feature correlation matrix revealing key relationships between transaction attributes.' },
    { src: `${API}/static/03_xgboost_performance.png`, title: 'XGBoost Performance', desc: 'ROC curve, precision-recall curve, and feature importance rankings from the trained model.' },
  ];

  return (
    <section className="section" id="eda">
      <div className="section-header">
        <span className="section-tag">🔬 Exploratory Data Analysis</span>
        <h2>Data Insights</h2>
        <p>Key visualizations from the PaySim dataset analysis powering this model.</p>
      </div>
      <div className="eda-grid">
        {images.map(({ src, title, desc }) => (
          <div key={title} className="glass eda-card">
            <div className="eda-img-wrap">
              <img src={src} alt={title} loading="lazy" />
            </div>
            <div className="eda-caption">
              <h4>{title}</h4>
              <p>{desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ─── Hero ─────────────────────────────────────────────────────────────────── */
function Hero() {
  return (
    <section className="hero" id="home">
      <div className="hero-bg" />
      <div className="hero-grid" />
      <div className="hero-content">
        <div className="badge">
          <span className="badge-dot" />
          XGBoost Model Active
        </div>
        <h1>
          <span className="gradient-text">FraudShield</span>
          <br />
          AI Detection System
        </h1>
        <p>
          Real-time bank transaction fraud detection powered by XGBoost trained on the PaySim dataset.
          Sub-millisecond inference with 99%+ accuracy.
        </p>
        <div className="hero-stats">
          {[
            { num: '6.3M+', label: 'Transactions Analyzed' },
            { num: '99%+', label: 'Detection Accuracy' },
            { num: '15', label: 'Engineered Features' },
            { num: '<1ms', label: 'Inference Time' },
          ].map(({ num, label }) => (
            <div key={label} className="hero-stat">
              <span className="num">{num}</span>
              <span className="label">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ─── App ──────────────────────────────────────────────────────────────────── */
export default function App() {
  return (
    <>
      <nav className="nav" role="navigation" aria-label="Main navigation">
        <div className="nav-logo">
          <div className="nav-logo-icon">🛡️</div>
          FraudShield AI
        </div>
        <ul className="nav-links">
          {[['#predict', 'Predict'], ['#metrics', 'Metrics'], ['#eda', 'EDA']].map(([href, label]) => (
            <li key={href}><a href={href}>{label}</a></li>
          ))}
        </ul>
      </nav>

      <main>
        <Hero />
        <div className="divider" />
        <PredictionSection />
        <div className="divider" />
        <MetricsSection />
        <div className="divider" />
        <EDASection />
      </main>

      <footer className="footer">
        <p>FraudShield AI — XGBoost on PaySim Dataset · Built with FastAPI + React/Vite</p>
      </footer>
    </>
  );
}
