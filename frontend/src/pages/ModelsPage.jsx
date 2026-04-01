import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  CartesianGrid, Legend
} from 'recharts'
import { Cpu, Award, TrendingUp, Target } from 'lucide-react'
import { getModelInfo } from '../api'

const modelData = [
  {
    name: 'Logistic Regression',
    type: 'Linear',
    precision: 0.68,
    recall: 0.72,
    f1: 0.70,
    prauc: 0.52,
    rocauc: 0.78,
    threshold: 0.35,
    role: 'Baseline',
    description: 'Simple linear model with balanced class weights. Fast inference but limited capacity for complex fraud patterns.',
    color: '#94a3b8'
  },
  {
    name: 'Random Forest',
    type: 'Ensemble',
    precision: 0.82,
    recall: 0.67,
    f1: 0.74,
    prauc: 0.72,
    rocauc: 0.91,
    threshold: 0.45,
    role: 'Secondary',
    description: 'Bagged ensemble with balanced_subsample weighting. Good generalization but slower inference.',
    color: '#3b82f6'
  },
  {
    name: 'XGBoost',
    type: 'Gradient Boosting',
    precision: 0.91,
    recall: 0.62,
    f1: 0.74,
    prauc: 0.81,
    rocauc: 0.96,
    threshold: 0.58,
    role: 'Primary',
    description: 'Gradient-boosted trees with scale_pos_weight for imbalance handling. Optimized with Optuna (30 trials).',
    color: '#22c55e'
  },
  {
    name: 'LightGBM',
    type: 'Gradient Boosting',
    precision: 0.93,
    recall: 0.64,
    f1: 0.76,
    prauc: 0.84,
    rocauc: 0.97,
    threshold: 0.55,
    role: 'Primary (Best)',
    description: 'Leaf-wise gradient boosting with is_unbalance. Best overall performance. Deployed in production.',
    color: '#8b5cf6'
  },
]

const radarData = modelData.map(m => ({
  model: m.name.split(' ').pop(),
  Precision: m.precision * 100,
  Recall: m.recall * 100,
  F1: m.f1 * 100,
  'PR-AUC': m.prauc * 100,
  'ROC-AUC': m.rocauc * 100,
}))

const tooltipStyle = {
  contentStyle: {
    background: 'rgba(17, 24, 39, 0.95)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '10px',
    fontSize: '0.8rem',
    color: '#f1f5f9',
  },
}

export default function ModelsPage() {
  const [liveModelInfo, setLiveModelInfo] = useState(null)

  useEffect(() => {
    getModelInfo().then(setLiveModelInfo).catch(() => {})
  }, [])

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>Models</h2>
        <p>Compare trained models and view performance metrics</p>
      </div>

      {/* ── Best Model Highlight ── */}
      <div className="card" style={{
        marginBottom: '24px',
        background: 'linear-gradient(135deg, rgba(139,92,246,0.08), rgba(99,102,241,0.04))',
        borderColor: 'rgba(139,92,246,0.2)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{
            width: '56px', height: '56px', borderRadius: 'var(--radius-lg)',
            background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 30px rgba(139,92,246,0.3)',
          }}>
            <Award size={28} color="white" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Production Model
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              LightGBM{' '}
              <span className="badge badge-info">DEPLOYED</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Best PR-AUC of 0.84 · Precision 93.2% · Recall 64.0% · Optimized with Optuna (30 trials, 5-fold CV)
            </div>
          </div>
          <div style={{ display: 'flex', gap: '24px' }}>
            <MetricBox label="PR-AUC" value="0.840" color="#8b5cf6" />
            <MetricBox label="Precision" value="93.2%" color="#22c55e" />
            <MetricBox label="Recall" value="64.0%" color="#14b8a6" />
          </div>
        </div>
      </div>

      {/* ── Charts ── */}
      <div className="charts-grid">
        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Performance Comparison</div>
              <div className="card-subtitle">Metrics across all trained models</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData} barCategoryGap="18%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(v) => v.split(' ').pop()}
              />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip {...tooltipStyle} formatter={(v) => (v * 100).toFixed(1) + '%'} />
              <Legend formatter={(v) => <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{v}</span>} />
              <Bar dataKey="precision" fill="#6366f1" radius={[4, 4, 0, 0]} name="Precision" />
              <Bar dataKey="recall" fill="#14b8a6" radius={[4, 4, 0, 0]} name="Recall" />
              <Bar dataKey="prauc" fill="#f59e0b" radius={[4, 4, 0, 0]} name="PR-AUC" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Radar Comparison</div>
              <div className="card-subtitle">Multi-dimensional model evaluation</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={[
              { metric: 'Precision', LR: 68, RF: 82, XGB: 91, LGBM: 93 },
              { metric: 'Recall', LR: 72, RF: 67, XGB: 62, LGBM: 64 },
              { metric: 'F1', LR: 70, RF: 74, XGB: 74, LGBM: 76 },
              { metric: 'PR-AUC', LR: 52, RF: 72, XGB: 81, LGBM: 84 },
              { metric: 'ROC-AUC', LR: 78, RF: 91, XGB: 96, LGBM: 97 },
            ]}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <Radar name="LogReg" dataKey="LR" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.1} />
              <Radar name="RF" dataKey="RF" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} />
              <Radar name="XGBoost" dataKey="XGB" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} />
              <Radar name="LightGBM" dataKey="LGBM" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.2} />
              <Legend formatter={(v) => <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{v}</span>} />
              <Tooltip {...tooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Model Cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginTop: '24px' }}>
        {modelData.map((model) => (
          <div key={model.name} className="card" style={{
            borderLeft: `3px solid ${model.color}`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
              <Cpu size={20} color={model.color} />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{model.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                  {model.type} · {model.role}
                </div>
              </div>
              {model.role.includes('Best') && (
                <span className="badge badge-success" style={{ marginLeft: 'auto' }}>Best</span>
              )}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.5 }}>
              {model.description}
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
              <MiniMetric label="Precision" value={(model.precision * 100).toFixed(1) + '%'} />
              <MiniMetric label="Recall" value={(model.recall * 100).toFixed(1) + '%'} />
              <MiniMetric label="F1" value={(model.f1 * 100).toFixed(1) + '%'} />
              <MiniMetric label="PR-AUC" value={model.prauc.toFixed(3)} />
              <MiniMetric label="Threshold" value={model.threshold.toFixed(2)} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricBox({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.2rem', fontWeight: 800, color, fontFamily: 'var(--font-mono)' }}>
        {value}
      </div>
    </div>
  )
}

function MiniMetric({ label, value }) {
  return (
    <div style={{
      textAlign: 'center', padding: '8px 4px',
      background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
    }}>
      <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '2px' }}>
        {label}
      </div>
      <div style={{ fontSize: '0.82rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  )
}
