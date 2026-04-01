import { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, CartesianGrid, Legend
} from 'recharts'
import {
  ShieldAlert, ShieldCheck, Activity, TrendingUp,
  AlertTriangle, Zap, Target, FileBarChart
} from 'lucide-react'
import { checkHealth } from '../api'

/* ─── Mock data for dashboard visualization ─── */
const fraudDistribution = [
  { name: 'Legitimate', value: 569877, color: '#22c55e' },
  { name: 'Fraud', value: 20663, color: '#ef4444' },
]

const dailyFraudData = [
  { day: 'Mon', total: 84000, fraud: 2940, rate: 3.5 },
  { day: 'Tue', total: 87000, fraud: 3045, rate: 3.5 },
  { day: 'Wed', total: 82000, fraud: 3280, rate: 4.0 },
  { day: 'Thu', total: 89000, fraud: 2670, rate: 3.0 },
  { day: 'Fri', total: 91000, fraud: 3640, rate: 4.0 },
  { day: 'Sat', total: 78000, fraud: 3120, rate: 4.0 },
  { day: 'Sun', total: 79540, fraud: 2968, rate: 3.7 },
]

const modelPerformanceData = [
  { model: 'LogReg', precision: 0.68, recall: 0.72, f1: 0.70, prauc: 0.52 },
  { model: 'RF', precision: 0.82, recall: 0.67, f1: 0.74, prauc: 0.72 },
  { model: 'XGBoost', precision: 0.91, recall: 0.62, f1: 0.74, prauc: 0.81 },
  { model: 'LightGBM', precision: 0.93, recall: 0.64, f1: 0.76, prauc: 0.84 },
]

const riskDistributionData = [
  { name: 'Low', value: 520000, color: '#22c55e' },
  { name: 'Medium', value: 45000, color: '#f59e0b' },
  { name: 'High', value: 18000, color: '#f97316' },
  { name: 'Critical', value: 7540, color: '#ef4444' },
]

const recentAlerts = [
  { id: 'TXN-89124', amount: 4899.99, risk: 'critical', prob: 0.97, time: '2 min ago', card: '****4521' },
  { id: 'TXN-89118', amount: 1250.00, risk: 'high', prob: 0.82, time: '5 min ago', card: '****7832' },
  { id: 'TXN-89103', amount: 899.50, risk: 'high', prob: 0.76, time: '12 min ago', card: '****2190' },
  { id: 'TXN-89097', amount: 320.00, risk: 'medium', prob: 0.54, time: '18 min ago', card: '****6453' },
  { id: 'TXN-89085', amount: 2100.00, risk: 'critical', prob: 0.94, time: '23 min ago', card: '****1187' },
]

const tooltipStyle = {
  contentStyle: {
    background: 'rgba(17, 24, 39, 0.95)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '10px',
    fontSize: '0.8rem',
    color: '#f1f5f9',
    backdropFilter: 'blur(10px)',
  },
}

export default function Dashboard() {
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
  }, [])

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Real-time fraud detection monitoring and analytics</p>
      </div>

      {/* ── Stats Grid ── */}
      <div className="stats-grid">
        <StatCard
          icon={<Activity size={20} />}
          label="Total Transactions"
          value="590,540"
          change="+12.5%"
          positive
          color="#6366f1"
        />
        <StatCard
          icon={<ShieldAlert size={20} />}
          label="Fraud Detected"
          value="20,663"
          change="-3.2%"
          positive
          color="#ef4444"
        />
        <StatCard
          icon={<Target size={20} />}
          label="Model Precision"
          value="93.2%"
          change="+1.8%"
          positive
          color="#22c55e"
        />
        <StatCard
          icon={<Zap size={20} />}
          label="API Status"
          value={apiStatus === 'online' ? 'Online' : apiStatus === 'checking' ? '...' : 'Offline'}
          change={apiStatus === 'online' ? 'Healthy' : apiStatus === 'checking' ? 'Checking' : 'Unavailable'}
          positive={apiStatus === 'online'}
          color={apiStatus === 'online' ? '#14b8a6' : '#f59e0b'}
        />
      </div>

      {/* ── Charts ── */}
      <div className="charts-grid">
        {/* Fraud Distribution Pie */}
        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Fraud Distribution</div>
              <div className="card-subtitle">Overall class balance in dataset</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={fraudDistribution}
                cx="50%"
                cy="50%"
                innerRadius={65}
                outerRadius={100}
                paddingAngle={4}
                dataKey="value"
                stroke="none"
              >
                {fraudDistribution.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip {...tooltipStyle} formatter={(v) => v.toLocaleString()} />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value) => <span style={{ color: '#94a3b8', fontSize: '0.82rem' }}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Model Comparison */}
        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Model Performance</div>
              <div className="card-subtitle">Precision, Recall, and F1 across models</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelPerformanceData} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip {...tooltipStyle} formatter={(v) => (v * 100).toFixed(1) + '%'} />
              <Bar dataKey="precision" fill="#6366f1" radius={[4, 4, 0, 0]} name="Precision" />
              <Bar dataKey="recall" fill="#14b8a6" radius={[4, 4, 0, 0]} name="Recall" />
              <Bar dataKey="f1" fill="#f59e0b" radius={[4, 4, 0, 0]} name="F1" />
              <Legend
                formatter={(value) => <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{value}</span>}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Daily Fraud Trend */}
        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Daily Fraud Trend</div>
              <div className="card-subtitle">Fraud rate over the past week</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={dailyFraudData}>
              <defs>
                <linearGradient id="fraudGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="day" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip {...tooltipStyle} />
              <Area
                type="monotone" dataKey="fraud" stroke="#ef4444"
                fill="url(#fraudGrad)" strokeWidth={2}
                name="Fraud Transactions"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="chart-card">
          <div className="card-header">
            <div>
              <div className="card-title">Risk Distribution</div>
              <div className="card-subtitle">Transactions by risk level</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={riskDistributionData}
                cx="50%"
                cy="50%"
                innerRadius={65}
                outerRadius={100}
                paddingAngle={3}
                dataKey="value"
                stroke="none"
              >
                {riskDistributionData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip {...tooltipStyle} formatter={(v) => v.toLocaleString()} />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value) => <span style={{ color: '#94a3b8', fontSize: '0.82rem' }}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Recent Alerts Table ── */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} color="var(--color-warning)" />
              Recent Fraud Alerts
            </div>
            <div className="card-subtitle">Transactions flagged by the system</div>
          </div>
          <button className="btn btn-secondary btn-sm">
            <FileBarChart size={14} /> Export
          </button>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Transaction ID</th>
                <th>Card</th>
                <th>Amount</th>
                <th>Risk Level</th>
                <th>Fraud Prob.</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {recentAlerts.map((alert) => (
                <tr key={alert.id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {alert.id}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{alert.card}</td>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    ${alert.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </td>
                  <td>
                    <span className={`badge badge-risk-${alert.risk}`}>
                      {alert.risk}
                    </span>
                  </td>
                  <td>
                    <span style={{ 
                      fontFamily: 'var(--font-mono)', fontWeight: 700,
                      color: alert.prob >= 0.9 ? 'var(--risk-critical)' : 
                             alert.prob >= 0.7 ? 'var(--risk-high)' : 'var(--risk-medium)'
                    }}>
                      {(alert.prob * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-tertiary)' }}>{alert.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, change, positive, color }) {
  return (
    <div className="stat-card" style={{ '--stat-accent': color }}>
      <div className="stat-icon" style={{ background: `${color}15`, color }}>
        {icon}
      </div>
      <div className="stat-content">
        <div className="stat-label">{label}</div>
        <div className="stat-value" style={{ color }}>{value}</div>
        <div className={`stat-change ${positive ? 'positive' : 'negative'}`}>
          {change}
        </div>
      </div>
    </div>
  )
}
