import { useState } from 'react'
import { Clock, Filter, Download, ShieldAlert, ShieldCheck, Eye } from 'lucide-react'

const mockHistory = [
  { id: 'TXN-89124', time: '2026-03-31 23:42:18', amount: 4899.99, card: '****4521', email: 'user@proton.me', device: 'Mobile / iOS', prob: 0.97, risk: 'critical', isfraud: true },
  { id: 'TXN-89118', time: '2026-03-31 23:39:41', amount: 1250.00, card: '****7832', email: 'j.doe@gmail.com', device: 'Desktop / Win', prob: 0.82, risk: 'high', isfraud: true },
  { id: 'TXN-89115', time: '2026-03-31 23:38:05', amount: 89.50, card: '****1293', email: 'alice@yahoo.com', device: 'Desktop / Mac', prob: 0.05, risk: 'low', isfraud: false },
  { id: 'TXN-89103', time: '2026-03-31 23:32:12', amount: 899.50, card: '****2190', email: 'test@mail.ru', device: 'Mobile / Android', prob: 0.76, risk: 'high', isfraud: true },
  { id: 'TXN-89097', time: '2026-03-31 23:28:44', amount: 320.00, card: '****6453', email: 'shop@outlook.com', device: 'Desktop / Win', prob: 0.54, risk: 'medium', isfraud: false },
  { id: 'TXN-89091', time: '2026-03-31 23:25:01', amount: 45.00, card: '****8801', email: 'kim@gmail.com', device: 'Mobile / iOS', prob: 0.02, risk: 'low', isfraud: false },
  { id: 'TXN-89085', time: '2026-03-31 23:21:33', amount: 2100.00, card: '****1187', email: 'info@temp.xyz', device: 'Mobile / Android', prob: 0.94, risk: 'critical', isfraud: true },
  { id: 'TXN-89078', time: '2026-03-31 23:18:09', amount: 175.25, card: '****3342', email: 'sarah@icloud.com', device: 'Desktop / Mac', prob: 0.11, risk: 'low', isfraud: false },
  { id: 'TXN-89072', time: '2026-03-31 23:15:22', amount: 3500.00, card: '****5590', email: 'anon@pm.me', device: 'Mobile / Android', prob: 0.88, risk: 'high', isfraud: true },
  { id: 'TXN-89065', time: '2026-03-31 23:11:47', amount: 22.99, card: '****7710', email: 'jane@gmail.com', device: 'Desktop / Win', prob: 0.01, risk: 'low', isfraud: false },
  { id: 'TXN-89058', time: '2026-03-31 23:08:30', amount: 6780.00, card: '****0042', email: 'ghost@yopmail.com', device: 'Mobile / iOS', prob: 0.99, risk: 'critical', isfraud: true },
  { id: 'TXN-89051', time: '2026-03-31 23:05:15', amount: 130.00, card: '****4478', email: 'mike@gmail.com', device: 'Desktop / Win', prob: 0.08, risk: 'low', isfraud: false },
]

export default function HistoryPage() {
  const [filter, setFilter] = useState('all')
  const [selectedTxn, setSelectedTxn] = useState(null)

  const filtered = filter === 'all'
    ? mockHistory
    : filter === 'fraud'
      ? mockHistory.filter((t) => t.isfraud)
      : mockHistory.filter((t) => !t.isfraud)

  const getRiskColor = (level) => ({
    low: 'var(--risk-low)',
    medium: 'var(--risk-medium)',
    high: 'var(--risk-high)',
    critical: 'var(--risk-critical)',
  }[level])

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>Transaction History</h2>
        <p>View past predictions and fraud analysis results</p>
      </div>

      {/* ── Summary Stats ── */}
      <div className="stats-grid" style={{ marginBottom: '24px' }}>
        <div className="stat-card" style={{ '--stat-accent': 'var(--accent-primary)' }}>
          <div className="stat-icon" style={{ background: 'rgba(99,102,241,0.12)', color: 'var(--accent-primary)' }}>
            <Clock size={20} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Total Scored</div>
            <div className="stat-value">{mockHistory.length}</div>
          </div>
        </div>
        <div className="stat-card" style={{ '--stat-accent': 'var(--color-danger)' }}>
          <div className="stat-icon" style={{ background: 'rgba(239,68,68,0.12)', color: 'var(--color-danger)' }}>
            <ShieldAlert size={20} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Fraud Flagged</div>
            <div className="stat-value">{mockHistory.filter(t => t.isfraud).length}</div>
          </div>
        </div>
        <div className="stat-card" style={{ '--stat-accent': 'var(--color-success)' }}>
          <div className="stat-icon" style={{ background: 'rgba(34,197,94,0.12)', color: 'var(--color-success)' }}>
            <ShieldCheck size={20} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Legitimate</div>
            <div className="stat-value">{mockHistory.filter(t => !t.isfraud).length}</div>
          </div>
        </div>
      </div>

      {/* ── Filter & Table ── */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilter('all')}
            >All</button>
            <button
              className={`btn btn-sm ${filter === 'fraud' ? 'btn-danger' : 'btn-secondary'}`}
              onClick={() => setFilter('fraud')}
            ><ShieldAlert size={14} /> Fraud Only</button>
            <button
              className={`btn btn-sm ${filter === 'legit' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilter('legit')}
            ><ShieldCheck size={14} /> Legitimate</button>
          </div>
          <button className="btn btn-secondary btn-sm">
            <Download size={14} /> Export CSV
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Transaction ID</th>
                <th>Timestamp</th>
                <th>Amount</th>
                <th>Card</th>
                <th>Email</th>
                <th>Device</th>
                <th>Risk</th>
                <th>Probability</th>
                <th>Decision</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((txn) => (
                <tr key={txn.id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {txn.id}
                  </td>
                  <td style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>{txn.time}</td>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    ${txn.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{txn.card}</td>
                  <td style={{ fontSize: '0.78rem' }}>{txn.email}</td>
                  <td style={{ fontSize: '0.78rem' }}>{txn.device}</td>
                  <td>
                    <span className={`badge badge-risk-${txn.risk}`}>{txn.risk}</span>
                  </td>
                  <td>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontWeight: 700,
                      color: getRiskColor(txn.risk),
                    }}>
                      {(txn.prob * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    {txn.isfraud ? (
                      <span className="badge badge-danger">Fraud</span>
                    ) : (
                      <span className="badge badge-success">Legit</span>
                    )}
                  </td>
                  <td>
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ padding: '4px 8px' }}
                      onClick={() => setSelectedTxn(selectedTxn?.id === txn.id ? null : txn)}
                    >
                      <Eye size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Detail Panel */}
        {selectedTxn && (
          <div className="animate-in" style={{
            marginTop: '16px', padding: '20px',
            background: 'var(--bg-glass)', borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}>
            <div style={{ fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              {selectedTxn.isfraud ? <ShieldAlert size={18} color="var(--color-danger)" /> : <ShieldCheck size={18} color="var(--color-success)" />}
              Transaction Detail: {selectedTxn.id}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
              <DetailItem label="Amount" value={`$${selectedTxn.amount.toLocaleString()}`} />
              <DetailItem label="Card" value={selectedTxn.card} />
              <DetailItem label="Email" value={selectedTxn.email} />
              <DetailItem label="Device" value={selectedTxn.device} />
              <DetailItem label="Fraud Probability" value={`${(selectedTxn.prob * 100).toFixed(1)}%`} />
              <DetailItem label="Risk Level" value={selectedTxn.risk.toUpperCase()} />
              <DetailItem label="Decision" value={selectedTxn.isfraud ? 'FRAUD' : 'LEGITIMATE'} />
              <DetailItem label="Timestamp" value={selectedTxn.time} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function DetailItem({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px' }}>
        {label}
      </div>
      <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  )
}
