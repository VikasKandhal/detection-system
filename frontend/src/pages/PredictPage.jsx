import { useState } from 'react'
import { Search, ShieldAlert, ShieldCheck, AlertTriangle, Loader2, Send } from 'lucide-react'
import { predictFraud } from '../api'

const defaultTransaction = {
  TransactionAmt: 150.0,
  ProductCD: 'W',
  card1: 13926,
  card2: 361.0,
  card3: 150.0,
  card4: 'visa',
  card5: 226.0,
  card6: 'debit',
  addr1: 299.0,
  addr2: 87.0,
  P_emaildomain: 'gmail.com',
  R_emaildomain: '',
  DeviceType: 'desktop',
  DeviceInfo: 'Windows',
  TransactionDT: 86400,
}

const sampleTransactions = {
  legitimate: {
    TransactionAmt: 45.0,
    ProductCD: 'W',
    card1: 13926,
    card2: 361.0,
    card3: 150.0,
    card4: 'visa',
    card5: 226.0,
    card6: 'debit',
    addr1: 299.0,
    addr2: 87.0,
    P_emaildomain: 'gmail.com',
    R_emaildomain: '',
    DeviceType: 'desktop',
    DeviceInfo: 'Windows',
    TransactionDT: 86400,
  },
  suspicious: {
    TransactionAmt: 4999.99,
    ProductCD: 'H',
    card1: 99999,
    card2: 555.0,
    card3: 150.0,
    card4: 'mastercard',
    card5: 224.0,
    card6: 'credit',
    addr1: 100.0,
    addr2: 87.0,
    P_emaildomain: 'protonmail.com',
    R_emaildomain: '',
    DeviceType: 'mobile',
    DeviceInfo: 'iOS',
    TransactionDT: 10800,
  },
  highrisk: {
    TransactionAmt: 12500.0,
    ProductCD: 'R',
    card1: 54321,
    card2: 111.0,
    card3: 150.0,
    card4: 'visa',
    card5: 138.0,
    card6: 'credit',
    addr1: 450.0,
    addr2: 87.0,
    P_emaildomain: 'outlook.com',
    R_emaildomain: 'yahoo.com',
    DeviceType: 'mobile',
    DeviceInfo: 'Android',
    TransactionDT: 3600,
  }
}

export default function PredictPage() {
  const [formData, setFormData] = useState(defaultTransaction)
  const [jsonMode, setJsonMode] = useState(false)
  const [jsonInput, setJsonInput] = useState(JSON.stringify(defaultTransaction, null, 2))
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const loadSample = (type) => {
    const sample = sampleTransactions[type]
    setFormData(sample)
    setJsonInput(JSON.stringify(sample, null, 2))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = jsonMode ? JSON.parse(jsonInput) : formData
      const response = await predictFraud(data)
      setResult(response)
    } catch (err) {
      setError(err.message || 'Prediction failed. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (level) => {
    const colors = {
      low: 'var(--risk-low)',
      medium: 'var(--risk-medium)',
      high: 'var(--risk-high)',
      critical: 'var(--risk-critical)',
    }
    return colors[level] || 'var(--text-secondary)'
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>Predict Fraud</h2>
        <p>Score a transaction in real-time with explainability</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'start' }}>
        {/* ── Input Form ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={18} />
              Transaction Input
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                className={`btn btn-sm ${!jsonMode ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setJsonMode(false)}
              >Form</button>
              <button
                className={`btn btn-sm ${jsonMode ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setJsonMode(true)}
              >JSON</button>
            </div>
          </div>

          {/* Sample Transaction Buttons */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
            <button className="btn btn-sm btn-secondary" onClick={() => loadSample('legitimate')}>
              <ShieldCheck size={14} /> Legitimate
            </button>
            <button className="btn btn-sm btn-secondary" onClick={() => loadSample('suspicious')}>
              <AlertTriangle size={14} /> Suspicious
            </button>
            <button className="btn btn-sm btn-danger" onClick={() => loadSample('highrisk')}>
              <ShieldAlert size={14} /> High Risk
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            {jsonMode ? (
              <div className="form-group">
                <label className="form-label">JSON Transaction Data</label>
                <textarea
                  className="form-textarea"
                  value={jsonInput}
                  onChange={(e) => setJsonInput(e.target.value)}
                  style={{ minHeight: '340px' }}
                />
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <FormField label="Amount ($)" type="number" step="0.01"
                  value={formData.TransactionAmt}
                  onChange={(v) => handleChange('TransactionAmt', parseFloat(v))} />
                <FormField label="Product Code" type="text"
                  value={formData.ProductCD}
                  onChange={(v) => handleChange('ProductCD', v)} />
                <FormField label="Card ID" type="number"
                  value={formData.card1}
                  onChange={(v) => handleChange('card1', parseInt(v))} />
                <FormField label="Card Brand" type="text"
                  value={formData.card4}
                  onChange={(v) => handleChange('card4', v)} />
                <FormField label="Card Type" type="text"
                  value={formData.card6}
                  onChange={(v) => handleChange('card6', v)} />
                <FormField label="Address 1" type="number" step="1"
                  value={formData.addr1}
                  onChange={(v) => handleChange('addr1', parseFloat(v))} />
                <FormField label="Email Domain" type="text"
                  value={formData.P_emaildomain}
                  onChange={(v) => handleChange('P_emaildomain', v)} />
                <FormField label="Device Type" type="text"
                  value={formData.DeviceType}
                  onChange={(v) => handleChange('DeviceType', v)} />
                <FormField label="Device Info" type="text"
                  value={formData.DeviceInfo}
                  onChange={(v) => handleChange('DeviceInfo', v)} />
                <FormField label="Time Delta (s)" type="number"
                  value={formData.TransactionDT}
                  onChange={(v) => handleChange('TransactionDT', parseInt(v))} />
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ width: '100%', marginTop: '16px', justifyContent: 'center' }}
            >
              {loading ? (
                <><div className="spinner" /> Analyzing...</>
              ) : (
                <><Send size={16} /> Analyze Transaction</>
              )}
            </button>
          </form>

          {error && (
            <div style={{
              marginTop: '16px', padding: '12px 16px',
              background: 'var(--color-danger-bg)', borderRadius: 'var(--radius-md)',
              border: '1px solid rgba(239,68,68,0.2)', color: 'var(--color-danger)',
              fontSize: '0.85rem'
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* ── Result Panel ── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Prediction Result</div>
          </div>

          {!result && !loading && (
            <div style={{
              textAlign: 'center', padding: '60px 20px',
              color: 'var(--text-tertiary)',
            }}>
              <Search size={48} strokeWidth={1} style={{ marginBottom: '16px', opacity: 0.3 }} />
              <p>Submit a transaction to see the fraud analysis</p>
            </div>
          )}

          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              Analyzing transaction...
            </div>
          )}

          {result && (
            <div className="animate-in">
              {/* Fraud Probability Gauge */}
              <div className="prediction-result">
                <div className="fraud-gauge" style={{
                  background: `conic-gradient(${getRiskColor(result.risk_level)} ${result.fraud_probability * 360}deg, rgba(255,255,255,0.05) 0deg)`
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div className="gauge-value" style={{ color: getRiskColor(result.risk_level) }}>
                      {(result.fraud_probability * 100).toFixed(1)}%
                    </div>
                    <div className="gauge-label">FRAUD PROBABILITY</div>
                  </div>
                </div>

                <div style={{ marginBottom: '8px' }}>
                  {result.is_fraud ? (
                    <span className="badge badge-danger" style={{ fontSize: '0.9rem', padding: '6px 16px' }}>
                      <ShieldAlert size={16} /> FRAUD DETECTED
                    </span>
                  ) : (
                    <span className="badge badge-success" style={{ fontSize: '0.9rem', padding: '6px 16px' }}>
                      <ShieldCheck size={16} /> LEGITIMATE
                    </span>
                  )}
                </div>

                <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '8px' }}>
                  Risk Level: <span className={`badge badge-risk-${result.risk_level}`}>{result.risk_level}</span>
                  {' · '}Model: <span style={{ color: 'var(--text-secondary)' }}>{result.model_used}</span>
                  {' · '}Threshold: <span style={{ fontFamily: 'var(--font-mono)' }}>{result.threshold_used?.toFixed(3)}</span>
                </div>
              </div>

              {/* Risk Factors */}
              {result.top_risk_factors && result.top_risk_factors.length > 0 && (
                <div style={{ marginTop: '24px' }}>
                  <div style={{
                    fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)',
                    marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px'
                  }}>
                    Top Risk Factors (SHAP)
                  </div>
                  <div className="risk-factors">
                    {result.top_risk_factors.slice(0, 8).map((f, i) => (
                      <div key={i} className={`risk-factor ${f.direction === 'increases' ? 'increase' : 'decrease'}`}>
                        <span className="factor-name">{f.feature}</span>
                        <span className="factor-value">= {typeof f.value === 'number' ? f.value.toFixed(4) : f.value}</span>
                        <span className="factor-impact" style={{
                          color: f.direction === 'increases' ? 'var(--color-danger)' : 'var(--color-success)'
                        }}>
                          {f.direction === 'increases' ? '▲' : '▼'} {Math.abs(f.shap_value).toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Explanation */}
              {result.explanation && result.explanation.length > 0 && (
                <div style={{
                  marginTop: '20px', padding: '16px',
                  background: 'var(--bg-glass)', borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)'
                }}>
                  <div style={{
                    fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-tertiary)',
                    marginBottom: '8px', textTransform: 'uppercase'
                  }}>
                    AI Explanation
                  </div>
                  {result.explanation.map((reason, i) => (
                    <div key={i} style={{
                      fontSize: '0.82rem', color: 'var(--text-secondary)',
                      padding: '4px 0', lineHeight: 1.5
                    }}>
                      {reason}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function FormField({ label, type, value, onChange, step }) {
  return (
    <div className="form-group" style={{ margin: 0 }}>
      <label className="form-label">{label}</label>
      <input
        className="form-input"
        type={type}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  )
}
