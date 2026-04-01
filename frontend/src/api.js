const API_BASE = 'https://detection-system-production.up.railway.app'

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error('API unreachable')
  return res.json()
}

export async function predictFraud(transaction) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}

export async function predictBatch(transactions) {
  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transactions }),
  })
  if (!res.ok) throw new Error('Batch prediction failed')
  return res.json()
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model/info`)
  if (!res.ok) throw new Error('Could not fetch model info')
  return res.json()
}
