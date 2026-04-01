import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import PredictPage from './pages/PredictPage'
import ModelsPage from './pages/ModelsPage'
import HistoryPage from './pages/HistoryPage'
import './index.css'

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
