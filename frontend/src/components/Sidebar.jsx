import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, Search, Cpu, Clock, Shield
} from 'lucide-react'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/predict', icon: Search, label: 'Predict Fraud' },
  { path: '/models', icon: Cpu, label: 'Models' },
  { path: '/history', icon: Clock, label: 'History' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">
          <Shield size={22} color="white" />
        </div>
        <div>
          <h1>FraudShield</h1>
          <span className="badge">AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(({ path, icon: Icon, label }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            end={path === '/'}
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div style={{ 
        padding: '14px', 
        background: 'var(--bg-glass)', 
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-subtle)',
        marginTop: 'auto'
      }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
          SYSTEM STATUS
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem' }}>
          <div style={{ 
            width: 8, height: 8, borderRadius: '50%', 
            background: 'var(--color-success)',
            boxShadow: '0 0 8px var(--color-success)'
          }}/>
          <span style={{ color: 'var(--text-secondary)' }}>Operational</span>
        </div>
      </div>
    </aside>
  )
}
