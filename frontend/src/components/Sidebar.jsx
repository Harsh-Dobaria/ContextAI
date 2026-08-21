import { Link, useLocation } from 'react-router-dom';
import { useContext } from 'react';
import { LayoutDashboard, MessageSquare, UploadCloud, Settings, Database, LogOut } from 'lucide-react';
import { AuthContext } from '../context/AuthContext';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Upload', href: '/upload', icon: UploadCloud },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar() {
  const location = useLocation();
  const { user, logout } = useContext(AuthContext);

  const getInitials = (name) => {
    return name ? name.substring(0, 2).toUpperCase() : 'U';
  };

  return (
    <div className="flex h-full w-64 flex-col bg-[var(--background-card)] border-r border-[var(--border)] px-4 py-6">
      <div className="flex items-center gap-3 px-2 mb-10">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--primary)] text-white">
          <Database size={18} />
        </div>
        <span className="text-lg font-semibold tracking-tight text-[var(--text-primary)]">ContextAI</span>
      </div>

      <nav className="flex-1 space-y-1">
        {navigation.map((item) => {
          const isActive = location.pathname.startsWith(item.href);
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors ${isActive
                  ? 'bg-[var(--primary-soft)] text-[var(--primary)]'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--background-hover)] hover:text-[var(--text-primary)]'
                }`}
            >
              <item.icon
                className={`mr-3 h-5 w-5 flex-shrink-0 ${isActive ? 'text-[var(--primary)]' : 'text-[var(--text-secondary)] group-hover:text-[var(--text-primary)]'
                  }`}
                aria-hidden="true"
              />
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto px-2 space-y-2">
        {user && (
          <div className="flex items-center gap-3 rounded-lg border border-[var(--border)] p-3">
            <div className="h-8 w-8 rounded-full bg-[var(--background-hover)] flex items-center justify-center text-sm font-medium text-[var(--text-secondary)] shrink-0">
              {getInitials(user.username)}
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium text-[var(--text-primary)] truncate">{user.username}</span>
              <span className="text-xs text-[var(--text-secondary)] truncate">{user.email}</span>
            </div>
          </div>
        )}
        <button
          onClick={logout}
          className="w-full flex items-center gap-3 rounded-lg p-2 text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--background-hover)] hover:text-[var(--danger)] transition-colors"
        >
          <LogOut size={18} />
          Logout
        </button>
      </div>
    </div>
  );
}
