import { useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { ThemeContext } from '../context/ThemeContext';
import { Sun, Moon } from 'lucide-react';

export default function Settings() {
  const { user, logout } = useContext(AuthContext);
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Settings</h1>
        <p className="text-[var(--text-secondary)] mt-1">Manage your platform configurations and preferences.</p>
      </div>

      <div className="space-y-10">
        {/* Account Info */}
        <section className="bg-[var(--background-card)] shadow-sm ring-1 ring-[var(--border)] rounded-xl overflow-hidden">
          <div className="px-6 py-5 border-b border-[var(--border)] bg-[var(--background-hover)]">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">Account</h2>
            <p className="text-sm text-[var(--text-secondary)] mt-1">Your personal account information.</p>
          </div>
          <div className="px-6 py-6 space-y-4">
            <div>
              <h4 className="text-sm font-medium text-[var(--text-primary)]">Username</h4>
              <p className="text-sm text-[var(--text-secondary)]">{user?.username}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-[var(--text-primary)]">Email</h4>
              <p className="text-sm text-[var(--text-secondary)]">{user?.email}</p>
            </div>
          </div>
        </section>

        {/* Theme & About */}
        <section className="bg-[var(--background-card)] shadow-sm ring-1 ring-[var(--border)] rounded-xl overflow-hidden">
          <div className="px-6 py-5 border-b border-[var(--border)] bg-[var(--background-hover)]">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">Appearance & Info</h2>
          </div>
          <div className="px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-[var(--text-primary)]">Theme</h4>
                <p className="text-sm text-[var(--text-secondary)]">
                  Currently using {theme === 'dark' ? 'Dark' : 'Light'} Mode.
                </p>
              </div>
              <button
                onClick={toggleTheme}
                className="flex items-center gap-2 rounded-md bg-[var(--background-hover)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] shadow-sm ring-1 ring-[var(--border)] hover:bg-[var(--border)] transition-colors"
              >
                {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
                {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
              </button>
            </div>
            <div className="mt-6 pt-6 border-t border-[var(--border)]">
              <h4 className="text-sm font-medium text-[var(--text-primary)]">ContextAI</h4>
              <p className="text-sm text-[var(--text-secondary)] mt-1">Version 1.0.0 (Enterprise Edition)</p>
            </div>
          </div>
        </section>
        
        {/* Security */}
        <section className="bg-[var(--background-card)] shadow-sm ring-1 ring-[var(--border)] rounded-xl overflow-hidden">
          <div className="px-6 py-5 border-b border-[var(--border)] bg-[var(--background-hover)]">
            <h2 className="text-base font-semibold text-[var(--text-primary)]">Security</h2>
          </div>
          <div className="px-6 py-6">
            <button 
              onClick={logout}
              className="rounded-md bg-[var(--danger-bg)] text-[var(--danger)] px-4 py-2 text-sm font-semibold border border-[var(--danger-border)] shadow-sm hover:opacity-80 transition-colors"
            >
              Log out of all devices
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
