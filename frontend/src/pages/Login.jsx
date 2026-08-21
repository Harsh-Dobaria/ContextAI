import { useState, useContext } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { Loader2 } from 'lucide-react';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    try {
      await login(username, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background-subtle)] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-[var(--background-card)] p-10 rounded-2xl shadow-sm border border-[var(--border)]">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-[var(--text-primary)]">
            Welcome back
          </h2>
          <p className="mt-2 text-center text-sm text-[var(--text-secondary)]">
            Log in to ContextAI
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="bg-[var(--danger-bg)] text-[var(--danger-text)] p-3 rounded-lg text-sm border border-[var(--danger-border)]">
              {error}
            </div>
          )}
          <div className="rounded-md shadow-sm space-y-4">
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">Username or Email</label>
              <input
                name="username"
                type="text"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-2.5 border border-[var(--border)] bg-[var(--background-input)] placeholder-[var(--text-secondary)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent sm:text-sm"
                placeholder="Username or email"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">Password</label>
              <input
                name="password"
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-2.5 border border-[var(--border)] bg-[var(--background-input)] placeholder-[var(--text-secondary)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2.5 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-[var(--primary)] hover:bg-[var(--primary-hover)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--primary)] disabled:opacity-70 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? <Loader2 size={20} className="animate-spin" /> : 'Log in'}
            </button>
          </div>
          
          <div className="text-center text-sm">
            <span className="text-[var(--text-secondary)]">Don't have an account? </span>
            <Link to="/signup" className="font-medium text-[var(--primary)] hover:text-[var(--primary-hover)]">
              Sign up
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
