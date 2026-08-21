import { Link } from 'react-router-dom';
import { Plus, FileText, FolderOpen, ArrowUpRight } from 'lucide-react';
import { useEffect, useState } from "react";
import api from "../services/api";

export default function Dashboard() {
  const [workspaces, setWorkspaces] = useState([]);
  const [recentDocs, setRecentDocs] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState(null);
  const [loading, setLoading] = useState(true);

  const handleCreateWorkspace = async (e) => {
    e.preventDefault();
    if (!newWorkspaceName.trim()) return;

    setIsCreating(true);
    setCreateError(null);

    try {
      const res = await api.post("/api/workspaces/", {
        name: newWorkspaceName.trim()
      });
      setWorkspaces((prev) => [res.data, ...prev]);
      setIsModalOpen(false);
      setNewWorkspaceName("");
    } catch (err) {
      console.error("Failed to create workspace:", err);
      setCreateError("Failed to create workspace. Please try again.");
    } finally {
      setIsCreating(false);
    }
  };

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const wsRes = await api.get("/api/workspaces/");
        const fetchedWorkspaces = wsRes.data;
        setWorkspaces(fetchedWorkspaces);

        if (fetchedWorkspaces.length > 0) {
          let allDocs = [];
          // Fetch docs for all workspaces to find recent ones
          for (const ws of fetchedWorkspaces) {
            try {
              const docsRes = await api.get(`/api/documents/?workspace_id=${ws.id}`);
              const docsWithWs = docsRes.data.map(doc => ({
                ...doc,
                workspaceName: ws.name
              }));
              allDocs = [...allDocs, ...docsWithWs];
            } catch (err) {
              console.error(`Failed to fetch documents for workspace ${ws.id}:`, err);
            }
          }

          // Sort by id descending as a proxy for recency, and take top 5
          allDocs.sort((a, b) => b.id - a.id);
          setRecentDocs(allDocs.slice(0, 5));
        }
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const totalDocuments = workspaces.reduce((sum, ws) => sum + (ws.docs || 0), 0);

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto flex items-center justify-center min-h-[60vh]">
        <div className="text-[var(--text-secondary)]">Loading Dashboard...</div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">Dashboard</h1>
          <p className="text-[var(--text-secondary)] mt-1">Overview of your workspaces and documents.</p>
        </div>

        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white px-4 py-2 rounded-lg font-medium transition-colors shadow-sm"
        >
          <Plus size={18} />
          Create Workspace
        </button>
      </div>

      {workspaces.length === 0 ? (
        <div className="bg-[var(--background-card)] rounded-2xl border border-[var(--border)] shadow-sm p-12 text-center max-w-2xl mx-auto mt-12">
          <div className="w-16 h-16 bg-[var(--primary-soft)] rounded-full flex items-center justify-center mx-auto mb-6">
            <FolderOpen className="w-8 h-8 text-[var(--primary)]" />
          </div>
          <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">Welcome to ContextAI</h2>
          <p className="text-[var(--text-secondary)] mb-8 max-w-md mx-auto">
            Create your first workspace to organize documents and start asking questions.
          </p>
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => setIsModalOpen(true)}
              className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white px-6 py-3 rounded-lg font-medium transition-colors shadow-sm flex items-center gap-2"
            >
              <Plus size={18} /> Create Workspace
            </button>
            <Link
              to="/upload"
              className="bg-[var(--background-card)] border border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--background-hover)] px-6 py-3 rounded-lg font-medium transition-colors shadow-sm flex items-center gap-2"
            >
              <FileText size={18} /> Upload Document
            </Link>
          </div>
        </div>
      ) : (
        <>
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
            <div className="bg-[var(--background-card)] p-6 rounded-xl border border-[var(--border)] shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-[var(--text-secondary)]">Total Documents</p>
                  <p className="text-2xl font-bold text-[var(--text-primary)] mt-2">{totalDocuments}</p>
                </div>
                <div className="h-10 w-10 rounded-lg bg-[var(--background-hover)] flex items-center justify-center">
                  <FileText className="h-5 w-5 text-[var(--text-secondary)]" />
                </div>
              </div>
            </div>
            <div className="bg-[var(--background-card)] p-6 rounded-xl border border-[var(--border)] shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-[var(--text-secondary)]">Active Workspaces</p>
                  <p className="text-2xl font-bold text-[var(--text-primary)] mt-2">{workspaces.length}</p>
                </div>
                <div className="h-10 w-10 rounded-lg bg-[var(--background-hover)] flex items-center justify-center">
                  <FolderOpen className="h-5 w-5 text-[var(--text-secondary)]" />
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Workspaces */}
            <div className="lg:col-span-2">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-[var(--text-primary)]">Recent Workspaces</h2>
                <Link to="/chat" className="text-sm font-medium text-[var(--primary)] hover:text-[var(--primary-hover)]">View all</Link>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {workspaces.map((ws) => (
                  <Link key={ws.id} to={`/chat?workspace=${ws.id}`} className="block group">
                    <div className="bg-[var(--background-card)] p-5 rounded-xl border border-[var(--border)] shadow-sm hover:shadow-md hover:border-[var(--primary)] transition-all">
                      <div className="flex items-start justify-between mb-4">
                        <div className={`h-10 w-10 rounded-lg ${ws.color || 'bg-emerald-500'} bg-opacity-10 flex items-center justify-center`}>
                          <FolderOpen className={`h-5 w-5 ${(ws.color || 'bg-emerald-500').replace('bg-', 'text-')}`} />
                        </div>
                        <ArrowUpRight className="h-5 w-5 text-[var(--text-secondary)] group-hover:text-[var(--primary)] transition-colors" />
                      </div>
                      <h3 className="font-semibold text-[var(--text-primary)]">{ws.name}</h3>
                      <div className="flex items-center justify-between mt-3 text-sm text-[var(--text-secondary)]">
                        <span>{ws.docs} documents</span>
                        <span>Updated {ws.updated || 'Recently'}</span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>

            {/* Recent Documents */}
            <div>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-[var(--text-primary)]">Recent Documents</h2>
              </div>
              <div className="bg-[var(--background-card)] rounded-xl border border-[var(--border)] shadow-sm overflow-hidden">
                {recentDocs.length > 0 ? (
                  <>
                    <ul className="divide-y divide-[var(--border)]">
                      {recentDocs.map((doc) => (
                        <li key={doc.id} className="p-4 hover:bg-[var(--background-hover)] transition-colors cursor-pointer">
                          <div className="flex items-start gap-3">
                            <div className="mt-1">
                              <FileText className="h-5 w-5 text-[var(--text-secondary)]" />
                            </div>
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-[var(--text-primary)] truncate" title={doc.name}>
                                {doc.name}
                              </p>
                              <div className="flex items-center gap-2 mt-1 text-xs text-[var(--text-secondary)] truncate">
                                <span>{doc.workspaceName}</span>
                                <span>•</span>
                                <span className={doc.status === 'ready' ? 'text-green-600' : doc.status === 'failed' ? 'text-red-600' : 'text-yellow-600 capitalize'}>
                                  {doc.status}
                                </span>
                              </div>
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                    <div className="p-4 border-t border-[var(--border)] bg-[var(--background-hover)]">
                      <Link to="/upload" className="block text-sm font-medium text-[var(--primary)] w-full text-center hover:text-[var(--primary-hover)]">
                        Manage documents
                      </Link>
                    </div>
                  </>
                ) : (
                  <div className="p-8 text-center text-[var(--text-secondary)] text-sm">
                    No documents found.
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Create Workspace Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[var(--background-card)] rounded-xl shadow-xl w-full max-w-md p-6">
            <h2 className="text-xl font-bold text-[var(--text-primary)] mb-4">Create New Workspace</h2>

            <form onSubmit={handleCreateWorkspace}>
              <div className="mb-4">
                <label htmlFor="workspaceName" className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
                  Workspace Name
                </label>
                <input
                  id="workspaceName"
                  type="text"
                  value={newWorkspaceName}
                  onChange={(e) => setNewWorkspaceName(e.target.value)}
                  disabled={isCreating}
                  className="w-full border border-[var(--border)] bg-[var(--background-input)] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent disabled:opacity-50 text-[var(--text-primary)]"
                  placeholder="e.g. Engineering Docs"
                  autoFocus
                  required
                />
              </div>

              {createError && (
                <div className="mb-4 text-[var(--danger)] text-sm">
                  {createError}
                </div>
              )}

              <div className="flex justify-end gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  disabled={isCreating}
                  className="px-4 py-2 text-[var(--text-secondary)] hover:bg-[var(--background-hover)] rounded-lg transition-colors font-medium disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreating || !newWorkspaceName.trim()}
                  className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center min-w-[80px]"
                >
                  {isCreating ? 'Creating...' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
