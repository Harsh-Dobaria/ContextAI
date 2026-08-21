import { useState, useEffect, useRef } from 'react';
import { Upload as UploadIcon, FileText, CheckCircle2, AlertCircle, Loader2, Folder, Save } from 'lucide-react';
import api from '../services/api';

export default function Upload() {
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(true);
  
  // Pending file to be uploaded
  const [pendingFile, setPendingFile] = useState(null);
  const [uploadState, setUploadState] = useState(''); // 'selected', 'uploading', 'failed'
  const [uploadError, setUploadError] = useState('');

  const pollingRef = useRef(null);

  useEffect(() => {
    fetchWorkspaces();
    return () => clearInterval(pollingRef.current);
  }, []);

  useEffect(() => {
    if (selectedWorkspace) {
      fetchDocuments();
    }
  }, [selectedWorkspace]);

  // Poll for processing status every 3 seconds if there are processing files
  useEffect(() => {
    const processingFiles = uploadedFiles.filter(f => f.status === 'processing');
    
    if (processingFiles.length > 0) {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(() => {
          processingFiles.forEach(async (file) => {
            try {
              const res = await api.get(`/api/documents/${file.id}/status`);
              if (res.data.status !== 'processing') {
                setUploadedFiles(prev => prev.map(f => 
                  f.id === file.id ? { ...f, status: res.data.status } : f
                ));
              }
            } catch (err) {
              console.error("Failed to fetch status:", err);
            }
          });
        }, 3000);
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }
    
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [uploadedFiles]);

  const fetchWorkspaces = async () => {
    try {
      const res = await api.get('/api/workspaces/');
      setWorkspaces(res.data);
      if (res.data.length > 0) {
        setSelectedWorkspace(res.data[0].id.toString());
      }
    } catch (err) {
      console.error("Failed to fetch workspaces:", err);
    } finally {
      setLoadingWorkspaces(false);
    }
  };

  const fetchDocuments = async () => {
    if (!selectedWorkspace) return;
    try {
      const res = await api.get(`/api/documents/?workspace_id=${selectedWorkspace}`);
      // The API returns: {id, name, status, workspace_id}
      setUploadedFiles(res.data);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only PDF files are allowed');
      return;
    }

    setPendingFile(file);
    setUploadState('selected');
    setUploadError('');
    // Reset input
    e.target.value = null;
  };

  const handleUpload = async () => {
    if (!pendingFile || !selectedWorkspace) return;

    setUploadState('uploading');
    setUploadError('');

    const formData = new FormData();
    formData.append('file', pendingFile);

    try {
      const res = await api.post(`/api/documents/upload/${selectedWorkspace}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Backend should return status: 'processing'
      setUploadedFiles(prev => [res.data, ...prev]);
      
      // Clear pending
      setPendingFile(null);
      setUploadState('');
    } catch (error) {
      console.error("Upload error:", error);
      let errorMessage = "Upload failed. Check backend connection.";
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      setUploadState('failed');
      setUploadError(errorMessage);
    }
  };

  const getFileSize = (file) => {
    return (file.size / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Upload Documents</h1>
        <p className="text-[var(--text-secondary)] mt-1">Add files to your workspace to make them searchable.</p>
      </div>

      {/* Upload Zone or Pending File View */}
      {!pendingFile ? (
        <div className={`flex justify-center rounded-2xl border border-dashed border-[var(--border)] px-6 py-16 bg-[var(--background-card)] transition-all ${workspaces.length === 0 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[var(--background-hover)] hover:border-[var(--primary)] cursor-pointer group'}`}>
          <div className="text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-[var(--primary-soft)] group-hover:bg-[var(--primary-soft)] transition-colors">
              <UploadIcon className="h-6 w-6 text-[var(--primary)]" aria-hidden="true" />
            </div>
            <div className="mt-4 flex text-sm leading-6 text-[var(--text-secondary)] justify-center">
              <label
                htmlFor="file-upload"
                className={`relative rounded-md bg-transparent font-semibold focus-within:outline-none focus-within:ring-2 focus-within:ring-[var(--primary)] focus-within:ring-offset-2 ${workspaces.length === 0 ? 'cursor-not-allowed text-[var(--text-secondary)]' : 'cursor-pointer text-[var(--primary)] hover:text-[var(--primary-hover)]'}`}
              >
                <span>Select a PDF file</span>
                <input 
                  id="file-upload" 
                  name="file-upload" 
                  type="file" 
                  className="sr-only" 
                  accept=".pdf"
                  onChange={handleFileSelect}
                  disabled={workspaces.length === 0}
                />
              </label>
            </div>
            <p className="text-xs leading-5 text-[var(--text-secondary)] mt-2">PDF only up to 50MB</p>
            {workspaces.length === 0 && (
              <p className="text-sm text-[var(--danger)] mt-4">You must create a workspace before uploading.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="p-6 rounded-2xl border border-emerald-200 bg-emerald-50/30">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-lg bg-emerald-100 flex items-center justify-center shrink-0">
                <FileText className="h-6 w-6 text-emerald-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">{pendingFile.name}</p>
                <p className="text-xs text-[var(--text-secondary)] mt-1">{getFileSize(pendingFile)}</p>
              </div>
            </div>
            <button 
              onClick={() => {
                setPendingFile(null);
                setUploadState('');
                setUploadError('');
              }}
              className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-sm font-medium transition-colors"
              disabled={uploadState === 'uploading'}
            >
              Cancel
            </button>
          </div>
          
          <div className="bg-[var(--background-card)] p-4 rounded-xl border border-[var(--border)] shadow-sm mb-6">
            <label className="block text-sm font-medium text-[var(--text-primary)] mb-2 flex items-center gap-2">
              <Folder size={16} className="text-[var(--primary)]" />
              Select Target Workspace
            </label>
            {loadingWorkspaces ? (
              <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
                <Loader2 size={16} className="animate-spin" /> Loading workspaces...
              </div>
            ) : workspaces.length === 0 ? (
              <div className="text-sm text-[var(--danger)] bg-[var(--danger-bg)] p-3 rounded-lg border border-[var(--danger-border)]">
                No workspaces found. Please create a workspace in the dashboard first.
              </div>
            ) : (
              <select 
                value={selectedWorkspace}
                onChange={(e) => setSelectedWorkspace(e.target.value)}
                disabled={uploadState === 'uploading'}
                className="w-full sm:w-1/2 border border-[var(--border)] rounded-lg px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent text-[var(--text-primary)] bg-[var(--background-input)] disabled:opacity-50"
              >
                {workspaces.map(ws => (
                  <option key={ws.id} value={ws.id}>{ws.name}</option>
                ))}
              </select>
            )}
          </div>

          {uploadError && (
            <div className="mb-6 text-sm text-[var(--danger)] bg-[var(--danger-bg)] p-3 rounded-lg border border-[var(--danger-border)] flex items-center gap-2">
              <AlertCircle size={16} /> {uploadError}
            </div>
          )}

          <div className="flex justify-end">
            <button
              onClick={handleUpload}
              disabled={uploadState === 'uploading' || !selectedWorkspace}
              className="flex items-center gap-2 bg-[var(--primary)] text-white px-6 py-2.5 rounded-lg font-medium hover:bg-[var(--primary-hover)] transition-colors disabled:opacity-70 disabled:cursor-not-allowed shadow-sm"
            >
              {uploadState === 'uploading' ? (
                <><Loader2 size={18} className="animate-spin" /> Uploading...</>
              ) : (
                <><Save size={18} /> Upload PDF</>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Workspace Document List */}
      <div className="mt-10">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4">Workspace Documents</h2>
        {!selectedWorkspace ? (
           <div className="text-center p-8 bg-[var(--background-card)] border border-[var(--border)] rounded-xl text-[var(--text-secondary)] text-sm">
             Select a workspace to view its documents.
           </div>
        ) : uploadedFiles.length === 0 ? (
          <div className="text-center p-8 bg-[var(--background-card)] border border-[var(--border)] rounded-xl text-[var(--text-secondary)] text-sm">
            No documents in this workspace yet.
          </div>
        ) : (
          <div className="bg-[var(--background-card)] shadow-sm ring-1 ring-[var(--border)] sm:rounded-xl overflow-hidden">
            <ul role="list" className="divide-y divide-[var(--border)]">
              {uploadedFiles.map((file) => (
                <li key={file.id} className="p-4 flex items-center justify-between hover:bg-[var(--background-hover)] transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-lg bg-[var(--background-hover)] flex items-center justify-center shrink-0">
                      <FileText className="h-5 w-5 text-[var(--text-secondary)]" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-[var(--text-primary)] truncate max-w-[200px] sm:max-w-xs">{file.name}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 shrink-0">
                    {file.status === 'ready' && (
                      <div className="flex items-center gap-1.5 text-sm font-medium text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-100">
                        <CheckCircle2 size={14} /> Ready
                      </div>
                    )}
                    
                    {file.status === 'processing' && (
                      <div className="flex items-center gap-1.5 text-sm font-medium text-[var(--primary)] bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-100">
                        <Loader2 size={14} className="animate-spin" /> Processing
                      </div>
                    )}

                    {file.status === 'failed' && (
                      <div className="flex items-center gap-1.5 text-sm font-medium text-red-600 bg-red-50 px-2.5 py-1 rounded-full border border-red-100">
                        <AlertCircle size={14} /> Failed
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

