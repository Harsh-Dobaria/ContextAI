import { useState, useEffect, useRef } from 'react';
import {
  Send,
  FileText,
  ChevronRight,
  Plus,
  Paperclip,
  MoreVertical,
  Loader2,
  Bot,
  User,
  Zap,
  Target
} from 'lucide-react';
import api from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useNavigate } from 'react-router-dom';

export default function Chat() {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState(null);
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(true);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [retrievalMode, setRetrievalMode] = useState('fast');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth'
    });
  }, [messages, isTyping]);

  const fetchWorkspaces = async () => {
    try {
      const res = await api.get('/api/workspaces/');
      setWorkspaces(res.data);
      if (res.data.length > 0) {
        setSelectedWorkspace(res.data[0]);
      }
    } catch (err) {
      console.error('Failed to fetch workspaces:', err);
    } finally {
      setLoadingWorkspaces(false);
    }
  };

  const handleWorkspaceChange = (ws) => {
    setSelectedWorkspace(ws);
    setMessages([]);
    setConversationId(null);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!input.trim() || !selectedWorkspace) {
      return;
    }

    const userMessage = {
      role: 'user',
      content: input.trim()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const params = new URLSearchParams({
        query: userMessage.content,
        workspace_id: selectedWorkspace.id,
        retrieval_mode: retrievalMode
      });

      if (conversationId !== null) {
        params.append('conversation_id', conversationId);
      }

      const res = await api.get(
        `/api/chat/?${params.toString()}`
      );

      if (res.data.conversation_id) {
        setConversationId(res.data.conversation_id);
      }

      const aiMessage = {
        role: 'ai',
        content: res.data.answer,
        sources: res.data.sources || [],
        responseTime: res.data.response_time
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat API error:', error);

      setMessages(prev => [
        ...prev,
        {
          role: 'ai',
          content:
            `Error: ${
              error.response?.data?.detail ||
              error.message ||
              'Unknown error'
            }`,
          sources: []
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const latestSources =
    messages.length > 0 &&
    messages[messages.length - 1].role === 'ai'
      ? messages[messages.length - 1].sources
      : [];

  return (
    <div className="flex h-full bg-[var(--background)] overflow-hidden">
      <div className="w-64 border-r border-[var(--border)] flex flex-col bg-[var(--background-card)]">
        <div className="p-4 border-b border-[var(--border)] flex items-center justify-between">
          <h2 className="font-semibold text-[var(--text-primary)]">
            Workspaces
          </h2>
          <button className="text-[var(--text-secondary)] hover:text-[var(--primary)]">
            <Plus size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {loadingWorkspaces ? (
            <div className="flex items-center justify-center p-4 text-[var(--text-secondary)]">
              <Loader2 className="animate-spin" size={20} />
            </div>
          ) : workspaces.length === 0 ? (
            <div className="text-sm text-[var(--text-secondary)] p-2 text-center">
              No workspaces found.
            </div>
          ) : (
            workspaces.map((ws) => (
              <button
                key={ws.id}
                onClick={() => handleWorkspaceChange(ws)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedWorkspace?.id === ws.id
                    ? 'bg-[var(--background)] border border-[var(--border)] shadow-sm text-[var(--primary)]'
                    : 'text-[var(--text-secondary)] hover:bg-[var(--background-hover)] hover:text-[var(--text-primary)]'
                }`}
              >
                {ws.name}
              </button>
            ))
          )}
        </div>

        <div className="p-4 border-t border-[var(--border)]">
          <button
            onClick={() => {
              setMessages([]);
              setConversationId(null);
            }}
            className="w-full flex items-center justify-center gap-2 bg-[var(--background)] border border-[var(--border)] text-[var(--text-primary)] px-4 py-2 rounded-lg font-medium shadow-sm hover:bg-[var(--background-hover)] transition-colors text-sm"
          >
            <Plus size={16} />
            New Chat
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="h-14 border-b border-[var(--border)] flex items-center justify-between px-6 bg-[var(--background-card)] shrink-0">
          <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
            <span>
              {selectedWorkspace
                ? selectedWorkspace.name
                : 'Select a Workspace'}
            </span>

            {selectedWorkspace && (
              <>
                <ChevronRight size={14} />
                <span className="font-medium text-[var(--text-primary)]">
                  Workspace Search
                </span>
              </>
            )}
          </div>

          <button className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
            <MoreVertical size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[var(--background)] scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-[var(--primary-soft)] rounded-full flex items-center justify-center mb-4">
                <Bot
                  size={32}
                  className="text-[var(--primary)]"
                />
              </div>

              <h3 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
                {selectedWorkspace
                  ? `Ask about ${selectedWorkspace.name}`
                  : 'Select a workspace to begin'}
              </h3>

              <p className="text-[var(--text-secondary)] max-w-md">
                {selectedWorkspace
                  ? "Ask a question and I'll search through the documents in this workspace to find the answer."
                  : "You must select a workspace from the left panel to search its documents."}
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className="flex gap-4 max-w-3xl mx-auto"
              >
                {msg.role === 'user' ? (
                  <div className="w-8 h-8 rounded-full bg-[var(--background-hover)] flex items-center justify-center text-sm font-medium flex-shrink-0 text-[var(--text-secondary)]">
                    <User size={16} />
                  </div>
                ) : (
                  <div className="w-8 h-8 rounded-lg bg-[var(--primary)] text-white flex items-center justify-center flex-shrink-0 shadow-sm">
                    <Bot size={18} />
                  </div>
                )}

                <div className="flex-1 pt-1 min-w-0">
                  <p className="font-medium text-[var(--text-primary)] text-sm mb-1">
                    {msg.role === 'user' ? 'You' : 'ContextAI'}
                  </p>

                  <div className="text-[var(--text-primary)] leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>

                    {msg.role === 'ai' &&
                      msg.responseTime !== undefined &&
                      msg.responseTime !== null && (
                        <div className="mt-2 text-xs text-[var(--text-secondary)]">
                          Response time: {msg.responseTime}s
                        </div>
                      )}

                    {msg.role === 'ai' &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {msg.sources.map((sourceId, i) => (
                            <span
                              key={i}
                              className="inline-flex items-center justify-center px-2 py-0.5 rounded-full bg-[var(--primary-soft)] text-[var(--primary-soft-text)] text-xs font-medium border border-[var(--primary-soft-border)]"
                            >
                              Source {i + 1}
                            </span>
                          ))}
                        </div>
                      )}
                  </div>
                </div>
              </div>
            ))
          )}

          {isTyping && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="w-8 h-8 rounded-lg bg-[var(--primary)] text-white flex items-center justify-center flex-shrink-0 shadow-sm">
                <Loader2
                  size={18}
                  className="animate-spin"
                />
              </div>

              <div className="flex-1 pt-1">
                <p className="font-medium text-[var(--text-primary)] text-sm mb-1">
                  ContextAI
                </p>

                <div className="flex gap-1 mt-2">
                  <div className="w-2 h-2 rounded-full bg-[var(--text-secondary)] animate-bounce"></div>
                  <div
                    className="w-2 h-2 rounded-full bg-[var(--text-secondary)] animate-bounce"
                    style={{ animationDelay: '0.15s' }}
                  ></div>
                  <div
                    className="w-2 h-2 rounded-full bg-[var(--text-secondary)] animate-bounce"
                    style={{ animationDelay: '0.3s' }}
                  ></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-[var(--background-card)] shrink-0 border-t border-[var(--border)]">
          <div className="max-w-3xl mx-auto mb-3 flex items-center gap-4">
            <button
              type="button"
              onClick={() => setRetrievalMode('fast')}
              title="Faster responses with strong document retrieval."
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                retrievalMode === 'fast'
                  ? 'bg-[var(--primary)] text-white'
                  : 'bg-[var(--background-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Zap size={14} />
              Fast
            </button>
            <button
              type="button"
              onClick={() => setRetrievalMode('accurate')}
              title="More thorough retrieval using additional reranking. May take longer."
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                retrievalMode === 'accurate'
                  ? 'bg-[var(--primary)] text-white'
                  : 'bg-[var(--background-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Target size={14} />
              Accurate
            </button>
          </div>
          
          <form
            onSubmit={handleSendMessage}
            className="max-w-3xl mx-auto relative"
          >
            <div
              className={`overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--background-input)] shadow-sm transition-all ${
                isTyping || !selectedWorkspace
                  ? 'opacity-70'
                  : 'focus-within:border-[var(--primary)] focus-within:ring-1 focus-within:ring-[var(--primary)]'
              }`}
            >
              <textarea
                rows={1}
                name="message"
                id="message"
                className="block w-full resize-none border-0 bg-transparent py-3 pl-4 pr-20 text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:ring-0 sm:text-sm sm:leading-6 disabled:bg-transparent"
                placeholder={
                  selectedWorkspace
                    ? 'Ask about your documents...'
                    : 'Select a workspace...'
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isTyping || !selectedWorkspace}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
              />

              <div className="absolute right-2 top-2 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => navigate('/upload')}
                  className="p-1.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-md transition-colors"
                  disabled={isTyping || !selectedWorkspace}
                >
                  <Paperclip size={18} />
                </button>

                <button
                  type="submit"
                  disabled={
                    !input.trim() ||
                    isTyping ||
                    !selectedWorkspace
                  }
                  className="p-1.5 bg-[var(--primary)] text-white rounded-md hover:bg-[var(--primary-hover)] transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>

            <div className="mt-2 text-center text-xs text-[var(--text-secondary)]">
              AI can make mistakes. Verify important information using citations.
            </div>
          </form>
        </div>
      </div>

      <div className="w-80 border-l border-[var(--border)] bg-[var(--background-card)] flex-col hidden xl:flex shrink-0">
        <div className="h-14 border-b border-[var(--border)] flex items-center px-4 bg-[var(--background-card)]">
          <h3 className="font-medium text-[var(--text-primary)] text-sm">
            Sources (Latest Answer)
          </h3>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {latestSources.length === 0 ? (
            <div className="text-center text-sm text-[var(--text-secondary)] mt-10">
              No sources to display.
            </div>
          ) : (
            latestSources.map((sourceId, index) => (
              <div
                key={index}
                className="bg-[var(--background)] p-3 rounded-lg border border-[var(--border)] shadow-sm hover:border-[var(--primary)] transition-colors cursor-pointer group flex items-center gap-3"
              >
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-[var(--primary-soft)] text-[var(--primary-soft-text)] shrink-0">
                  <FileText size={16} />
                </div>

                <div>
                  <h4 className="text-sm font-medium text-[var(--text-primary)] group-hover:text-[var(--primary)] transition-colors">
                    Source {index + 1}
                  </h4>
                  <p className="text-xs text-[var(--text-secondary)]">
                    Reference Document
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}