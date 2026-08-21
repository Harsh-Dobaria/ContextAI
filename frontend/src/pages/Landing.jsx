import { Link } from 'react-router-dom';
import { Database, Search, Shield, Zap, ArrowRight } from 'lucide-react';

const features = [
  {
    name: 'Unified Knowledge Base',
    description: 'Connect all your data sources into a single, intelligent repository that your team can instantly query.',
    icon: Database,
  },
  {
    name: 'Advanced RAG Pipeline',
    description: 'State-of-the-art retrieval-augmented generation ensures accurate, hallucination-free answers backed by citations.',
    icon: Search,
  },
  {
    name: 'Enterprise Security',
    description: 'Bank-grade encryption, role-based access control, and complete data privacy for your most sensitive documents.',
    icon: Shield,
  },
  {
    name: 'Lightning Fast',
    description: 'Optimized vector search and caching mechanisms provide sub-second query responses even across millions of documents.',
    icon: Zap,
  },
];

export default function Landing() {
  return (
    <div className="bg-[var(--background)] min-h-screen">
      <header className="absolute inset-x-0 top-0 z-50">
        <nav className="flex items-center justify-between p-6 lg:px-8" aria-label="Global">
          <div className="flex lg:flex-1 items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--primary)] text-white">
              <Database size={18} />
            </div>
            <span className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">ContextAI</span>
          </div>
          <div className="flex flex-1 justify-end">
            <Link to="/dashboard" className="text-sm font-semibold leading-6 text-[var(--text-primary)] hover:text-[var(--primary)] transition-colors">
              Log in <span aria-hidden="true">&rarr;</span>
            </Link>
          </div>
        </nav>
      </header>

      <main className="isolate">
        {/* Hero section */}
        <div className="relative pt-14">
          <div className="py-24 sm:py-32 lg:pb-40">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
              <div className="mx-auto max-w-2xl text-center">
                <h1 className="text-4xl font-bold tracking-tight text-[var(--text-primary)] sm:text-6xl">
                  Enterprise Intelligence, <span className="text-[var(--primary)]">Unified.</span>
                </h1>
                <p className="mt-6 text-lg leading-8 text-[var(--text-secondary)]">
                  The multi-workspace RAG platform built for modern teams. Securely chat with your documents, extract insights, and accelerate decision-making without compromising privacy.
                </p>
                <div className="mt-10 flex items-center justify-center gap-x-6">
                  <Link
                    to="/dashboard"
                    className="rounded-lg bg-[var(--primary)] px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-[var(--primary-hover)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] flex items-center gap-2 transition-all"
                  >
                    Enter Workspace <ArrowRight size={16} />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature section */}
        <div className="mx-auto max-w-7xl px-6 lg:px-8 pb-24 sm:pb-32">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-[var(--primary)]">Deploy Faster</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-[var(--text-primary)] sm:text-4xl">
              Everything you need to scale AI
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-4xl">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 lg:gap-y-16">
              {features.map((feature) => (
                <div key={feature.name} className="relative pl-16">
                  <dt className="text-base font-semibold leading-7 text-[var(--text-primary)]">
                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50">
                      <feature.icon className="h-6 w-6 text-[var(--primary)]" aria-hidden="true" />
                    </div>
                    {feature.name}
                  </dt>
                  <dd className="mt-2 text-base leading-7 text-[var(--text-secondary)]">{feature.description}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </main>
    </div>
  );
}
