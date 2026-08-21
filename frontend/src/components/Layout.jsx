import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function Layout() {
  return (
    <div className="flex h-screen w-full bg-[var(--background-subtle)] overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-[var(--background)]">
        <Outlet />
      </main>
    </div>
  );
}
