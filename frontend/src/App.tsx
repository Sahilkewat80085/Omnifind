import { useEffect, useRef, useState } from "react";

import { api } from "./api/client";
import { Banner } from "./components/Banner";
import { Sidebar, type Page } from "./components/Sidebar";
import { ToastContainer, type ToastMessage } from "./components/Toast";
import { useHealth } from "./hooks/useHealth";
import { Dashboard } from "./pages/Dashboard";
import { IndexPage } from "./pages/IndexPage";
import { SearchPage } from "./pages/SearchPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const { health, online, refresh } = useHealth();
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const lastActivityTimestamp = useRef<number>(Date.now() / 1000);

  // Poll for background filesystem watcher indexing activity
  useEffect(() => {
    if (online === false) return;

    const interval = setInterval(() => {
      api
        .watcherActivity(lastActivityTimestamp.current)
        .then((activities) => {
          if (activities.length > 0) {
            const latest = Math.max(...activities.map((a) => a.timestamp));
            lastActivityTimestamp.current = latest;

            const newToasts: ToastMessage[] = activities.map((a) => ({
              id: `${a.path}-${a.timestamp}-${Math.random()}`,
              title: a.file_name,
              description: a.path,
              action: a.action,
            }));

            setToasts((prev) => [...prev, ...newToasts]);
          }
        })
        .catch(() => {});
    }, 1500);

    return () => clearInterval(interval);
  }, [online]);

  function handleDismissToast(id: string) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  return (
    <div className="app">
      <Sidebar page={page} onNavigate={setPage} health={health} online={online} />

      <main className="main">
        {online === false && page !== "settings" && (
          <Banner kind="error">
            Cannot reach the backend. Start it from <code>omnifind/backend</code> with{" "}
            <code>uvicorn main:app --reload --port 8000</code>, or change the URL under
            Settings.
          </Banner>
        )}

        {page === "dashboard" && <Dashboard onNavigate={setPage} />}
        {page === "index" && <IndexPage />}
        {page === "search" && <SearchPage />}
        {page === "settings" && (
          <SettingsPage health={health} online={online} onRefresh={refresh} />
        )}
      </main>

      <ToastContainer toasts={toasts} onDismiss={handleDismissToast} />
    </div>
  );
}
