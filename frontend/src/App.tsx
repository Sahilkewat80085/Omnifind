import { useState } from "react";

import { Banner } from "./components/Banner";
import { Sidebar, type Page } from "./components/Sidebar";
import { useHealth } from "./hooks/useHealth";
import { AskPage } from "./pages/AskPage";
import { Dashboard } from "./pages/Dashboard";
import { IndexPage } from "./pages/IndexPage";
import { SearchPage } from "./pages/SearchPage";
import { SettingsPage } from "./pages/SettingsPage";

/**
 * Navigation is plain component state rather than a router. The app has five
 * flat screens with no deep links or URL parameters, and keeping it
 * router-free avoids the file:// base-path problem when this same bundle is
 * later wrapped in Tauri.
 */
export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const { health, online, refresh } = useHealth();

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
        {page === "ask" && <AskPage health={health} />}
        {page === "settings" && (
          <SettingsPage health={health} online={online} onRefresh={refresh} />
        )}
      </main>
    </div>
  );
}
