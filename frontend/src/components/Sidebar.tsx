import type { Health } from "../api/types";

export type Page = "dashboard" | "index" | "search" | "ask" | "settings";

const NAV: { id: Page; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "▤" },
  { id: "index", label: "Index folder", icon: "⊕" },
  { id: "search", label: "Search", icon: "⌕" },
  { id: "ask", label: "Ask AI", icon: "✦" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

interface Props {
  page: Page;
  onNavigate: (page: Page) => void;
  health: Health | null;
  online: boolean | null;
}

export function Sidebar({ page, onNavigate, health, online }: Props) {
  const statusClass = online === null ? "pending" : online ? "online" : "offline";
  const statusText =
    online === null
      ? "Connecting…"
      : online
        ? `Backend online · ${health?.env ?? ""}`
        : "Backend offline";

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">⌕</div>
        <div>
          <div className="brand-name">OmniFind</div>
          <div className="brand-tag">Semantic file search</div>
        </div>
      </div>

      <nav className="nav">
        {NAV.map((item) => (
          <button
            key={item.id}
            className={`nav-item${page === item.id ? " active" : ""}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-line">
          <span className={`status-dot ${statusClass}`} />
          {statusText}
        </div>
        {online && (
          // Whether AI is available still matters — it decides if the Ask page
          // can answer at all. Which model serves it does not, so the name is
          // no longer shown here; Settings has it for anyone who needs it.
          <div className="status-line" style={{ marginTop: 6 }}>
            <span className={`status-dot ${health?.ai_enabled ? "online" : "pending"}`} />
            {health?.ai_enabled ? "AI ready" : "AI: no API key"}
          </div>
        )}
      </div>
    </aside>
  );
}
