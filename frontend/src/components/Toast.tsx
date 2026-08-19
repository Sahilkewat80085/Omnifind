import { useEffect } from "react";

export interface ToastMessage {
  id: string;
  title: string;
  description: string;
  action: "indexed" | "removed";
}

interface Props {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ toasts, onDismiss }: Props) {
  if (toasts.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        maxWidth: 380,
      }}
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastMessage;
  onDismiss: (id: string) => void;
}) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss(toast.id);
    }, 4500);
    return () => clearTimeout(timer);
  }, [toast.id, onDismiss]);

  const isIndexed = toast.action === "indexed";

  return (
    <div
      style={{
        background: "var(--color-surface, #1e1e24)",
        border: `1px solid ${
          isIndexed
            ? "var(--color-success, #22c55e)"
            : "var(--color-danger, #ef4444)"
        }`,
        boxShadow: "0 8px 24px rgba(0, 0, 0, 0.4)",
        borderRadius: 8,
        padding: "12px 16px",
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        animation: "fadeIn 0.25s ease-out",
        color: "var(--color-text, #ffffff)",
      }}
    >
      <div style={{ fontSize: 18, marginTop: 1 }}>{isIndexed ? "⚡" : "🗑"}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontWeight: 600,
            fontSize: 13.5,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span>{toast.title}</span>
          <span
            style={{
              fontSize: 10.5,
              padding: "1px 5px",
              borderRadius: 3,
              background: isIndexed
                ? "rgba(34, 197, 94, 0.2)"
                : "rgba(239, 68, 68, 0.2)",
              color: isIndexed ? "#22c55e" : "#ef4444",
            }}
          >
            {isIndexed ? "Auto-Indexed" : "Removed"}
          </span>
        </div>
        <div
          className="mono"
          style={{
            fontSize: 12,
            marginTop: 3,
            color: "var(--color-muted, #94a3b8)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          title={toast.description}
        >
          {toast.description}
        </div>
      </div>
      <button
        onClick={() => onDismiss(toast.id)}
        style={{
          background: "transparent",
          border: "none",
          color: "var(--color-muted, #94a3b8)",
          cursor: "pointer",
          fontSize: 14,
          padding: "0 2px",
        }}
      >
        ×
      </button>
    </div>
  );
}
