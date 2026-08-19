import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { Health } from "../api/types";

const POLL_INTERVAL_MS = 10_000;

export interface HealthState {
  health: Health | null;
  /** null while the very first probe is still in flight. */
  online: boolean | null;
  refresh: () => void;
}

/**
 * Polls /health so the sidebar can show backend status and the Ask page
 * knows up front whether a Gemini key is configured.
 */
export function useHealth(): HealthState {
  const [health, setHealth] = useState<Health | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  const refresh = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setOnline(true);
      })
      .catch(() => {
        setHealth(null);
        setOnline(false);
      });
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return { health, online, refresh };
}
