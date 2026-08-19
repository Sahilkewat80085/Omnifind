import type { ReactNode } from "react";

interface Props {
  kind: "error" | "warning" | "success" | "info";
  children: ReactNode;
}

export function Banner({ kind, children }: Props) {
  return <div className={`banner ${kind}`}>{children}</div>;
}
