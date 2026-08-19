interface Props {
  label?: string;
}

export function LoadingIndicator({ label = "Loading…" }: Props) {
  return (
    <div className="loading-row">
      <span className="spinner" />
      <span>{label}</span>
    </div>
  );
}
