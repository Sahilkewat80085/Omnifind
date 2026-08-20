import { useState } from "react";

import { api } from "../api/client";
import type { RelatedImage } from "../api/types";

interface Props {
  image: RelatedImage;
}

export function RelatedImageCard({ image }: Props) {
  const [failed, setFailed] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);

  async function handleOpen() {
    setOpenError(null);
    try {
      await api.openFile(image.path);
    } catch (err) {
      setOpenError(err instanceof Error ? err.message : "Could not open file");
    }
  }

  return (
    <div className="image-card">
      {failed ? (
        <div className="image-card-missing">File unavailable</div>
      ) : (
        <img
          className="image-card-thumb"
          src={api.fileContentUrl(image.file_id)}
          alt={image.file_name}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}

      <div className="image-card-meta">
        <div className="image-card-name" title={image.path}>
          {image.file_name}
        </div>
        <div className="result-meta">
          {image.width} × {image.height}
        </div>
        {openError && (
          <p className="result-meta" style={{ color: "var(--danger)", marginTop: 4 }}>
            {openError}
          </p>
        )}
        <button className="btn secondary small" onClick={handleOpen} style={{ marginTop: 8 }}>
          Open
        </button>
      </div>
    </div>
  );
}
