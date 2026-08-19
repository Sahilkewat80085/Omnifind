import { useState } from "react";

import { api } from "../api/client";
import type { RelatedImage } from "../api/types";

interface Props {
  image: RelatedImage;
}

/**
 * A picture the question matched visually. Deliberately not a ReferenceCard:
 * no [n] marker, because nothing in the answer cites it — the model never saw
 * this image, CLIP just found it similar to the question.
 *
 * The thumbnail streams from the backend rather than the file:// path in
 * `image.path`, which a page served over http cannot load. Same reason the
 * Search results use api.fileContentUrl.
 */
export function RelatedImageCard({ image }: Props) {
  const [failed, setFailed] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);

  const percent = Math.round(Math.max(0, Math.min(1, image.similarity)) * 100);

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
        // The file was indexed but has since moved or been deleted; say so
        // rather than leaving a broken-image icon.
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
          {image.width} × {image.height} · {percent}% match
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
