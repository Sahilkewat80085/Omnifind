import { useState } from "react";

import { ApiError, api } from "../api/client";
import type { AskResponse, Health } from "../api/types";
import { AnswerCard } from "../components/AnswerCard";
import { Banner } from "../components/Banner";
import { LoadingIndicator } from "../components/LoadingIndicator";
import { ReferenceCard } from "../components/ReferenceCard";
import { RelatedImageCard } from "../components/RelatedImageCard";
import { SearchBar } from "../components/SearchBar";
import { isRelevant } from "../utils/relevance";

interface Props {
  health: Health | null;
}

export function AskPage({ health }: Props) {
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const aiDisabled = health !== null && !health.ai_enabled;

  async function handleAsk(query: string) {
    setBusy(true);
    setError(null);
    try {
      setResponse(await api.ask(query));
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setError(`${err.message} Then restart the backend so it picks up the key.`);
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
      setResponse(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Ask AI</h1>
        <p className="subtitle">
          Retrieval-augmented answers. OmniFind finds the most relevant passages in
          your own files and asks Gemini to answer from those alone — every claim is
          cited back to the page it came from.
        </p>
      </header>

      {aiDisabled && (
        <Banner kind="warning">
          No Gemini API key is configured, so answers are unavailable. Add{" "}
          <code>GEMINI_API_KEY</code> to <code>backend/.env</code> and restart the
          backend. Semantic search on the Search page works without a key.
        </Banner>
      )}

      <SearchBar
        placeholder="e.g. what did I write about database normalization?"
        buttonLabel="Ask"
        busy={busy}
        onSubmit={handleAsk}
      />

      {error && (
        <div style={{ marginTop: 20 }}>
          <Banner kind="error">{error}</Banner>
        </div>
      )}

      {busy && <LoadingIndicator label="Retrieving passages and generating an answer…" />}

      {!busy && response && (() => {
        const relatedImages = response.related_images.filter((i) => isRelevant(i.similarity));

        return (
          <>
            <div className="section-label">Answer</div>
            <AnswerCard response={response} />

            {response.citations.length > 0 && (
              <>
                <div className="section-label" style={{ marginTop: 28 }}>
                  Sources ({response.citations.length})
                </div>
                {response.citations.map((c) => (
                  <ReferenceCard key={c.marker} citation={c} />
                ))}
              </>
            )}

            {relatedImages.length > 0 && (
              <>
                <div className="section-label" style={{ marginTop: 28 }}>
                  Related pictures ({relatedImages.length})
                </div>
                <p className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
                  Found by visual similarity to your question. The AI cannot read what
                  is in them, only find pictures that look like what was described.
                </p>
                <div className="image-grid">
                  {relatedImages.map((img) => (
                    <RelatedImageCard key={img.file_id} image={img} />
                  ))}
                </div>
              </>
            )}
          </>
        );
      })()}
    </>
  );
}
