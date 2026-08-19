import type { ReactNode } from "react";

import type { AskResponse } from "../api/types";

interface Props {
  response: AskResponse;
}

/**
 * Renders inline [1] / [2] markers as visual chips rather than raw brackets,
 * so a cited answer reads like a report instead of like model output.
 */
function withCitationChips(answer: string): ReactNode[] {
  // Capturing group keeps the markers in the split output.
  return answer.split(/(\[\d+\])/g).map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    return match ? (
      <span key={i} className="cite">
        {match[1]}
      </span>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

export function AnswerCard({ response }: Props) {
  return (
    <div className="answer-card">
      <div className="answer-text">{withCitationChips(response.answer)}</div>
      <div className="answer-footer">
        {response.used_context
          ? `Answered from ${response.citations.length} excerpt${
              response.citations.length === 1 ? "" : "s"
            } in your files · ${response.model}`
          : "No matching content found in the index"}
      </div>
    </div>
  );
}
