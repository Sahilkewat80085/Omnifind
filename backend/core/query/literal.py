"""Which of a query's words a file must actually contain to be shown.

The Search page answers a *containment* question, not a similarity question.
"Files with my college name in them" is answered correctly only by files that
have those words in them; a file that is merely *about* colleges is a wrong
answer however high its cosine score. So similarity is demoted to a tie-break:
it may reorder the files that already passed this gate, and may never add one.

Two rules keep the gate strict without making it brittle:

* **Words the index has never seen are dropped from the demand.** A file cannot
  be required to contain a word that appears nowhere, or a single typo would
  empty the whole result list. This is what BM25 does with a zero document
  frequency term, for the same reason. If *none* of the words are findable
  there is genuinely nothing to show, and the search returns empty.
* **Matching is by word stem and prefix, not by exact string.** "college" has
  to find "colleges", and "normalization" has to find "normalizing", or the
  gate rejects files the user can plainly see are matches. Short words (<= 3
  characters) still demand an exact stem, because prefix-matching "ai" against
  "airport" is how a strict filter quietly stops being strict.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass

from core.query.intent import TYPE_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that carry no identity of their own. Requiring a file to contain "the"
# is not a filter, and dropping these is what lets "how much was the fee?"
# reduce to the one word that actually names what is being looked for.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "as", "into", "than", "then",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "have", "has", "had",
    "how", "what", "why", "when", "where", "who", "whom", "whose", "which",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "about", "any", "some", "all", "much", "many", "more", "most",
    "please", "find", "show", "search", "give", "need", "want", "get",
    "file", "files", "folder", "folders",
}

_MIN_STEM = 4
"""Below this length a stem is too generic to prefix-match anything safely."""

# Stripped longest-first, and only when enough of the word survives. The pairs
# that have to come out equal are the ones a person would call the same word:
# college/colleges, note/notes, normalization/normalizing. A plain plural is
# allowed to leave a 3-letter stem so "fee" and "fees" meet; every other suffix
# holds the line at four, because "notes" reduced to "not" would start matching
# files that merely say "not".
_SUFFIXES = (
    ("ization", _MIN_STEM), ("isation", _MIN_STEM),
    ("ations", _MIN_STEM), ("ation", _MIN_STEM),
    ("ings", _MIN_STEM), ("ing", _MIN_STEM),
    ("ions", _MIN_STEM), ("ion", _MIN_STEM),
    ("ives", _MIN_STEM), ("ive", _MIN_STEM),
    ("ies", _MIN_STEM), ("ied", _MIN_STEM),
    ("es", _MIN_STEM), ("ed", _MIN_STEM),
    ("s", 3),
    ("e", _MIN_STEM),
)

# "normalizing" loses "ing" and lands on "normaliz"; "normalization" loses
# "ation" and lands on "normal". One more pass over the verb-forming "iz"/"is"
# brings the first to the second. Without it the two spellings of one word
# produce two different stems, and the gate rejects a file the user can see is
# a match.
_VERB_ROOTS = ("iz", "is")


def stem(word: str) -> str:
    """Reduce a word to the root it shares with its own inflections."""
    for suffix, min_stem in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            word = word[: -len(suffix)]
            break
    for root in _VERB_ROOTS:
        if word.endswith(root) and len(word) - len(root) >= _MIN_STEM:
            return word[: -len(root)]
    return word


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class StemIndex:
    """Every word stem of one file, answerable by prefix.

    Built per file per search rather than stored, because it is only ever a few
    thousand short strings and keeping it on disk would mean a migration and a
    re-index for a structure that rebuilds in milliseconds.
    """

    def __init__(self) -> None:
        self._stems: set[str] = set()
        self._sorted: list[str] | None = None

    def add(self, text: str) -> None:
        self._stems.update(stem(word) for word in tokenize(text))
        self._sorted = None

    def merge(self, other: "StemIndex") -> None:
        self._stems |= other._stems
        self._sorted = None

    def contains(self, query_stem: str) -> bool:
        if query_stem in self._stems:
            return True
        if len(query_stem) < _MIN_STEM:
            # Short words get exact treatment only: "aws" must be "aws".
            return False

        # A longer word that *starts with* the query counts: "check" is allowed
        # to find "checkpw". The reverse is not - a search for "database" must
        # not be satisfied by a file that only ever says "data", which is what
        # made this gate quietly permissive when the rule ran both ways. Sorted
        # stems put the candidates together, so the first entry at or after the
        # query is the only one that can start with it.
        if self._sorted is None:
            self._sorted = sorted(self._stems)
        position = bisect_left(self._sorted, query_stem)
        return position < len(self._sorted) and self._sorted[position].startswith(query_stem)

    def __bool__(self) -> bool:
        return bool(self._stems)


@dataclass(frozen=True)
class LiteralTerms:
    """What a file has to contain before it is allowed on screen."""

    tokens: tuple[str, ...]
    """Content words of the query, stopwords and file-type words removed."""

    phrase: str | None = None
    """Set when the user used quotes: this must appear verbatim, or nothing does."""

    @property
    def is_empty(self) -> bool:
        """True when the query demanded nothing literal - e.g. just "images"."""
        return not self.tokens and not self.phrase

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(stem(token) for token in self.tokens)


def extract_literal_terms(query: str) -> LiteralTerms:
    """Read the words a file must contain out of what the user typed.

    File-type words go too, not just stopwords: `detect_intent` has already
    turned "photo" into a filter over which partitions are searched, so also
    demanding the letters "photo" be inside the file would reject every image
    that is not literally captioned.
    """
    stripped = query.strip()
    phrase: str | None = None
    if len(stripped) > 2 and stripped.startswith('"') and stripped.endswith('"'):
        phrase = " ".join(stripped[1:-1].lower().split()) or None

    tokens = tuple(
        token
        for token in tokenize(stripped)
        if token not in STOPWORDS and token not in TYPE_WORDS
    )
    return LiteralTerms(tokens=tokens, phrase=phrase)
