import re
from collections import Counter

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "table", "slide", "sheet", "page", "section", "document"
}


class KeywordExtractor:
    """Extracts high-signal keywords and 2-gram keyphrases from normalized text."""

    def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
        if not text:
            return []

        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        meaningful = [t for t in tokens if t not in _STOPWORDS]

        # 1-gram frequencies
        unigram_counts = Counter(meaningful)

        # 2-gram frequencies
        bigrams = []
        for i in range(len(meaningful) - 1):
            w1, w2 = meaningful[i], meaningful[i + 1]
            if w1 != w2:
                bigrams.append(f"{w1} {w2}")
        bigram_counts = Counter(bigrams)

        combined: list[tuple[str, int]] = []
        # Score bigrams with a boost
        for bg, count in bigram_counts.most_common(top_n):
            if count >= 2:
                combined.append((bg, count * 3))

        for ug, count in unigram_counts.most_common(top_n * 2):
            combined.append((ug, count))

        combined.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        results: list[str] = []
        for phrase, _ in combined:
            if phrase not in seen:
                seen.add(phrase)
                results.append(phrase)
            if len(results) >= top_n:
                break

        return results
