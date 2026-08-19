import re
from dataclasses import dataclass

from models.schemas.file_schemas import FileType

# Words that name a *kind of file* rather than describe its contents.
#
# Kept deliberately conservative. A word earns its place here only if it is
# almost never the subject of a search in its own right: "photo" is a file
# type, "portrait" is a subject. The cost of a wrong entry is high — it would
# silently hide every result of the other two types — while the cost of a
# missing one is only that the search behaves as it did before.
_IMAGE_WORDS = {
    "image", "images", "img", "imgs",
    "photo", "photos", "photograph", "photographs",
    "picture", "pictures", "pic", "pics",
    "screenshot", "screenshots",
    "jpg", "jpeg", "png",
    "thumbnail", "thumbnails", "wallpaper", "selfie", "selfies",
}
_DOCUMENT_WORDS = {
    "pdf", "pdfs", "document", "documents", "doc", "docs", "docx", "txt",
}
# "class" and "method" are excluded on purpose: in a student's own files
# "class notes" and "research method" are far more likely than either sense
# meant here.
_CODE_WORDS = {
    "code", "sourcecode", "function", "functions",
    "script", "scripts", "snippet", "snippets",
    "py", "js", "ts", "tsx", "jsx",
}

_WORDS_BY_TYPE = {
    FileType.image: _IMAGE_WORDS,
    FileType.document: _DOCUMENT_WORDS,
    FileType.code: _CODE_WORDS,
}

# Left behind once the type word is removed: "picture of a dog" → "of a dog".
# Trimming these gives the embedding "dog", which is what was actually meant.
_EDGE_STOPWORDS = {"of", "a", "an", "the", "for", "with", "in", "on", "any", "some", "all"}

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QueryIntent:
    """What the user asked for, split from how they asked for it."""

    query: str  # what gets embedded, with the type word removed
    file_type: FileType | None  # None means "no preference stated"


def detect_intent(raw_query: str) -> QueryIntent:
    """Read a file type out of the query, if one is stated plainly.

    Searching "mountain image" and being handed a PDF that happens to discuss
    mountains is a wrong answer, however good the semantic match: the user
    named the type they wanted. This turns that word into a filter and strips
    it from the text that gets embedded, since "image" describes the container
    rather than the content and only blurs the query vector.

    Ambiguity resolves to no filter. "image processing code" names two types
    and means neither as a filter, so it searches everything — the previous
    behaviour, which is never wrong, only unhelpful.
    """
    tokens = _TOKEN.findall(raw_query.lower())
    if not tokens:
        return QueryIntent(query=raw_query, file_type=None)

    matched = {
        file_type: words & set(tokens)
        for file_type, words in _WORDS_BY_TYPE.items()
        if words & set(tokens)
    }
    if len(matched) != 1:
        return QueryIntent(query=raw_query, file_type=None)

    file_type, type_words = next(iter(matched.items()))

    kept = [t for t in tokens if t not in type_words]
    while kept and kept[0] in _EDGE_STOPWORDS:
        kept.pop(0)
    while kept and kept[-1] in _EDGE_STOPWORDS:
        kept.pop()

    # "images" on its own is a type with no subject. There is nothing left to
    # embed, so search on the original wording and let the filter do the work.
    cleaned = " ".join(kept) if kept else raw_query
    return QueryIntent(query=cleaned, file_type=file_type)
