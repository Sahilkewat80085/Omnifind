import re
from core.document_pipeline.models import EntityItem


class EntityRecognizer:
    """Extracts named entities (PERSON, ORG, GPE/LOC, DATE, MONEY) with spaCy and regex fallback."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except Exception:
            self._nlp = None

    def extract_entities(self, text: str) -> list[EntityItem]:
        if not text:
            return []

        # Bound sample text for entity extraction efficiency (first 25,000 chars)
        sample = text[:25000]

        if self._nlp is not None:
            doc = self._nlp(sample)
            entities: list[EntityItem] = []
            seen = set()
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"):
                    key = (ent.text.strip(), ent.label_)
                    if key not in seen and len(ent.text.strip()) > 1:
                        seen.add(key)
                        entities.append(
                            EntityItem(
                                text=ent.text.strip(),
                                type=ent.label_,
                                start=ent.start_char,
                                end=ent.end_char,
                            )
                        )
            return entities[:50]

        # Rule-based fallback if spaCy model is not installed locally
        return self._regex_fallback_entities(sample)

    def _regex_fallback_entities(self, text: str) -> list[EntityItem]:
        entities: list[EntityItem] = []
        # Dates (e.g. 2026-08-20, August 20, 2026, 20/08/2026)
        date_pattern = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b")
        for m in date_pattern.finditer(text):
            entities.append(EntityItem(text=m.group(), type="DATE", start=m.start(), end=m.end()))

        # Money (e.g. $1,200, ₹50,000, 100 USD)
        money_pattern = re.compile(r"[\$€£₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*(?:USD|EUR|INR|GBP)\b")
        for m in money_pattern.finditer(text):
            entities.append(EntityItem(text=m.group(), type="MONEY", start=m.start(), end=m.end()))

        return entities[:30]
