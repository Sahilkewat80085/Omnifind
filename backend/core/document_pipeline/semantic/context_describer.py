from core.document_pipeline.models import NormalizedDocument


class ContextDescriber:
    """Generates an intuitive 'what this document is about' description for retrieval and explainability."""

    def describe(self, doc: NormalizedDocument, keywords: list[str], summary: str) -> str:
        file_type_labels = {
            "pdf": "PDF document",
            "docx": "Word document",
            "pptx": "PowerPoint presentation",
            "xlsx": "Excel workbook",
            "md": "Markdown document",
            "txt": "Text document",
        }
        type_str = file_type_labels.get(doc.file_type, f"{doc.file_type.upper()} file")

        # Format-specific context nuances
        extras = []
        if doc.format_metadata.page_count:
            extras.append(f"{doc.format_metadata.page_count} pages")
        if doc.format_metadata.slide_count:
            extras.append(f"{doc.format_metadata.slide_count} slides")
        if doc.format_metadata.sheet_names:
            extras.append(f"sheets: {', '.join(doc.format_metadata.sheet_names[:4])}")
        if doc.tables:
            extras.append(f"{len(doc.tables)} structured tables")

        meta_detail = f" ({', '.join(extras)})" if extras else ""

        kw_str = ", ".join(keywords[:4]) if keywords else "general topics"

        description = f"A {type_str}{meta_detail} focusing on {kw_str}."
        if summary and len(summary) > 30:
            description += f" Key highlight: {summary[:160].rstrip('.')}."

        return description
