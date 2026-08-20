from abc import ABC, abstractmethod
from core.document_pipeline.models import NormalizedDocument


class BaseExtractor(ABC):
    """Abstract interface for all format-specific document extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> NormalizedDocument:
        """Extracts text blocks, structured tables, and format-specific metadata from a file.

        Args:
            file_path: Absolute or relative path to the target document.

        Returns:
            NormalizedDocument intermediate representation.
        """
        pass
