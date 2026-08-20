class ModelsNotAvailableError(RuntimeError):
    """The embedding weights are not in the local cache and cannot be fetched.

    Raised in place of the HuggingFace OSError, which opens with "We couldn't
    connect to 'https://huggingface.co'" and closes by telling the user to
    check their internet connection. Both read as though the app needs the
    network to run. It does not: it needs it once, to install. This message
    names the step that was missed instead of the host that was unreachable.

    The text is ASCII on purpose. It goes to a Windows console as well as to
    the UI, and cp1252 turns an em dash into a question mark.
    """

    def __init__(self, model_name: str, cause: Exception) -> None:
        # First line only. The full HuggingFace error is several lines of
        # troubleshooting advice aimed at the opposite problem, and repeating
        # it here would tell the user to do the one thing that will not help.
        summary = str(cause).strip().splitlines()[0] if str(cause).strip() else type(cause).__name__

        super().__init__(
            f"Embedding model '{model_name}' is not downloaded. OmniFind runs "
            f"fully offline, but the models must be fetched once during setup. "
            f"Connect to the internet and run:  python scripts/fetch_models.py"
            f"  [{type(cause).__name__}: {summary}]"
        )
        self.model_name = model_name
