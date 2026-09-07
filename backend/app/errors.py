class ApiError(Exception):
    """A safe error that can be returned to an API client."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class EmbeddingProviderError(Exception):
    """A normalized embedding failure with a client-safe message."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.public_message = message
        self.status_code = status_code


class StorageError(Exception):
    """Raised when persisted source or vector data is inconsistent."""
