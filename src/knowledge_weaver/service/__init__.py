"""KW 2.0 HTTP-over-Unix-socket service."""

from .app import ServiceApp
from .errors import ApiError, error_payload, safe_message

__all__ = ["ApiError", "ServiceApp", "error_payload", "safe_message"]
