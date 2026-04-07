"""
Shared utilities — retry decorator, validation helpers
"""
import time
import functools
import requests


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0,
                     exceptions=(requests.RequestException, ConnectionError, TimeoutError)):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = backoff_factor * (2 ** attempt)
                        time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator


def validate_gene_name(name: str) -> bool:
    """Check if a string looks like a valid gene symbol."""
    if not name or len(name) < 2 or len(name) > 15:
        return False
    # Must start with letter
    if not name[0].isalpha():
        return False
    # Only alphanumeric + dash
    if not all(c.isalnum() or c in "-_" for c in name):
        return False
    return True
