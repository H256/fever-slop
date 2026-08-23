"""Deprecation helpers for the feverslop public API."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def deprecated(
    reason: str = "",
    *,
    since: str = "",
    alternative: str = "",
) -> Callable[[T], T]:
    """Decorator that emits a DeprecationWarning on each call (functions)
    or each instantiation (classes).

    Usage on a function:
        @deprecated("Use new_func instead", since="0.2.0", alternative="feverslop.new_func")
        def old_func(): ...

    Usage on a class:
        @deprecated("Use NewClass", since="0.2.0", alternative="feverslop.NewClass")
        class OldClass: ...
    """
    def decorator(obj: T) -> T:
        msg_parts = [f"{obj.__name__} is deprecated"]
        if since:
            msg_parts.append(f"since {since}")
        if reason:
            msg_parts.append(reason)
        if alternative:
            msg_parts.append(f"Use {alternative} instead")
        msg = ". ".join(msg_parts)

        if isinstance(obj, type):
            # Class: warn on instantiation
            original_init = obj.__init__

            @functools.wraps(original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[assignment]
            return obj

        # Function/method
        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
