#!/usr/bin/env python3
from nndp.errors import NotBuiltError, AlreadyBuiltError


def require_built(function):
    def require(*args, **kwargs):
        if not args[0].is_built():
            raise NotBuiltError(
                f"attempt to operate on an non-built {args[0].__class__.__name__}."
            )
        return function(*args, **kwargs)
    return require


def require_not_built(function):
    def require(*args, **kwargs):
        if args[0].is_built():
            raise AlreadyBuiltError(
                f"attempt to operate on an built {args[0].__class__.__name__}."
            )
        return function(*args, **kwargs)
    return require