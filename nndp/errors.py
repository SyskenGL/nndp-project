#!/usr/bin/env python3


class IncompatibleLayersError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)


class EmptyModelError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)


class NotBuiltError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)


class AlreadyBuiltError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)