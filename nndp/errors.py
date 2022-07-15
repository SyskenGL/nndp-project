#!/usr/bin/env python3


class IncompatibleLayersError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)


class NotBuiltLayerError(Exception):

    def __init__(self, message: str = None):
        super().__init__(message)
