#!/usr/bin/env python3


class IncompatibleLayersError(Exception):

    def __init__(self, message):
        super().__init__(message)
