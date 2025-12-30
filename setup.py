"""
Setup script for the multitask library.

This file provides backward compatibility for systems that don't support pyproject.toml.
All configuration is now in pyproject.toml - this file mainly delegates to it.
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for backward compatibility
setup()
