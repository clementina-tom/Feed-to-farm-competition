"""
Data pipeline components.

Responsible for ingesting CSV files, handling memory downcasting,
and formatting initial dataset types.
"""
from .loader import DataLoader

__all__ = ["DataLoader"]
