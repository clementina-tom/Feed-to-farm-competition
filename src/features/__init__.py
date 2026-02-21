"""
Feature Engineering components.

Contains logic to create rolling aggregates, universal product trends,
customer momentum features, and time-based seasonality indicators.
"""
from .engineer import FeatureEngineer

__all__ = ["FeatureEngineer"]
