"""DataProvider contract, Axis 2 of the platform. Every source implements this."""

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def fetch_bars(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return bars in canonical schema (see nakagai.data.schema), left-edge UTC labels."""
