"""Tests for backtest page."""
import os
import sys
import importlib
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_backtest():
    """Import backtest module from the numbered file."""
    pages_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'pages')
    # Find the backtest file (may be numbered)
    for f in os.listdir(pages_dir):
        if 'backtest' in f.lower():
            spec = importlib.util.spec_from_file_location("backtest", os.path.join(pages_dir, f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("backtest module not found")


class TestBacktestPage:
    """Tests for the backtest page module."""
    
    def test_import(self):
        mod = _import_backtest()
        assert hasattr(mod, 'render_backtest_page')
    
    def test_get_predictions_returns_dataframe(self):
        mod = _import_backtest()
        result = mod.get_predictions('0700', '2026-01-01', '2026-06-01', '5d')
        assert isinstance(result, pd.DataFrame)
