"""Tests for portfolio page."""
import os
import sys
import importlib
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_portfolio():
    """Import portfolio module from the numbered file."""
    pages_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'pages')
    for f in os.listdir(pages_dir):
        if 'portfolio' in f.lower():
            spec = importlib.util.spec_from_file_location("portfolio", os.path.join(pages_dir, f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("portfolio module not found")


class TestPortfolioPage:
    """Tests for the portfolio page module."""
    
    def test_import(self):
        mod = _import_portfolio()
        assert hasattr(mod, 'render_portfolio_page')
    
    def test_get_current_signals_returns_dataframe(self):
        mod = _import_portfolio()
        result = mod.get_current_signals(['0700'])
        assert isinstance(result, pd.DataFrame)
