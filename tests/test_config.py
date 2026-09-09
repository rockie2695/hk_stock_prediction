"""
Tests for config.py - environment variable loading and validation.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_env_file_exists(self):
        """Test that .env file exists."""
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        assert os.path.exists(env_path), ".env file not found"
    
    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        env_example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.example')
        assert os.path.exists(env_example_path), ".env.example file not found"
    
    def test_config_imports(self):
        """Test that config module can be imported."""
        import config
        assert hasattr(config, 'SUPABASE_URL')
        assert hasattr(config, 'SUPABASE_KEY')
        assert hasattr(config, 'STOCK_LIST')
        assert hasattr(config, 'USE_ENSEMBLE')
        assert hasattr(config, 'USE_STACKING')
        assert hasattr(config, 'USE_BLENDING')
        assert hasattr(config, 'USE_CATBOOST')
        assert hasattr(config, 'USE_SMOTE')
    
    def test_stock_list_parsing(self):
        """Test that STOCK_LIST is parsed correctly."""
        import config
        assert isinstance(config.STOCK_LIST, list)
        assert len(config.STOCK_LIST) > 0
        for code in config.STOCK_LIST:
            assert isinstance(code, str)
            assert len(code) == 4  # HK stock codes are 4 digits
    
    def test_boolean_config_values(self):
        """Test that boolean config values are actually booleans."""
        import config
        assert isinstance(config.USE_ENSEMBLE, bool)
        assert isinstance(config.USE_STACKING, bool)
        assert isinstance(config.USE_BLENDING, bool)
        assert isinstance(config.USE_CATBOOST, bool)
        assert isinstance(config.USE_SMOTE, bool)
    
    def test_supabase_url_format(self):
        """Test that SUPABASE_URL has correct format."""
        import config
        assert config.SUPABASE_URL.startswith('https://')
        assert config.SUPABASE_URL.endswith('.supabase.co')
    
    def test_supabase_key_not_empty(self):
        """Test that SUPABASE_KEY is not empty."""
        import config
        assert config.SUPABASE_KEY is not None
        assert len(config.SUPABASE_KEY) > 0
    
    def test_timezone_config(self):
        """Test that timezone is configured."""
        import config
        assert hasattr(config, 'HK_TZ')
        assert str(config.HK_TZ) == 'Asia/Hong_Kong'


class TestConfigWithMock:
    """Test config with mocked environment variables."""
    
    @patch.dict(os.environ, {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_KEY': 'test-key',
        'STOCK_LIST': '0700,9988',
        'USE_ENSEMBLE': 'False',
        'USE_STACKING': 'True',
        'USE_BLENDING': 'False',
        'USE_CATBOOST': 'True',
        'USE_SMOTE': 'False'
    })
    def test_custom_config_values(self):
        """Test that custom config values are loaded correctly."""
        # Need to reload config module to pick up new env vars
        import importlib
        import config
        importlib.reload(config)
        
        assert config.SUPABASE_URL == 'https://test.supabase.co'
        assert config.SUPABASE_KEY == 'test-key'
        assert config.STOCK_LIST == ['0700', '9988']
        assert config.USE_ENSEMBLE is False
        assert config.USE_STACKING is True
        assert config.USE_BLENDING is False
        assert config.USE_CATBOOST is True
        assert config.USE_SMOTE is False
    
    @patch.dict(os.environ, {
        'SUPABASE_URL': '',
        'SUPABASE_KEY': '',
        'STOCK_LIST': ''
    })
    def test_missing_config_raises_error(self):
        """Test that missing required config raises ValueError."""
        import importlib
        import config
        with pytest.raises(ValueError):
            importlib.reload(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
