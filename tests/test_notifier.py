"""
Tests for notifier.py - Telegram notification module.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNotifier:
    """Tests for Telegram notification functions."""

    def test_is_enabled_false_when_no_token(self):
        """is_enabled returns False when TELEGRAM_BOT_TOKEN is not set."""
        from src import notifier
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', ''):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', ''):
                assert notifier.is_enabled() is False

    def test_is_enabled_true_when_both_set(self):
        """is_enabled returns True when both env vars are set."""
        from src import notifier
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', 'fake_token'):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', 'fake_chat'):
                assert notifier.is_enabled() is True

    def test_send_returns_false_when_disabled(self):
        """_send returns False when Telegram is not configured."""
        from src import notifier
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', ''):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', ''):
                result = notifier._send('test message')
                assert result is False

    def test_send_calls_telegram_api(self):
        """_send calls Telegram API when configured."""
        from src import notifier
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', 'fake_token'):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', 'fake_chat'):
                with patch('requests.post', return_value=mock_resp) as mock_post:
                    result = notifier._send('test message')
                    assert result is True
                    mock_post.assert_called_once()

    def test_notify_failure_returns_false_when_disabled(self):
        """notify_failure returns False when not configured."""
        from src import notifier
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', ''):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', ''):
                result = notifier.notify_failure('training', RuntimeError('test'))
                assert result is False

    def test_notify_success_returns_false_when_disabled(self):
        """notify_success returns False when not configured."""
        from src import notifier
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', ''):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', ''):
                result = notifier.notify_success('training', 'completed')
                assert result is False

    def test_message_truncation(self):
        """Messages exceeding _MAX_LEN are truncated."""
        from src import notifier
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        long_msg = 'x' * 5000
        with patch.object(notifier, 'TELEGRAM_BOT_TOKEN', 'fake_token'):
            with patch.object(notifier, 'TELEGRAM_CHAT_ID', 'fake_chat'):
                with patch('requests.post', return_value=mock_resp) as mock_post:
                    notifier._send(long_msg)
                    # requests.post(url, json={...}) - json is a keyword arg
                    call_kwargs = mock_post.call_args[1]
                    sent_text = call_kwargs['json']['text']
                    assert len(sent_text) < 5000
                    assert 'truncated' in sent_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
