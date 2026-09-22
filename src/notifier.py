"""
Failure Notifications - 失敗通知
Sends Telegram notifications when training or prediction fails.
訓練或預測失敗時透過 Telegram 發送通知。

Optional — disabled when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set.
可選 — 當未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 時停用。
"""
import os
import traceback
import requests
from src.logger import setup_logger

logger = setup_logger('notifier')

# Optional env vars / 可選環境變數
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Cap message length to avoid Telegram 4096-char limit / 限制訊息長度避免 Telegram 字數上限
_MAX_LEN = 3800


def is_enabled() -> bool:
    """Whether notifications are configured / 是否已設定通知"""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _send(text: str) -> bool:
    """Send a raw message to Telegram / 發送原始訊息至 Telegram"""
    if not is_enabled():
        logger.info("  Telegram not configured — skipping notification / 未設定 Telegram，跳過通知")
        return False
    if len(text) > _MAX_LEN:
        text = text[:_MAX_LEN] + "\n...(truncated)"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
        }, timeout=15)
        resp.raise_for_status()
        logger.info("  Notification sent / 通知已發送")
        return True
    except Exception as e:
        logger.warning(f"  Failed to send Telegram notification: {e} / 發送 Telegram 通知失敗")
        return False


def notify_failure(component: str, error: Exception) -> bool:
    """Notify about a failure / 通知失敗事件

    Args / 參數:
        component: 'training' or 'prediction' / '訓練' 或 '預測'
        error: The exception raised / 拋出的例外

    Returns / 返回:
        True if sent, False otherwise / 已發送為 True，否則 False
    """
    tb = traceback.format_exc()[-1200:]
    msg = (
        f"<b>⚠️ {component} 失敗 / {component} FAILED</b>\n"
        f"<code>{type(error).__name__}: {str(error)[:500]}</code>\n"
        f"<pre>{tb}</pre>"
    )
    return _send(msg)


def notify_success(component: str, summary: str) -> bool:
    """Notify about successful completion / 通知完成事件"""
    msg = f"<b>✅ {component} 完成 / {component} DONE</b>\n{summary}"
    return _send(msg)