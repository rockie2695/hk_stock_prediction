"""
HK trading calendar - 香港交易日曆
Single source of truth for "is the HK stock market open today?"
單一來源判斷「今日港股市場是否開市」。

Used by src/train_model.py, src/predict_upload.py and run_daily.bat so the batch
script and manual runs share one decision instead of duplicating the logic.
由 train_model.py、predict_upload.py 與 run_daily.bat 共用，避免重複判斷邏輯。

Holiday data comes from the official 1823.gov.hk iCal feed and is cached locally
for 30 days. The cache is authoritative for gate decisions, but a failed lookup is
treated as UNRELIABLE and never used to skip work - see should_run_today().
假期資料來自 1823.gov.hk 官方 iCal，於本機快取 30 天。
查詢失敗時視為「不可靠」，絕不因此跳過作業（fail-open）。
"""
import json
import os
from datetime import date, datetime, timedelta
from typing import Optional, Set, Tuple

import pytz

from src.logger import setup_logger

logger = setup_logger('trading_calendar')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

HK_HOLIDAY_API = "https://www.1823.gov.hk/common/ical/en.json"
HK_HOLIDAY_CACHE_FILE = os.path.join(CACHE_DIR, 'hk_holidays.json')
HK_HOLIDAY_CACHE_TTL = 30 * 24 * 3600  # 30 days / 30 天

HK_TZ = pytz.timezone('Asia/Hong_Kong')


def now_hk() -> datetime:
    """Current time in Hong Kong / 香港時間。"""
    return datetime.now(HK_TZ)


def _read_cache() -> Optional[Set[str]]:
    """Read the holiday cache if it exists and is fresh. None if missing/stale/unreadable.
    讀取快取假期（存在且未過期時）。不存在／過期／讀取失敗則回傳 None。
    """
    if not os.path.exists(HK_HOLIDAY_CACHE_FILE):
        return None
    try:
        age = now_hk().timestamp() - os.path.getmtime(HK_HOLIDAY_CACHE_FILE)
    except OSError:
        return None
    if age >= HK_HOLIDAY_CACHE_TTL:
        logger.info(f"[holiday-api] Cache stale ({age / 86400:.1f} days old), refetching")
        return None
    try:
        with open(HK_HOLIDAY_CACHE_FILE, 'r') as f:
            cached = json.load(f)
        logger.info(f"[holiday-api] Loaded {len(cached)} holidays from cache")
        return set(cached)
    except Exception as e:
        logger.warning(f"[holiday-api] Cache read failed: {e}")
        return None


def _fetch_online() -> Optional[Set[str]]:
    """Fetch holidays from the 1823.gov.hk iCal feed. None on any failure.
    從 1823.gov.hk 抓取假期。任何失敗皆回傳 None。
    """
    import urllib.request

    try:
        req = urllib.request.Request(HK_HOLIDAY_API, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8-sig'))

        vevents = data.get('vcalendar', [{}])[0].get('vevent', [])
        holidays = set()
        for event in vevents:
            dtstart = event.get('dtstart', [''])[0] if isinstance(event.get('dtstart'), list) else event.get('dtstart', '')
            if dtstart and len(dtstart) == 8:
                holidays.add(f"{dtstart[:4]}-{dtstart[4:6]}-{dtstart[6:]}")

        logger.info(f"[holiday-api] Fetched {len(holidays)} holidays from 1823.gov.hk")
        return holidays
    except Exception as e:
        logger.warning(f"[holiday-api] Online fetch failed: {e}")
        return None


def _fixed_date_fallback(year: int) -> Set[str]:
    """Best-effort holidays for a year, used only when both cache and API fail.
    僅在快取與 API 都失敗時使用的最佳努力假期清單。

    These are the Gregorian-fixed HK holidays. Lunar-calendar holidays (Lunar New
    Year, Ching Ming, Buddha's Birthday, Tuen Ng, Mid-Autumn, Chung Yeung) cannot be
    hardcoded, so this list is INCOMPLETE by construction - which is exactly why
    callers treat it as unreliable.
    此清單僅含西曆固定假期，農曆假期無法硬編碼，因此必然不完整，
    呼叫端必須視為「不可靠」。
    """
    return {
        f"{year}-01-01",   # New Year's Day / 元旦
        f"{year}-05-01",   # Labour Day / 勞動節
        f"{year}-07-01",   # HKSAR Establishment Day / 香港特別行政區成立紀念日
        f"{year}-10-01",   # National Day / 國慶日
        f"{year}-12-25",   # Christmas / 聖誕節
        f"{year}-12-26",   # The weekday after Christmas / 聖誕節後第一個周日
    }


def load_holiday_set() -> Tuple[Set[str], bool]:
    """Return (holidays, reliable).

    reliable=True  -> cache or official API; safe to use for skip decisions
                      (快取或官方 API，可安全用於跳過判斷)
    reliable=False -> partial hardcoded fallback; must NOT be used to skip work
                      (不完整硬編碼回退，不得用於跳過作業)

    Resolution order: fresh cache -> 1823.gov.hk API -> fixed-date fallback.
    解析順序：有效快取 -> 1823.gov.hk API -> 固定日期回退。
    """
    cached = _read_cache()
    if cached is not None:
        return cached, True

    fetched = _fetch_online()
    if fetched is not None:
        try:
            with open(HK_HOLIDAY_CACHE_FILE, 'w') as f:
                json.dump(sorted(fetched), f, indent=2)
        except Exception as e:
            logger.warning(f"[holiday-api] Cache write failed: {e}")
        return fetched, True

    year = now_hk().year
    logger.warning("[holiday-api] Using incomplete fixed-date fallback; "
                   "treat holiday list as UNRELIABLE")
    return _fixed_date_fallback(year), False


def fetch_hk_holidays() -> Set[str]:
    """Holiday date strings ('YYYY-MM-DD'). Returns a fresh mutable copy.
    假期日期字串集合。回傳可變的新副本。
    """
    holidays, _ = load_holiday_set()
    return set(holidays)


def is_market_holiday(d: Optional[date] = None) -> bool:
    """True if d is a HK public holiday (weekends excluded from this check).
    d 是否為香港公眾假期（此檢查不包含週末）。
    """
    target = d or now_hk().date()
    return target.isoformat() in fetch_hk_holidays()


def is_trading_day(d: Optional[date] = None) -> bool:
    """True if the HK stock market is expected to trade on d (Mon-Fri, not a holiday).
    d 是否為港股交易日（週一至週五且非公眾假期）。
    """
    target = d or now_hk().date()
    if target.weekday() >= 5:
        return False
    return target.isoformat() not in fetch_hk_holidays()


def should_run_today(d: Optional[date] = None) -> Tuple[bool, str]:
    """Decide whether the daily pipeline should run. Returns (should_run, reason).
    判斷每日作業是否應執行。回傳 (是否執行, 原因)。

    Fail-open by design: if the holiday list cannot be determined we RUN anyway.
    A false 'holiday' verdict would silently drop a real trading day's prediction,
    which is far worse than the wasted compute of a false 'trading day' verdict.
    採 fail-open 設計：若無法確定假期清單則照常執行。
    「誤判為假期」會靜默漏掉真實交易日的預測，比「誤判為交易日」浪費算力嚴重得多。

    Note: typhoon / rainstorm closures are not in any published calendar and cannot
    be predicted ahead of time; this check covers scheduled closures only.
    注意：八號風球／黑雨等臨時停市不在任何預Published日曆中，無法預先判斷；
    此檢查僅涵蓋已排定的休市日。
    """
    target = d or now_hk().date()
    iso = target.isoformat()

    if target.weekday() >= 5:
        return False, f"{iso} is a {target.strftime('%A')} - HK market closed"

    holidays, reliable = load_holiday_set()
    if not reliable:
        return True, (f"{iso} - holiday list UNRELIABLE (API unreachable, no fresh "
                      f"cache); running anyway (fail-open)")

    if iso in holidays:
        return False, f"{iso} is a HK public holiday - HK market closed"

    return True, f"{iso} is a HK trading day"


def next_trading_day(d: Optional[date] = None, inclusive: bool = False) -> date:
    """Next HK trading day on or after d. inclusive=True includes d itself.
    d 當日或之後的下一個港股交易日。inclusive=True 時包含 d 本身。
    """
    target = d or now_hk().date()
    if not inclusive:
        target += timedelta(days=1)

    holidays = fetch_hk_holidays()
    year = target.year
    holidays.add(f"{year - 1}-12-31")
    holidays.add(f"{year + 1}-01-01")

    for _ in range(400):
        if target.weekday() < 5 and target.isoformat() not in holidays:
            return target
        target += timedelta(days=1)

    # Practically unreachable; a 400-day window always contains trading days.
    # 實際上不可達：400 天內必然存在交易日。
    raise RuntimeError(f"Could not find a trading day within 400 days of {d}")