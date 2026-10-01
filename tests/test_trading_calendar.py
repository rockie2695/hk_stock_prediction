"""Tests for the HK trading calendar / 港股交易日曆測試."""
import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import src.trading_calendar as tc


# A real HK public holiday: National Day 2026-10-01 (Thursday).
NATIONAL_DAY_2026 = date(2026, 10, 1)
# A normal trading day near it.
TRADING_DAY_2026 = date(2026, 9, 30)


class TestShouldRunToday:
    """should_run_today gates the whole daily pipeline / 判斷每日作業是否執行。"""

    def test_weekend_is_skipped(self):
        # 2026-10-03 is a Saturday, 2026-10-04 a Sunday
        for d, name in ((date(2026, 10, 3), "Saturday"), (date(2026, 10, 4), "Sunday")):
            with patch.object(tc, "load_holiday_set", return_value=(set(), True)):
                should_run, reason = tc.should_run_today(d)
            assert should_run is False
            assert name in reason

    def test_public_holiday_is_skipped(self):
        """Regression: 2026-10-01 (HK National Day) must NOT trigger the daily run."""
        with patch.object(tc, "load_holiday_set",
                          return_value=({NATIONAL_DAY_2026.isoformat()}, True)):
            should_run, reason = tc.should_run_today(NATIONAL_DAY_2026)

        assert should_run is False
        assert "holiday" in reason.lower()
        assert NATIONAL_DAY_2026.isoformat() in reason

    def test_trading_day_runs(self):
        with patch.object(tc, "load_holiday_set",
                          return_value=({NATIONAL_DAY_2026.isoformat()}, True)):
            should_run, reason = tc.should_run_today(TRADING_DAY_2026)

        assert should_run is True
        assert "trading day" in reason.lower()

    def test_holiday_lookup_never_skips_work_when_unreliable(self):
        """Fail-open: an undeterminable holiday list must RUN, not skip.

        Rationale: a false 'holiday' verdict silently drops a real trading day's
        prediction, which is much worse than wasting compute.
        回退設計：假期清單無法確定時必須「執行」而非「跳過」。
        """
        # Unreliable list that happens to contain the date - must still run
        with patch.object(tc, "load_holiday_set",
                          return_value=({NATIONAL_DAY_2026.isoformat()}, False)):
            should_run, reason = tc.should_run_today(NATIONAL_DAY_2026)

        assert should_run is True
        assert "UNRELIABLE" in reason
        assert "fail-open" in reason

    def test_reasons_are_loggable(self):
        """Every verdict must carry a non-empty reason for run_log.txt.
        每個判斷結果都必須帶有原因供日誌記錄。
        """
        with patch.object(tc, "load_holiday_set", return_value=(set(), True)):
            for d in (NATIONAL_DAY_2026, TRADING_DAY_2026, date(2026, 10, 3)):
                _, reason = tc.should_run_today(d)
                assert reason and d.isoformat() in reason


class TestLoadHolidaySet:
    """Cache / API / fallback resolution / 快取、API、回退解析。"""

    def test_fresh_cache_is_reliable(self, tmp_path):
        cache = tmp_path / "hk_holidays.json"
        cache.write_text(json.dumps(["2026-10-01", "2026-12-25"]))
        with patch.object(tc, "HK_HOLIDAY_CACHE_FILE", str(cache)), \
             patch.object(tc, "_fetch_online") as m_fetch:
            holidays, reliable = tc.load_holiday_set()

        assert reliable is True
        assert "2026-10-01" in holidays
        m_fetch.assert_not_called()  # fresh cache must not hit the network

    def test_stale_cache_triggers_refetch(self, tmp_path):
        cache = tmp_path / "hk_holidays.json"
        cache.write_text(json.dumps(["2026-10-01"]))
        old = (tc.now_hk().timestamp() - 40 * 24 * 3600)
        os.utime(cache, (old, old))
        with patch.object(tc, "HK_HOLIDAY_CACHE_FILE", str(cache)), \
             patch.object(tc, "_fetch_online", return_value={"2026-10-01", "2027-01-01"}):
            holidays, reliable = tc.load_holiday_set()

        assert reliable is True
        assert "2027-01-01" in holidays

    def test_api_result_is_cached(self, tmp_path):
        cache = tmp_path / "hk_holidays.json"
        with patch.object(tc, "HK_HOLIDAY_CACHE_FILE", str(cache)), \
             patch.object(tc, "_read_cache", return_value=None), \
             patch.object(tc, "_fetch_online", return_value={"2026-10-01"}):
            holidays, reliable = tc.load_holiday_set()

        assert reliable is True
        assert json.loads(cache.read_text()) == ["2026-10-01"]

    def test_total_failure_is_unreliable_fallback(self, tmp_path):
        """No cache + no API -> incomplete list, flagged unreliable.
        無快取、無 API -> 回退清單不完整，標記為不可靠。
        """
        with patch.object(tc, "HK_HOLIDAY_CACHE_FILE", str(tmp_path / "none.json")), \
             patch.object(tc, "_fetch_online", return_value=None):
            holidays, reliable = tc.load_holiday_set()

        assert reliable is False
        assert f"{tc.now_hk().year}-10-01" in holidays  # fixed-date holidays only

    def test_fallback_excludes_lunar_holidays(self, tmp_path):
        """Documented limitation: the fallback cannot contain lunar-calendar dates.
        已記錄的限制：回退清單不可能包含農曆假期。
        """
        fallback = tc._fixed_date_fallback(2026)
        # Only Gregorian-fixed HK holidays
        assert fallback == {
            "2026-01-01", "2026-05-01", "2026-07-01",
            "2026-10-01", "2026-12-25", "2026-12-26",
        }


class TestTradingDayHelpers:
    def test_is_trading_day(self):
        with patch.object(tc, "load_holiday_set",
                          return_value=({NATIONAL_DAY_2026.isoformat()}, True)):
            assert tc.is_trading_day(TRADING_DAY_2026) is True
            assert tc.is_trading_day(NATIONAL_DAY_2026) is False
            assert tc.is_trading_day(date(2026, 10, 3)) is False  # Saturday

    def test_is_market_holiday(self):
        with patch.object(tc, "load_holiday_set",
                          return_value=({NATIONAL_DAY_2026.isoformat()}, True)):
            assert tc.is_market_holiday(NATIONAL_DAY_2026) is True
            assert tc.is_market_holiday(TRADING_DAY_2026) is False

    def test_fetch_hk_holidays_returns_mutable_copy(self):
        a = tc.fetch_hk_holidays()
        a.add("MUTATED")
        assert "MUTATED" not in tc.fetch_hk_holidays()


class TestNextTradingDay:
    def test_skips_holiday_and_weekend(self):
        # 2026-09-30 Wed -> 2026-10-01 holiday, 10-02 Fri should be next
        holidays = {NATIONAL_DAY_2026.isoformat()}
        with patch.object(tc, "fetch_hk_holidays", return_value=set(holidays)):
            assert tc.next_trading_day(TRADING_DAY_2026) == date(2026, 10, 2)

    def test_inclusive_returns_same_day_if_trading(self):
        with patch.object(tc, "fetch_hk_holidays", return_value=set()):
            assert tc.next_trading_day(TRADING_DAY_2026, inclusive=True) == TRADING_DAY_2026

    def test_skips_year_boundary(self):
        """Year-end holidays must not break the search / 年末假期不得使搜尋失敗。"""
        with patch.object(tc, "fetch_hk_holidays", return_value={"2026-12-25", "2027-01-01"}):
            nxt = tc.next_trading_day(date(2026, 12, 24))
        assert nxt > date(2026, 12, 24)
        assert nxt.isoformat() not in {"2026-12-25", "2027-01-01"}


class TestRealWorldToday:
    """Sanity check against the project's actual holiday cache.
    以專案實際的假期快取做合理性驗證。
    """

    def test_national_day_2026_detected_via_real_data_path(self):
        """Exercise the real cache-reading path with a temp cache.
        以暫存快取走真實的快取讀取路徑。
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "hk_holidays.json")
            with open(cache, 'w') as f:
                json.dump(["2026-09-26", "2026-10-01", "2026-10-19"], f)
            with patch.object(tc, "HK_HOLIDAY_CACHE_FILE", cache):
                should_run, reason = tc.should_run_today(NATIONAL_DAY_2026)

        assert should_run is False
        assert "2026-10-01" in reason