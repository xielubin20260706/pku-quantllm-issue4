#!/usr/bin/env python3
"""Daily long-only A-share backtester for the course Feather matrices."""

from __future__ import annotations

import argparse
import csv
import io
import inspect
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


MATRICES = (
    "open",
    "high",
    "low",
    "close",
    "amt",
    "num_trades",
    "cumadj",
)


def _normalize_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raw_index = pd.Series(frame.index)
        if pd.api.types.is_numeric_dtype(raw_index):
            text_index = raw_index.astype("Int64").astype(str)
            if text_index.str.fullmatch(r"\d{8}").all():
                parsed_index = pd.to_datetime(text_index, format="%Y%m%d", errors="coerce")
            else:
                parsed_index = pd.to_datetime(frame.index, errors="coerce")
        else:
            parsed_index = pd.to_datetime(frame.index, errors="coerce")
        if (len(parsed_index) and not pd.isna(parsed_index).any()
                and parsed_index.min().year >= 1990 and parsed_index.max().year <= 2100):
            frame.index = pd.DatetimeIndex(parsed_index, name="date")
        date_col = next(
            (c for c in frame.columns if str(c).strip().lower() in
             {"date", "datetime", "trade_date", "trading_date", "日期"}),
            None,
        )
        if not isinstance(frame.index, pd.DatetimeIndex) and date_col is None and len(frame.columns):
            first = frame.columns[0]
            raw_dates = frame[first]
            if pd.api.types.is_numeric_dtype(raw_dates):
                text_dates = raw_dates.astype("Int64").astype(str)
                if text_dates.str.fullmatch(r"\d{8}").all():
                    parsed = pd.to_datetime(text_dates, format="%Y%m%d", errors="coerce")
                else:
                    parsed = pd.to_datetime(raw_dates, errors="coerce")
            else:
                parsed = pd.to_datetime(raw_dates, errors="coerce")
            if (parsed.notna().all() and len(parsed)
                    and parsed.min().year >= 1990 and parsed.max().year <= 2100):
                date_col = first
        if not isinstance(frame.index, pd.DatetimeIndex) and date_col is not None:
            raw_dates = frame.pop(date_col)
            if pd.api.types.is_numeric_dtype(raw_dates):
                text_dates = raw_dates.astype("Int64").astype(str)
                if text_dates.str.fullmatch(r"\d{8}").all():
                    parsed = pd.to_datetime(text_dates, format="%Y%m%d", errors="coerce")
                else:
                    parsed = pd.to_datetime(raw_dates, errors="coerce")
            else:
                parsed = pd.to_datetime(raw_dates, errors="coerce")
            frame.index = pd.DatetimeIndex(parsed, name="date")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name}: 找不到交易日期索引或日期列")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
    frame.columns = pd.Index([str(c) for c in frame.columns], name="symbol")
    if frame.index.has_duplicates:
        raise ValueError(f"{name}: 日期存在重复值")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name}: 证券代码存在重复值")
    frame = frame.sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    values: dict[str, np.ndarray]
    simulation_start: int = 0
    status_source: str = "missing"

    @classmethod
    def load(cls, source: str | Path, start: str | None = None, end: str | None = None,
             warmup: int = 0) -> "MarketData":
        path = Path(source)
        source_dir = path if path.is_dir() else path.parent
        frames: dict[str, pd.DataFrame] = {}
        if path.is_dir():
            if all((path / f"{name}.feather").exists() for name in MATRICES):
                for name in MATRICES:
                    frames[name] = _normalize_frame(pd.read_feather(path / f"{name}.feather"), name)
                for extra_path in sorted(path.glob("*.feather")):
                    if extra_path.stem in MATRICES or extra_path.name.startswith("._"):
                        continue
                    name = extra_path.stem
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        raise ValueError(f"扩展矩阵文件名必须是 Python 字段名：{extra_path.name}")
                    frames[name] = _normalize_frame(pd.read_feather(extra_path), name)
            else:
                archives = [path / "feathers.zip"] if (path / "feathers.zip").exists() else sorted(path.glob("*.zip"))
                if not archives:
                    raise FileNotFoundError(f"{path} 下未找到 7 个 Feather 矩阵或 feathers.zip")
                path = archives[0]
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names_by_base = {Path(item).name: item for item in archive.namelist() if not item.endswith("/")}
                for name in MATRICES:
                    filename = f"{name}.feather"
                    if filename not in names_by_base:
                        raise FileNotFoundError(f"数据压缩包中缺少 {filename}")
                    raw = archive.read(names_by_base[filename])
                    frames[name] = _normalize_frame(pd.read_feather(io.BytesIO(raw)), name)
                for item in archive.namelist():
                    item_path = Path(item)
                    if (item.endswith("/") or item_path.suffix.lower() != ".feather"
                            or item_path.stem in MATRICES or item_path.name.startswith("._")
                            or "__MACOSX" in item_path.parts):
                        continue
                    name = item_path.stem
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        raise ValueError(f"扩展矩阵文件名必须是 Python 字段名：{item_path.name}")
                    if name in frames:
                        raise ValueError(f"数据包内扩展矩阵字段重名：{name}")
                    raw = archive.read(item)
                    frames[name] = _normalize_frame(pd.read_feather(io.BytesIO(raw)), name)
        elif path.is_file() and path.suffix.lower() == ".feather":
            raise ValueError("请传入包含 7 个矩阵的目录或 feathers.zip，而不是单个 Feather 文件")
        elif not frames:
            raise FileNotFoundError(f"数据路径不存在或格式不支持：{path}")

        base = frames["close"]
        dates = base.index
        symbols = base.columns
        for name, frame in frames.items():
            if not frame.index.equals(dates):
                raise ValueError(f"{name}: 日期与 close.feather 不一致")
            if not frame.columns.equals(symbols):
                raise ValueError(f"{name}: 证券列与 close.feather 不一致")

        status_source = "missing"
        status_path = source_dir / "security_status.csv"
        if status_path.is_file():
            status_frame, limit_frame = _read_security_status(status_path, dates, symbols)
            frames["is_st"] = status_frame
            frames["limit_pct"] = limit_frame
            status_source = status_path.name
        elif "is_st" in frames:
            status_values = frames["is_st"].to_numpy(dtype=np.float64)
            invalid = np.isfinite(status_values) & ~np.isin(status_values, (0.0, 1.0))
            if invalid.any():
                raise ValueError("is_st.feather 只能包含 0（非 ST）、1（ST）或空值")
            status_source = "is_st.feather"
        else:
            frames["is_st"] = pd.DataFrame(np.nan, index=dates, columns=symbols)
            status_source = "missing"
        if "limit_pct" not in frames:
            frames["limit_pct"] = pd.DataFrame(np.nan, index=dates, columns=symbols)
        else:
            limit_values = frames["limit_pct"].to_numpy(dtype=np.float64)
            invalid_limits = np.isfinite(limit_values) & ((limit_values < 0) | (limit_values >= 1))
            if invalid_limits.any():
                raise ValueError("limit_pct.feather 中有效值必须在 [0, 1) 内")
        absolute_start = int(dates.searchsorted(pd.Timestamp(start), side="left")) if start else 0
        absolute_end = int(dates.searchsorted(pd.Timestamp(end), side="right")) if end else len(dates)
        if absolute_start >= absolute_end:
            raise ValueError("筛选范围内没有交易日")
        data_start = max(0, absolute_start - max(0, warmup)) if start else 0
        positions = np.arange(data_start, absolute_end)
        simulation_start = absolute_start - data_start

        trimmed: dict[str, np.ndarray] = {}
        for name, frame in frames.items():
            trimmed[name] = np.asarray(frame.to_numpy(copy=False)[positions], dtype=np.float64)
        return cls(dates[positions], symbols, trimmed, simulation_start, status_source)


def _read_security_status(path: Path, dates: pd.DatetimeIndex,
                          symbols: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read inclusive per-symbol status intervals used for historical ST exclusion and limits."""
    table = pd.read_csv(path, dtype={"symbol": str}, keep_default_na=False)
    required = {"symbol", "start_date", "end_date"}
    status_column = "status" if "status" in table.columns else "is_st"
    if not required.issubset(table.columns) or status_column not in table.columns:
        raise ValueError("security_status.csv 需要 symbol,start_date,end_date,status 列；limit_pct 可选")
    unknown_symbols = sorted(set(table["symbol"].astype(str).str.strip()) - set(map(str, symbols)))
    if unknown_symbols:
        raise ValueError(f"security_status.csv 含矩阵中不存在的证券代码：{', '.join(unknown_symbols[:5])}")

    st_values = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    limit_values = np.full_like(st_values, np.nan)
    symbol_index = {str(symbol): index for index, symbol in enumerate(symbols)}
    occupied = np.zeros_like(st_values, dtype=bool)
    first_date, last_date = dates[0], dates[-1]
    for row_number, row in enumerate(table.to_dict(orient="records"), start=2):
        symbol = str(row["symbol"]).strip()
        if not symbol:
            raise ValueError(f"security_status.csv 第 {row_number} 行缺少 symbol")
        raw_status = str(row[status_column]).strip().upper().replace(" ", "")
        if raw_status in {"1", "1.0", "ST", "*ST", "TRUE", "Y", "YES", "是"}:
            is_st = 1.0
        elif raw_status in {"0", "0.0", "NORMAL", "NON-ST", "NON_ST", "FALSE", "N", "NO", "否", "非ST"}:
            is_st = 0.0
        else:
            raise ValueError(f"security_status.csv 第 {row_number} 行 status 无法识别：{raw_status}")
        start_raw = str(row.get("start_date", "")).strip()
        end_raw = str(row.get("end_date", "")).strip()
        interval_start = pd.Timestamp(start_raw) if start_raw else first_date
        interval_end = pd.Timestamp(end_raw) if end_raw else last_date
        if interval_start > interval_end:
            raise ValueError(f"security_status.csv 第 {row_number} 行 start_date 晚于 end_date")
        lo = int(dates.searchsorted(interval_start, side="left"))
        hi = int(dates.searchsorted(interval_end, side="right"))
        if lo >= hi:
            continue
        column = symbol_index[symbol]
        if occupied[lo:hi, column].any():
            raise ValueError(f"{symbol}: security_status.csv 状态区间重叠")
        occupied[lo:hi, column] = True
        st_values[lo:hi, column] = is_st
        raw_limit = str(row.get("limit_pct", "")).strip()
        if raw_limit:
            pct = float(raw_limit)
            if not np.isfinite(pct) or pct < 0 or pct >= 1:
                raise ValueError(f"security_status.csv 第 {row_number} 行 limit_pct 应在 [0, 1) 内")
            limit_values[lo:hi, column] = pct

    return (pd.DataFrame(st_values, index=dates, columns=symbols),
            pd.DataFrame(limit_values, index=dates, columns=symbols))


@dataclass
class Fees:
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    sell_stamp_duty: float = 0.0005
    slippage_bps: float = 2.0


@dataclass
class EngineConfig:
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    default_limit_pct: float = 0.10
    max_daily_turnover_participation: float | None = None
    liquidate_at_end: bool = False
    fees: Fees = field(default_factory=Fees)


@dataclass
class PortfolioSnapshot:
    """Point-in-time account state passed to strategies that request it."""

    positions: np.ndarray
    average_cost: np.ndarray
    mark_prices: np.ndarray
    cash: float
    market_value: float
    nav: float


def infer_limit_pct(symbol: str, default: float = 0.10,
                    date: pd.Timestamp | None = None, is_st: bool = False) -> float:
    """Infer common board bands; IPO exceptions still need a supplied per-date override."""
    code = "".join(ch for ch in str(symbol).split(".")[0] if ch.isdigit())
    if code.startswith(("300", "301")):
        if date is not None and date < pd.Timestamp("2020-08-24"):
            return default
        return 0.20
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("4", "8", "92")):
        return 0.30
    if is_st:
        return 0.10 if date is not None and date >= pd.Timestamp("2026-07-06") else 0.05
    return default


@dataclass(frozen=True)
class LimitOverride:
    symbol: str
    limit_pct: float
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None

    def applies(self, date: pd.Timestamp) -> bool:
        return ((self.start_date is None or date >= self.start_date)
                and (self.end_date is None or date <= self.end_date))


def load_limit_rules(path: str | Path | None) -> list[LimitOverride]:
    """Read CSV columns symbol,limit_pct and optional inclusive start_date,end_date."""
    rules: list[LimitOverride] = []
    if path is None:
        return rules
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"symbol", "limit_pct"}.issubset(reader.fieldnames):
            raise ValueError("限价规则 CSV 必须包含 symbol,limit_pct 两列")
        for row in reader:
            symbol = str(row["symbol"]).strip()
            pct = float(row["limit_pct"])
            if pct < 0 or pct >= 1:
                raise ValueError(f"{symbol}: limit_pct 应在 [0, 1) 内")
            start_raw = (row.get("start_date") or row.get("start") or "").strip()
            end_raw = (row.get("end_date") or row.get("end") or "").strip()
            start_date = pd.Timestamp(start_raw) if start_raw else None
            end_date = pd.Timestamp(end_raw) if end_raw else None
            if start_date is not None and end_date is not None and start_date > end_date:
                raise ValueError(f"{symbol}: start_date 不能晚于 end_date")
            rules.append(LimitOverride(symbol, pct, start_date, end_date))
    return rules


class Strategy(Protocol):
    def target_weights(self, data: MarketData, day: int,
                       previous_weights: np.ndarray,
                       portfolio: PortfolioSnapshot | None = None) -> np.ndarray | None:
        """Return target weights after the close, or None to keep the prior target."""


class MomentumTopN:
    """Monthly cross-sectional momentum example, with next-session open execution."""

    def __init__(self, lookback: int = 20, top_n: int = 20):
        if lookback < 1 or top_n < 1:
            raise ValueError("lookback 和 top_n 必须为正整数")
        self.lookback = lookback
        self.top_n = top_n
        self.last_month: tuple[int, int] | None = None

    def target_weights(self, data: MarketData, day: int,
                       previous_weights: np.ndarray) -> np.ndarray | None:
        current_month = (data.dates[day].year, data.dates[day].month)
        signal_ready = day >= self.lookback
        rebalance = current_month != self.last_month or (previous_weights.sum() == 0 and signal_ready)
        if not rebalance:
            return None
        self.last_month = current_month
        result = np.zeros(len(data.symbols), dtype=np.float64)
        if not signal_ready:
            return result

        close = data.values["close"]
        factor = data.values["cumadj"]
        adjusted_now = close[day] * factor[day]
        adjusted_then = close[day - self.lookback] * factor[day - self.lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            score = adjusted_now / adjusted_then - 1.0
        liquid = (data.values["amt"][day] > 0) & (data.values["num_trades"][day] > 0)
        status = data.values.get("is_st")
        not_st = np.ones(len(data.symbols), dtype=bool) if status is None else ~(np.isfinite(status[day]) & (status[day] >= 0.5))
        valid = np.isfinite(score) & np.isfinite(adjusted_now) & (adjusted_then > 0) & liquid & not_st
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return result
        count = min(self.top_n, len(candidates))
        best = candidates[np.argpartition(score[candidates], -count)[-count:]]
        best = best[np.argsort(score[best])[::-1]]
        result[best] = 1.0 / len(best)
        return result


class Backtester:
    def __init__(self, data: MarketData, strategy: Strategy, config: EngineConfig,
                 limit_overrides: list[LimitOverride] | None = None):
        self.data = data
        self.strategy = strategy
        self.config = config
        self.limit_overrides = list(limit_overrides or [])
        try:
            parameters = inspect.signature(self.strategy.target_weights).parameters.values()
            self._strategy_accepts_snapshot = (len(list(parameters)) >= 4)
        except (TypeError, ValueError):
            self._strategy_accepts_snapshot = False
        self.symbols = list(map(str, data.symbols))
        self.symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.st_mask = np.isfinite(data.values["is_st"]) & (data.values["is_st"] >= 0.5)
        digits = ["".join(ch for ch in symbol.split(".")[0] if ch.isdigit()) for symbol in self.symbols]
        self.chinext_mask = np.array([code.startswith(("300", "301")) for code in digits], dtype=bool)
        self.special_board_mask = np.array([
            code.startswith(("300", "301", "688", "689", "4", "8", "92")) for code in digits
        ], dtype=bool)
        self.limit_pcts = np.array([infer_limit_pct(symbol, config.default_limit_pct) for symbol in self.symbols], dtype=float)
        for rule in self.limit_overrides:
            if rule.symbol not in self.symbol_index:
                raise ValueError(f"限价规则中的证券代码不在数据矩阵中：{rule.symbol}")
        self._validate_config()

    def _strategy_signal(self, day: int, weights: np.ndarray, quantities: np.ndarray,
                         average_cost: np.ndarray, cash: float, marks: np.ndarray) -> np.ndarray | None:
        if not self._strategy_accepts_snapshot:
            return self.strategy.target_weights(self.data, day, weights)
        valid_marks = np.isfinite(marks) & (marks > 0)
        market_value = float(np.sum(quantities[valid_marks] * marks[valid_marks]))
        snapshot = PortfolioSnapshot(
            positions=quantities.copy(),
            average_cost=average_cost.copy(),
            mark_prices=marks.copy(),
            cash=float(cash),
            market_value=market_value,
            nav=float(cash + market_value),
        )
        return self.strategy.target_weights(self.data, day, weights, snapshot)

    def _validate_config(self) -> None:
        if self.config.initial_cash <= 0 or self.config.lot_size < 1:
            raise ValueError("initial_cash 和 lot_size 必须为正数")
        if not 0 <= self.config.default_limit_pct < 1:
            raise ValueError("default_limit_pct 应在 [0, 1) 内")
        participation = self.config.max_daily_turnover_participation
        if participation is not None and not 0 < participation <= 1:
            raise ValueError("max_daily_turnover_participation 应在 (0, 1] 内")
        f = self.config.fees
        if min(f.commission_rate, f.minimum_commission, f.sell_stamp_duty, f.slippage_bps) < 0:
            raise ValueError("费率、最低佣金和滑点不能为负")

    @staticmethod
    def _fee(notional: float, fees: Fees, side: str) -> tuple[float, float]:
        commission = max(fees.minimum_commission, notional * fees.commission_rate)
        tax = notional * fees.sell_stamp_duty if side == "sell" else 0.0
        return commission, tax

    @staticmethod
    def _round_cent(price: np.ndarray) -> np.ndarray:
        return np.floor(price * 100.0 + 0.5 + 1e-9) / 100.0

    def _limits(self, day: int) -> tuple[np.ndarray, np.ndarray]:
        values = self.data.values
        prev_close = values["close"][day - 1]
        prev_factor = values["cumadj"][day - 1]
        today_factor = values["cumadj"][day]
        ratio = np.ones_like(prev_close)
        valid_factor = np.isfinite(prev_factor) & np.isfinite(today_factor) & (prev_factor > 0) & (today_factor > 0)
        ratio[valid_factor] = prev_factor[valid_factor] / today_factor[valid_factor]
        reference = prev_close * ratio
        date = self.data.dates[day]
        limit_pcts = self.limit_pcts.copy()
        if date < pd.Timestamp("2020-08-24"):
            limit_pcts[self.chinext_mask] = self.config.default_limit_pct
        main_board_st = self.st_mask[day] & ~self.special_board_mask
        limit_pcts[main_board_st] = 0.10 if date >= pd.Timestamp("2026-07-06") else 0.05
        explicit_limits = values["limit_pct"][day]
        has_explicit_limit = np.isfinite(explicit_limits) & (explicit_limits >= 0) & (explicit_limits < 1)
        limit_pcts[has_explicit_limit] = explicit_limits[has_explicit_limit]
        for rule in self.limit_overrides:
            if rule.applies(date):
                limit_pcts[self.symbol_index[rule.symbol]] = rule.limit_pct
        up = self._round_cent(reference * (1.0 + limit_pcts))
        down = self._round_cent(reference * (1.0 - limit_pcts))
        no_reference = ~np.isfinite(reference) | (reference <= 0) | (limit_pcts <= 0)
        up[no_reference] = np.inf
        down[no_reference] = -np.inf
        return up, down

    def _validated_weights(self, signal: np.ndarray, day: int) -> np.ndarray:
        weights = np.asarray(signal, dtype=np.float64)
        if (weights.shape != (len(self.symbols),) or not np.isfinite(weights).all()
                or (weights < -1e-10).any() or weights.sum() > 1.000001):
            raise ValueError("策略权重必须是一维有限非负数组，且总和不超过 1")
        weights = np.maximum(weights, 0.0)
        # Exclude securities identified as ST on the signal date. Unknown status is reported separately.
        weights[self.st_mask[day]] = 0.0
        return weights

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        data = self.data
        values = data.values
        n_days, n_symbols = values["close"].shape
        simulation_start = data.simulation_start
        quantities = np.zeros(n_symbols, dtype=np.float64)
        average_cost = np.zeros(n_symbols, dtype=np.float64)
        cash = float(self.config.initial_cash)
        weights = np.zeros(n_symbols, dtype=np.float64)
        pending_rebalance = False
        last_close = np.full(n_symbols, np.nan, dtype=np.float64)
        last_factor = np.full(n_symbols, np.nan, dtype=np.float64)
        nav_rows: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []
        gross_turnover: list[float] = []

        first_close = values["close"][0]
        last_close[np.isfinite(first_close) & (first_close > 0)] = first_close[np.isfinite(first_close) & (first_close > 0)]
        first_factor = values["cumadj"][0]
        last_factor[np.isfinite(first_factor) & (first_factor > 0)] = first_factor[np.isfinite(first_factor) & (first_factor > 0)]
        if simulation_start == 0:
            nav_rows.append({"date": data.dates[0], "cash": cash, "market_value": 0.0, "nav": cash,
                             "limit_up_count": 0, "limit_down_count": 0,
                             "st_excluded_count": int(self.st_mask[0].sum()),
                             "st_status_coverage": self._status_coverage(0)})
        if simulation_start == 0:
            signal = self._strategy_signal(0, weights, quantities, average_cost, cash, last_close)
            if signal is not None:
                weights = self._validated_weights(signal, 0)
                pending_rebalance = True
        else:
            for warm_day in range(simulation_start):
                signal = self._strategy_signal(warm_day, weights, quantities, average_cost, cash, last_close)
                if signal is not None:
                    weights = self._validated_weights(signal, warm_day)
                    pending_rebalance = True
        for day in range(max(1, simulation_start), n_days):
            date = data.dates[day]
            newly_excluded = self.st_mask[day] & (quantities > 1e-8)
            if self.st_mask[day].any():
                weights[self.st_mask[day]] = 0.0
            if newly_excluded.any():
                pending_rebalance = True
            open_px = values["open"][day]
            high = values["high"][day]
            low = values["low"][day]
            close = values["close"][day]
            amount = values["amt"][day]
            num_trades = values["num_trades"][day]
            factor = values["cumadj"][day]
            prev_factor = last_factor.copy()

            # cumadj changes represent split/dividend reinvestment; apply them to overnight shares.
            valid_ratio = np.isfinite(factor) & (factor > 0) & np.isfinite(prev_factor) & (prev_factor > 0)
            factor_ratio = np.ones(n_symbols, dtype=np.float64)
            factor_ratio[valid_ratio] = factor[valid_ratio] / prev_factor[valid_ratio]
            quantities *= factor_ratio
            adjusted_cost = valid_ratio & (average_cost > 0)
            average_cost[adjusted_cost] /= factor_ratio[adjusted_cost]
            sellable = quantities.copy()  # T+1: only shares held before today's buys can be sold.

            marks_open = np.where(np.isfinite(open_px) & (open_px > 0), open_px, last_close)
            open_nav = cash + np.nansum(quantities * marks_open)
            desired_value = weights * open_nav
            with np.errstate(divide="ignore", invalid="ignore"):
                desired_qty = desired_value / marks_open
            desired_qty[~np.isfinite(desired_qty)] = 0.0
            up_limit, down_limit = self._limits(day)
            liquid = (np.isfinite(open_px) & (open_px > 0) & np.isfinite(amount) & (amount > 0)
                      & np.isfinite(num_trades) & (num_trades > 0))
            capacity_remaining = np.full(n_symbols, np.inf, dtype=np.float64)
            participation = self.config.max_daily_turnover_participation
            if participation is not None:
                prior_amount = values["amt"][day - 1] if day > 0 else np.zeros(n_symbols)
                capacity_remaining = np.where(
                    np.isfinite(prior_amount) & (prior_amount > 0),
                    prior_amount * participation,
                    0.0,
                )
            locked_up = (open_px >= up_limit - 0.005) & (low >= up_limit - 0.005)
            locked_down = (open_px <= down_limit + 0.005) & (high <= down_limit + 0.005)
            slippage = self.config.fees.slippage_bps / 10_000.0
            day_turnover = 0.0

            # Sell first so proceeds can fund buys. A locked limit-down board cannot be sold.
            sell_candidates = np.flatnonzero((quantities - desired_qty) > 1e-8)
            if not pending_rebalance:
                sell_candidates = np.empty(0, dtype=int)
            retry_rebalance = False
            for j in sell_candidates:
                requested = min(sellable[j], quantities[j] - desired_qty[j])
                if requested <= 1e-8:
                    continue
                exit_all = requested >= quantities[j] - 1e-8
                order_quantity = requested if exit_all else math.floor(requested / self.config.lot_size) * self.config.lot_size
                if order_quantity <= 1e-8:
                    continue
                reason = ""
                if not liquid[j]:
                    reason = "停牌或无成交"
                elif locked_down[j]:
                    reason = "一字跌停，卖出未成交"
                if reason:
                    trade_rows.append(self._trade_row(date, self.symbols[j], "sell", 0, order_quantity, 0, 0, 0, 0, "blocked", reason))
                    retry_rebalance = True
                    continue
                price = open_px[j] * (1.0 - slippage)
                if np.isfinite(down_limit[j]):
                    price = max(price, down_limit[j])
                capacity_quantity = capacity_remaining[j] / price if price > 0 else 0.0
                if capacity_quantity + 1e-8 >= order_quantity:
                    qty = order_quantity
                else:
                    qty = math.floor(capacity_quantity / self.config.lot_size) * self.config.lot_size
                if qty <= 1e-8:
                    trade_rows.append(self._trade_row(date, self.symbols[j], "sell", 0, order_quantity, 0, 0, 0, 0,
                                                      "blocked", "单股成交容量限制"))
                    retry_rebalance = True
                    continue
                notional = qty * price
                commission, tax = self._fee(notional, self.config.fees, "sell")
                cash += notional - commission - tax
                quantities[j] -= qty
                if quantities[j] <= 1e-8:
                    quantities[j] = 0.0
                    average_cost[j] = 0.0
                day_turnover += notional
                capacity_remaining[j] -= notional
                status = "filled" if qty + 1e-8 >= order_quantity else "partial"
                if status == "partial":
                    retry_rebalance = True
                reason = "" if status == "filled" else "单股成交容量限制"
                trade_rows.append(self._trade_row(date, self.symbols[j], "sell", qty, order_quantity, price, notional,
                                                  commission, tax, status, reason))

            # A locked sell keeps its position pending but must not block other buys funded by cash on hand.
            # Buy in descending target weight; use whole lots and retain cash for fees.
            buy_candidates = np.flatnonzero((desired_qty - quantities) > 1e-8)
            if not pending_rebalance:
                buy_candidates = np.empty(0, dtype=int)
            buy_candidates = buy_candidates[np.argsort(weights[buy_candidates])[::-1]]
            for j in buy_candidates:
                requested = desired_qty[j] - quantities[j]
                qty = math.floor(requested / self.config.lot_size) * self.config.lot_size
                if qty < self.config.lot_size:
                    continue
                reason = ""
                if not liquid[j]:
                    reason = "停牌或无成交"
                elif locked_up[j]:
                    reason = "一字涨停，买入未成交"
                if reason:
                    trade_rows.append(self._trade_row(date, self.symbols[j], "buy", 0, qty, 0, 0, 0, 0, "blocked", reason))
                    retry_rebalance = True
                    continue
                price = open_px[j] * (1.0 + slippage)
                if np.isfinite(up_limit[j]):
                    price = min(price, up_limit[j])
                order_quantity = qty
                capacity_limited = False
                capacity_quantity = capacity_remaining[j] / price if price > 0 else 0.0
                if capacity_quantity + 1e-8 < qty:
                    qty = math.floor(capacity_quantity / self.config.lot_size) * self.config.lot_size
                    capacity_limited = True
                    if qty < self.config.lot_size:
                        trade_rows.append(self._trade_row(date, self.symbols[j], "buy", 0, order_quantity, 0, 0, 0, 0,
                                                          "blocked", "单股成交容量限制"))
                        retry_rebalance = True
                        continue
                max_lots = int(max(0.0, cash) // (self.config.lot_size * price))
                qty = min(qty, max_lots * self.config.lot_size)
                while qty >= self.config.lot_size:
                    notional = qty * price
                    commission, tax = self._fee(notional, self.config.fees, "buy")
                    if notional + commission <= cash + 1e-8:
                        break
                    qty -= self.config.lot_size
                if qty < self.config.lot_size:
                    trade_rows.append(self._trade_row(date, self.symbols[j], "buy", 0, order_quantity, 0, 0, 0, 0, "blocked", "可用资金不足"))
                    retry_rebalance = True
                    continue
                notional = qty * price
                commission, tax = self._fee(notional, self.config.fees, "buy")
                cash -= notional + commission
                old_quantity = quantities[j]
                average_cost[j] = ((old_quantity * average_cost[j] + qty * price)
                                   / (old_quantity + qty)) if old_quantity + qty > 0 else 0.0
                quantities[j] += qty
                day_turnover += notional
                capacity_remaining[j] -= notional
                status = "filled" if qty + 1e-8 >= order_quantity else "partial"
                if status == "partial":
                    retry_rebalance = True
                reason = "" if status == "filled" else ("单股成交容量限制" if capacity_limited else "资金或整手限制")
                trade_rows.append(self._trade_row(date, self.symbols[j], "buy", qty, order_quantity, price, notional, commission, tax, status,
                                                  reason))

            if self.config.liquidate_at_end and day == n_days - 1:
                close_liquid = (np.isfinite(close) & (close > 0) & np.isfinite(amount) & (amount > 0)
                                & np.isfinite(num_trades) & (num_trades > 0))
                close_locked_down = (close <= down_limit + 0.005) & (high <= down_limit + 0.005)
                for j in np.flatnonzero((quantities > 1e-8) & (sellable > 1e-8)):
                    requested = min(quantities[j], sellable[j])
                    if requested <= 1e-8:
                        continue
                    reason = ""
                    if not close_liquid[j]:
                        reason = "收盘停牌或无成交"
                    elif close_locked_down[j]:
                        reason = "收盘一字跌停，清仓未成交"
                    if reason:
                        trade_rows.append(self._trade_row(date, self.symbols[j], "sell", 0, requested, 0, 0, 0, 0,
                                                          "blocked", reason))
                        continue
                    price = close[j] * (1.0 - slippage)
                    if np.isfinite(down_limit[j]):
                        price = max(price, down_limit[j])
                    capacity_quantity = capacity_remaining[j] / price if price > 0 else 0.0
                    if capacity_quantity + 1e-8 >= requested:
                        qty = requested
                    else:
                        qty = math.floor(capacity_quantity / self.config.lot_size) * self.config.lot_size
                    if qty <= 1e-8:
                        trade_rows.append(self._trade_row(date, self.symbols[j], "sell", 0, requested, 0, 0, 0, 0,
                                                          "blocked", "收盘清仓受单股容量限制"))
                        continue
                    notional = qty * price
                    commission, tax = self._fee(notional, self.config.fees, "sell")
                    cash += notional - commission - tax
                    quantities[j] -= qty
                    if quantities[j] <= 1e-8:
                        quantities[j] = 0.0
                        average_cost[j] = 0.0
                    day_turnover += notional
                    capacity_remaining[j] -= notional
                    status = "filled" if qty + 1e-8 >= requested else "partial"
                    reason = "" if status == "filled" else "收盘清仓受单股容量限制"
                    trade_rows.append(self._trade_row(date, self.symbols[j], "sell", qty, requested, price, notional,
                                                      commission, tax, status, reason))
            pending_rebalance = retry_rebalance

            # Carry the last valid factor and close for suspended/missing observations.
            valid_close = np.isfinite(close) & (close > 0)
            valid_open = np.isfinite(open_px) & (open_px > 0)
            close_mark = np.where(valid_close, close, np.where(valid_open, open_px, last_close))
            valid_mark = np.isfinite(close_mark) & (close_mark > 0)
            last_close[valid_mark] = close_mark[valid_mark]
            factor_ok = np.isfinite(factor) & (factor > 0)
            last_factor[factor_ok] = factor[factor_ok]
            market_value = float(np.nansum(quantities * last_close))
            nav = cash + market_value
            nav_rows.append({"date": date, "cash": cash, "market_value": market_value, "nav": nav,
                             "limit_up_count": int(locked_up.sum()),
                             "limit_down_count": int(locked_down.sum()),
                             "st_excluded_count": int(self.st_mask[day].sum()),
                             "st_status_coverage": self._status_coverage(day)})
            gross_turnover.append(day_turnover)

            # A close signal is executed at the next session's open.
            signal = self._strategy_signal(day, weights, quantities, average_cost, cash, last_close)
            if signal is not None:
                weights = self._validated_weights(signal, day)
                pending_rebalance = True

        nav_frame = pd.DataFrame(nav_rows).set_index("date")
        trades_frame = pd.DataFrame(trade_rows, columns=[
            "date", "symbol", "side", "quantity", "requested_quantity", "price", "notional",
            "commission", "tax", "status", "reason"
        ])
        metrics = self._metrics(nav_frame, gross_turnover)
        metrics.update(_trade_metrics(trades_frame))
        return nav_frame, trades_frame, metrics

    @staticmethod
    def _trade_row(date: pd.Timestamp, symbol: str, side: str, qty: float, requested_qty: float,
                   price: float, notional: float, commission: float, tax: float,
                   status: str, reason: str) -> dict[str, object]:
        return {"date": date, "symbol": symbol, "side": side, "quantity": qty, "price": price,
                "requested_quantity": requested_qty, "notional": notional, "commission": commission,
                "tax": tax, "status": status, "reason": reason}

    def _status_coverage(self, day: int) -> float:
        observed = np.isfinite(self.data.values["close"][day]) & (self.data.values["close"][day] > 0)
        if not observed.any():
            return 0.0
        known = np.isfinite(self.data.values["is_st"][day])
        return float((observed & known).sum() / observed.sum())

    def _metrics(self, nav: pd.DataFrame, turnover: list[float]) -> dict[str, float]:
        values = nav["nav"].astype(float)
        daily_returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        elapsed_years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1.0 / 252.0)
        final_value = float(values.iloc[-1])
        initial = float(self.config.initial_cash)
        total_return = final_value / initial - 1.0
        annualized = (final_value / initial) ** (1.0 / elapsed_years) - 1.0 if final_value > 0 else -1.0
        drawdown = values / values.cummax() - 1.0
        volatility = float(daily_returns.std(ddof=1) * math.sqrt(252)) if len(daily_returns) > 1 else 0.0
        daily_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
        sharpe = float(daily_returns.mean() / daily_std * math.sqrt(252)) if daily_std > 0 else 0.0
        avg_nav = float(values.mean())
        return {
            "initial_cash": initial,
            "final_nav": final_value,
            "total_return": total_return,
            "annualized_return": annualized,
            "annualized_volatility": volatility,
            "sharpe_ratio_rf0": sharpe,
            "max_drawdown": float(drawdown.min()),
            "turnover_over_average_nav": float(sum(turnover) / avg_nav) if avg_nav else 0.0,
            "st_status_coverage": float(nav["st_status_coverage"].mean()) if "st_status_coverage" in nav else 0.0,
            "average_limit_up_count": float(nav["limit_up_count"].mean()) if "limit_up_count" in nav else 0.0,
            "average_limit_down_count": float(nav["limit_down_count"].mean()) if "limit_down_count" in nav else 0.0,
            "trading_days": float(len(nav)),
        }


def write_outputs(output_dir: str | Path, nav: pd.DataFrame, trades: pd.DataFrame,
                  metrics: dict[str, float], run_info: dict[str, object]) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    nav.to_csv(directory / "nav.csv", float_format="%.8f", date_format="%Y-%m-%d")
    trades.to_csv(directory / "trades.csv", index=False, float_format="%.8f", date_format="%Y-%m-%d")
    _trade_summary_frame(trades).to_csv(directory / "trade_summary.csv", index=False, float_format="%.8f")
    with open(directory / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with open(directory / "run.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        (nav["nav"] / nav["nav"].iloc[0]).plot(ax=ax, color="#1769aa", lw=1.5)
        ax.set_title("Portfolio NAV")
        ax.set_ylabel("Growth of 1 CNY")
        ax.set_xlabel("")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / "nav.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"submitted_orders": 0.0, "filled_orders": 0.0, "partial_orders": 0.0,
                "blocked_orders": 0.0, "order_completion_rate": 0.0,
                "order_execution_rate": 0.0, "share_fill_rate": 0.0}
    total = int(len(trades))
    status = trades["status"].astype(str)
    requested = (trades["requested_quantity"] if "requested_quantity" in trades
                 else trades["quantity"]).astype(float).clip(lower=0)
    filled_quantity = trades["quantity"].astype(float).clip(lower=0)
    filled = int((status == "filled").sum())
    partial = int((status == "partial").sum())
    blocked = int((status == "blocked").sum())
    requested_total = float(requested.sum())
    return {
        "submitted_orders": float(total),
        "filled_orders": float(filled),
        "partial_orders": float(partial),
        "blocked_orders": float(blocked),
        "order_completion_rate": float(filled / total) if total else 0.0,
        "order_execution_rate": float((filled + partial) / total) if total else 0.0,
        "share_fill_rate": float(filled_quantity.sum() / requested_total) if requested_total else 0.0,
    }


def _trade_summary_frame(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["side", "status", "reason", "orders", "requested_shares", "filled_shares", "share_fill_rate"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    frame = trades.copy()
    if "requested_quantity" not in frame:
        frame["requested_quantity"] = frame["quantity"]
    frame["reason"] = frame["reason"].fillna("").replace("", "正常成交")
    result = frame.groupby(["side", "status", "reason"], dropna=False).agg(
        orders=("symbol", "size"),
        requested_shares=("requested_quantity", "sum"),
        filled_shares=("quantity", "sum"),
    ).reset_index()
    result["share_fill_rate"] = np.divide(
        result["filled_shares"], result["requested_shares"],
        out=np.zeros(len(result), dtype=float), where=result["requested_shares"].to_numpy() > 0,
    )
    return result[columns].sort_values(["side", "status", "reason"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="课程 Feather 日频矩阵回测系统（A 股多头）")
    parser.add_argument("--data", default="data", help="Feather 矩阵目录、feathers.zip 或含该压缩包的目录")
    parser.add_argument("--output", default="runs/momentum", help="结果输出目录")
    parser.add_argument("--start", help="回测起始日，含当天，格式 YYYY-MM-DD")
    parser.add_argument("--end", help="回测结束日，含当天，格式 YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rebalance", choices=["monthly"], default="monthly", help="示例策略按月调仓")
    parser.add_argument("--default-limit", type=float, default=0.10, help="默认涨跌停比例；如 0.10 表示 10%%")
    parser.add_argument("--limit-rules", help="CSV: symbol,limit_pct，可覆盖 ST、IPO 等个股规则；0 表示无限价")
    parser.add_argument("--commission", type=float, default=0.0003, help="佣金率，默认 3 bp")
    parser.add_argument("--min-commission", type=float, default=5.0, help="每笔最低佣金，元")
    parser.add_argument("--stamp-duty", type=float, default=0.0005, help="卖出印花税率，默认 5 bp")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="开盘成交滑点，基点")
    parser.add_argument("--lot-size", type=int, default=100, help="买入整手股数；卖出可整仓清掉零股")
    args = parser.parse_args()

    data = MarketData.load(args.data, start=args.start, end=args.end, warmup=args.lookback + 1)
    strategy = MomentumTopN(lookback=args.lookback, top_n=args.top_n)
    config = EngineConfig(
        initial_cash=args.initial_cash,
        lot_size=args.lot_size,
        default_limit_pct=args.default_limit,
        fees=Fees(args.commission, args.min_commission, args.stamp_duty, args.slippage_bps),
    )
    limits = load_limit_rules(args.limit_rules)
    nav, trades, metrics = Backtester(data, strategy, config, limits).run()
    info = {
        "data": str(args.data),
        "date_start": str(data.dates[data.simulation_start].date()),
        "date_end": str(data.dates[-1].date()),
        "symbols": len(data.symbols),
        "strategy": "momentum_top_n",
        "lookback": args.lookback,
        "top_n": args.top_n,
        "rebalance": args.rebalance,
        "fees": asdict(config.fees),
        "default_limit_pct": config.default_limit_pct,
        "limit_overrides": len(limits),
        "execution": "previous close signal, next session open; long-only; T+1; 100-share buy lots",
        "corporate_actions": "cumadj ratio scales overnight shares; dividends approximated as reinvested",
    }
    write_outputs(args.output, nav, trades, metrics, info)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果已写入：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
