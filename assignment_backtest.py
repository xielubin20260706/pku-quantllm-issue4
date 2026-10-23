#!/usr/bin/env python3
"""Reproducible walk-forward assignment run on the course A-share matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt import Backtester, EngineConfig, Fees, MarketData, write_outputs


START_DATE = pd.Timestamp("2025-01-02")
END_DATE = pd.Timestamp("2026-09-30")
ANCHOR_DATE = pd.Timestamp("2024-12-31")
EXIT_SIGNAL_DATE = pd.Timestamp("2026-09-29")


def _round_cent(values: np.ndarray) -> np.ndarray:
    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0


def infer_st_from_price_limits(data: MarketData, rolling_days: int = 60,
                               hits_required: int = 2) -> tuple[np.ndarray, dict[str, object]]:
    """Point-in-time heuristic: two sealed 5% limit days within 60 sessions flag a symbol.

    It is deliberately labeled an inference, not an official ST history. Detection only
    uses the current and prior rows; once inferred, the flag persists through this sample.
    """
    values = data.values
    dates, symbols = data.dates, data.symbols
    n_days, n_symbols = values["close"].shape
    codes = ["".join(ch for ch in str(symbol).split(".")[0] if ch.isdigit()) for symbol in symbols]
    main_board = np.array([code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
                           for code in codes], dtype=bool)
    chinext = np.array([code.startswith(("300", "301")) for code in codes], dtype=bool)
    known_board_5pct_period = np.zeros(n_days, dtype=bool)
    for day, date in enumerate(dates):
        known_board_5pct_period[day] = date < pd.Timestamp("2026-07-06")

    detected = np.zeros((n_days, n_symbols), dtype=bool)
    ring = np.zeros((rolling_days, n_symbols), dtype=bool)
    recent_hits = np.zeros(n_symbols, dtype=np.int16)
    persistent = np.zeros(n_symbols, dtype=bool)
    events_total = 0
    for day in range(1, n_days):
        previous_close = values["close"][day - 1]
        previous_factor = values["cumadj"][day - 1]
        today_factor = values["cumadj"][day]
        ratio = np.ones(n_symbols, dtype=np.float64)
        valid_factor = (np.isfinite(previous_factor) & (previous_factor > 0)
                        & np.isfinite(today_factor) & (today_factor > 0))
        ratio[valid_factor] = previous_factor[valid_factor] / today_factor[valid_factor]
        reference = previous_close * ratio
        upper = _round_cent(reference * 1.05)
        lower = _round_cent(reference * 0.95)
        op, high, low, close = (values[name][day] for name in ("open", "high", "low", "close"))
        liquid = (np.isfinite(values["amt"][day]) & (values["amt"][day] > 0)
                  & np.isfinite(values["num_trades"][day]) & (values["num_trades"][day] > 0))
        # ST 5% behavior can be inferred on the main board; ChiNext's 10% regime
        # before its 2020 reform is included. The post-2026 10% ST regime is excluded.
        eligible_board = main_board | (chinext & (dates[day] < pd.Timestamp("2020-08-24")))
        eligible = eligible_board & known_board_5pct_period[day] & valid_factor & liquid
        at_up = (np.abs(op - upper) <= 0.0051) & (np.abs(high - upper) <= 0.0051) \
            & (np.abs(low - upper) <= 0.0051) & (np.abs(close - upper) <= 0.0051)
        at_down = (np.abs(op - lower) <= 0.0051) & (np.abs(high - lower) <= 0.0051) \
            & (np.abs(low - lower) <= 0.0051) & (np.abs(close - lower) <= 0.0051)
        event = eligible & np.isfinite(reference) & (reference > 0) & (at_up | at_down)
        events_total += int(event.sum())
        slot = day % rolling_days
        recent_hits -= ring[slot]
        ring[slot] = event
        recent_hits += event
        persistent |= recent_hits >= hits_required
        detected[day] = persistent

    observed = np.isfinite(values["close"]) & (values["close"] > 0)
    official = values.get("is_st", np.full((n_days, n_symbols), np.nan))
    combined = np.where(np.isfinite(official), official >= 0.5, detected)
    status = np.where(observed, combined.astype(np.float64), np.nan)
    info = {
        "method": "two sealed +/-5% one-price sessions within 60 trading days; point-in-time, persistent after detection",
        "price_limit_events": events_total,
        "inferred_symbols": int(detected.any(axis=0).sum()),
        "inferred_symbol_days": int(detected.sum()),
        "official_status_source": data.status_source,
        "is_official_history": data.status_source != "missing",
    }
    data.values["is_st"] = status
    data.status_source = "official_plus_5pct_price_inference" if info["is_official_history"] \
        else "5pct_price_inference_2hits_60d"
    return detected, info


class FiveDayAssignmentStrategy:
    """Five-session portfolios: trimmed reversal, default 5DR, or eligible-universe equal weight."""

    lookback = 5

    def __init__(self, mode: str, anchor_date: pd.Timestamp = ANCHOR_DATE,
                 exit_signal_date: pd.Timestamp = EXIT_SIGNAL_DATE, top_n: int = 20,
                 listing_days: int = 60, liquidity_exclusion: float = 0.20,
                 tail_exclusion: float = 0.10):
        self.mode = mode
        self.anchor_date = anchor_date
        self.exit_signal_date = exit_signal_date
        self.top_n = top_n
        self.listing_days = listing_days
        self.liquidity_exclusion = liquidity_exclusion
        self.tail_exclusion = tail_exclusion
        self.universe_counts: dict[str, dict[str, int]] = {}
        self.anchor_index: int | None = None
        self.first_active = None
        self.name = {
            "trimmed": "剔除两端极值的 5 日反转",
            "five_dr": "系统默认 5DR 五日反转",
            "equal": "同口径过滤后等权基准",
        }[mode]

    def _prepare(self, data: MarketData) -> None:
        if self.anchor_index is None:
            found = data.dates.get_indexer([self.anchor_date])[0]
            if found < 0:
                raise ValueError(f"数据中找不到初始信号日 {self.anchor_date.date()}")
            self.anchor_index = int(found)
        if self.first_active is None:
            valid = np.isfinite(data.values["close"]) & (data.values["close"] > 0)
            any_valid = valid.any(axis=0)
            first_active = valid.argmax(axis=0)
            first_active[~any_valid] = len(data.dates) + 1
            self.first_active = first_active

    def target_weights(self, data: MarketData, day: int, previous_weights: np.ndarray,
                       portfolio=None) -> np.ndarray | None:
        self._prepare(data)
        date = data.dates[day]
        empty = np.zeros(len(data.symbols), dtype=np.float64)
        # Signal after the 2026-09-29 close; the engine executes on 2026-09-30,
        # then attempts a final close liquidation if any capacity-limited shares remain.
        if date == self.exit_signal_date:
            self.universe_counts[date.strftime("%Y-%m-%d")] = {"eligible": 0, "selected": 0}
            return empty
        if day < self.anchor_index or (day - self.anchor_index) % 5:
            return None
        if day < max(self.listing_days, 5) or day == 0:
            self.universe_counts[date.strftime("%Y-%m-%d")] = {"eligible": 0, "selected": 0}
            return empty

        values = data.values
        close = values["close"]
        factor = values["cumadj"]
        adjusted = close * factor
        tradable = (np.isfinite(values["open"][day]) & (values["open"][day] > 0)
                    & np.isfinite(values["amt"][day]) & (values["amt"][day] > 0)
                    & np.isfinite(values["num_trades"][day]) & (values["num_trades"][day] > 0))
        st_values = values["is_st"][day]
        non_st = ~(np.isfinite(st_values) & (st_values >= 0.5))
        old_enough = day - self.first_active >= self.listing_days
        previous_adjusted = adjusted[day - 5:day]
        valid_previous = np.isfinite(previous_adjusted) & (previous_adjusted > 0)
        five_dr_mean = np.divide(
            np.where(valid_previous, previous_adjusted, 0.0).sum(axis=0),
            valid_previous.sum(axis=0),
            out=np.full(len(data.symbols), np.nan),
            where=valid_previous.sum(axis=0) == 5,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            five_dr = adjusted[day] / five_dr_mean - 1.0
            five_day_return = adjusted[day] / adjusted[day - 5] - 1.0
        amount_yesterday = values["amt"][day - 1]
        base = (tradable & non_st & old_enough & np.isfinite(five_dr)
                & np.isfinite(five_day_return) & np.isfinite(amount_yesterday) & (amount_yesterday > 0))
        base_indices = np.flatnonzero(base)
        if not len(base_indices):
            self.universe_counts[date.strftime("%Y-%m-%d")] = {"eligible": 0, "selected": 0}
            return empty
        liquidity_floor = float(np.nanquantile(amount_yesterday[base_indices], self.liquidity_exclusion))
        eligible = base & (amount_yesterday >= liquidity_floor)
        candidates = np.flatnonzero(eligible)
        selected = candidates
        if self.mode == "trimmed":
            ordered = candidates[np.argsort(five_day_return[candidates], kind="stable")]
            trim_each_side = int(np.floor(len(ordered) * self.tail_exclusion))
            high_index = len(ordered) - trim_each_side if trim_each_side else len(ordered)
            selected = ordered[trim_each_side:high_index]
        elif self.mode == "five_dr":
            ordered = candidates[np.argsort(five_dr[candidates], kind="stable")]
            selected = ordered[:min(self.top_n, len(ordered))]
        self.universe_counts[date.strftime("%Y-%m-%d")] = {
            "eligible": int(len(candidates)), "selected": int(len(selected)),
        }
        if not len(selected):
            return empty
        result = np.zeros(len(data.symbols), dtype=np.float64)
        result[selected] = 1.0 / len(selected)
        return result


def _slice_data(data: MarketData, start_date: pd.Timestamp, end_date: pd.Timestamp) -> MarketData:
    start_index = int(data.dates.searchsorted(start_date, side="left"))
    end_index = int(data.dates.searchsorted(end_date, side="right"))
    if start_index >= end_index or data.dates[start_index] != start_date or data.dates[end_index - 1] != end_date:
        raise ValueError("指定的回测起止日必须都在交易日索引中")
    values = {name: array[:end_index].copy() for name, array in data.values.items()}
    return MarketData(data.dates[:end_index], data.symbols, values, start_index, data.status_source)


def _run_one(data: MarketData, name: str, mode: str, output: Path,
             slippage_bps: float, initial_cash: float) -> tuple[dict[str, float], dict[str, object]]:
    strategy = FiveDayAssignmentStrategy(mode)
    config = EngineConfig(
        initial_cash=initial_cash,
        lot_size=100,
        max_daily_turnover_participation=0.10,
        liquidate_at_end=True,
        fees=Fees(commission_rate=0.0001, minimum_commission=0.0,
                  sell_stamp_duty=0.0005, slippage_bps=slippage_bps),
    )
    nav, trades, metrics = Backtester(data, strategy, config).run()
    metrics["ending_market_value"] = float(nav["market_value"].iloc[-1])
    run_info = {
        "strategy": strategy.name,
        "mode": mode,
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "signal_anchor": str(ANCHOR_DATE.date()),
        "end_clear_signal": str(EXIT_SIGNAL_DATE.date()),
        "rebalance_every_trading_sessions": 5,
        "initial_cash": initial_cash,
        "listing_filter_sessions": 60,
        "liquidity_excluded_fraction": 0.20,
        "tail_trim_fraction_each_side": 0.10 if mode == "trimmed" else 0.0,
        "fees": {"commission_rate": 0.0001, "minimum_commission": 0.0,
                 "sell_stamp_duty": 0.0005, "slippage_bps": slippage_bps},
        "max_turnover_participation_previous_day_amount": 0.10,
        "data_symbols": len(data.symbols),
        "status_source": data.status_source,
        "universe_counts_at_signals": strategy.universe_counts,
        "execution": "signal after close; next session open; final close liquidation; A-share daily limits/T+1/board lots",
    }
    write_outputs(output, nav, trades, metrics, run_info)
    with open(output / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return metrics, run_info


def _comparison_plot(paths: dict[str, Path], output: Path, initial_cash: float) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = {"主策略": "Trimmed 5-day return", "5DR基准": "Default 5DR (20 stocks)",
              "等权基准": "Filtered universe equal weight", "主策略5bp滑点": "Trimmed, 5bp slippage"}
    colors = {"主策略": "#17835d", "5DR基准": "#3b78a8", "等权基准": "#d0913c", "主策略5bp滑点": "#a45c79"}
    for label, directory in paths.items():
        nav = pd.read_csv(directory / "nav.csv", parse_dates=["date"])
        ax.plot(nav["date"], nav["nav"] / initial_cash, label=labels[label],
                linewidth=1.7, color=colors.get(label))
    ax.axhline(1.0, color="#8b9690", linewidth=0.8, linestyle="--")
    ax.set_title("A-share daily backtest comparison")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _factor_layers(data: MarketData, output: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Gross equal-weight quintiles, on the same point-in-time eligible universe."""
    start = data.simulation_start
    anchor = int(data.dates.get_indexer([ANCHOR_DATE])[0])
    n_days, n_symbols = data.values["close"].shape
    layer_nav = np.ones((n_days - start, 5), dtype=np.float64)
    holdings = np.zeros((5, n_symbols), dtype=np.float64)
    universe = FiveDayAssignmentStrategy("equal")
    previous_weights = np.zeros(n_symbols, dtype=np.float64)
    close_adjusted = data.values["close"] * data.values["cumadj"]

    for day in range(anchor, n_days):
        if day > anchor:
            previous = close_adjusted[day - 1]
            current = close_adjusted[day]
            valid = (np.isfinite(previous) & (previous > 0) & np.isfinite(current) & (current > 0))
            daily_return = np.zeros(n_symbols, dtype=np.float64)
            daily_return[valid] = current[valid] / previous[valid] - 1.0
            if day >= start:
                row = day - start
                prior_nav = np.ones(5) if row == 0 else layer_nav[row - 1]
                layer_nav[row] = prior_nav * (1.0 + holdings @ daily_return)

        equal_weights = universe.target_weights(data, day, previous_weights)
        if equal_weights is None:
            continue
        previous_weights = equal_weights
        candidates = np.flatnonzero(equal_weights > 0)
        holdings.fill(0.0)
        if not len(candidates) or day < 5:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            score = close_adjusted[day] / close_adjusted[day - 5] - 1.0
        candidates = candidates[np.isfinite(score[candidates])]
        ordered = candidates[np.argsort(score[candidates], kind="stable")]
        for bucket, selected in enumerate(np.array_split(ordered, 5)):
            if len(selected):
                holdings[bucket, selected] = 1.0 / len(selected)

    frame = pd.DataFrame(layer_nav, index=data.dates[start:], columns=[f"Q{i}" for i in range(1, 6)])
    frame.index.name = "date"
    frame.to_csv(output / "layers.csv", float_format="%.8f", date_format="%Y-%m-%d")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#3d78a8", "#4d9d8b", "#8ba54a", "#d29a45", "#bf665d"]
    for index, column in enumerate(frame.columns):
        ax.plot(frame.index, frame[column], label=column, color=colors[index], linewidth=1.7)
    ax.axhline(1.0, color="#8b9690", linewidth=0.8, linestyle="--")
    ax.set_title("5-day return quintiles (gross, equal weighted)")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "layers.png", dpi=160)
    plt.close(fig)
    return frame, {column: float(frame[column].iloc[-1] - 1.0) for column in frame.columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行课程作业口径的 A 股全市场回测")
    parser.add_argument("--data", default="feathers", help="包含 7 个 Feather 矩阵的目录或 feathers.zip")
    parser.add_argument("--output", default="runs/assignment", help="输出目录")
    parser.add_argument("--cash", type=float, default=1_000_000_000.0, help="初始资金，默认 10 亿元")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="主策略基础滑点，默认 0bp")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = MarketData.load(args.data)
    _, inference_info = infer_st_from_price_limits(data)
    data = _slice_data(data, START_DATE, END_DATE)
    metrics_by_name: dict[str, dict[str, float]] = {}
    runs: dict[str, Path] = {}
    settings = [
        ("main", "主策略", "trimmed", args.slippage_bps),
        ("five_dr", "5DR基准", "five_dr", args.slippage_bps),
        ("equal", "等权基准", "equal", args.slippage_bps),
        ("main_slip5", "主策略5bp滑点", "trimmed", 5.0),
    ]
    for folder, label, mode, slip in settings:
        print(f"Running {label} ({slip:g} bp slippage)...", flush=True)
        run_dir = output / folder
        metrics, run_info = _run_one(data, label, mode, run_dir, slip, args.cash)
        metrics_by_name[label] = metrics
        runs[label] = run_dir
        run_info["st_inference"] = inference_info
        with open(run_dir / "run.json", "w", encoding="utf-8") as handle:
            json.dump(run_info, handle, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame([
        {"策略": label, "总收益": metrics["total_return"], "年化收益": metrics["annualized_return"],
         "最大回撤": metrics["max_drawdown"], "夏普": metrics["sharpe_ratio_rf0"],
         "期末净值": metrics["final_nav"], "期末持仓市值": metrics["ending_market_value"],
         "委托尝试记录": metrics["submitted_orders"],
         "完整成交率": metrics["order_completion_rate"], "股数成交率": metrics["share_fill_rate"],
         "状态数据覆盖(含推断)": metrics["st_status_coverage"]}
        for label, metrics in metrics_by_name.items()
    ])
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    _comparison_plot(runs, output / "comparison_nav.png", args.cash)
    _, layer_returns = _factor_layers(data, output)

    main_return = metrics_by_name["主策略"]["total_return"]
    benchmark_return = metrics_by_name["等权基准"]["total_return"]
    five_dr_return = metrics_by_name["5DR基准"]["total_return"]
    slip_difference = main_return - metrics_by_name["主策略5bp滑点"]["total_return"]
    main_info = json.loads((runs["主策略"] / "run.json").read_text())
    signal_counts = list(main_info["universe_counts_at_signals"].values())
    mean_eligible = float(np.mean([row["eligible"] for row in signal_counts])) if signal_counts else 0.0
    mean_selected = float(np.mean([row["selected"] for row in signal_counts])) if signal_counts else 0.0
    report = f'''# A 股日频回测作业报告

## 实验口径

- 数据：课程 Feather 全市场矩阵，{len(data.symbols):,} 只证券；回测区间 2025-01-02 至 2026-09-30。
- 信号锚点：2024-12-31 收盘；次交易日开盘建仓；之后每 5 个交易日收盘更新一次信号。
- 清仓：2026-09-29 收盘发出零仓位目标，2026-09-30 开盘执行；剩余可交易持仓在 9 月 30 日收盘按剩余容量尝试清仓。
- 资金及约束：初始资金 ¥{args.cash:,.0f}；买入 100 股整手；每只证券每日买卖合计不超过前一交易日成交额的 10%。
- 费用：佣金单边 1 bp、无最低佣金；卖出印花税 5 bp；主结果滑点 {args.slippage_bps:g} bp。
- 股票池：上市至少 60 个交易日；按前一交易日成交额剔除候选池中流动性最低 20%；信号日有有效行情和成交。
- ST 推断：课程矩阵没有 ST 标记。使用无未来信息的规则：60 个交易日内两次出现 ±5% 一字板，则从第二次起推断为 ST 并持续剔除。它不是官方历史 ST 名单。

## 策略

主策略每 5 个交易日计算复权 5 日收益，剔除收益最高和最低的各 10%，其余证券等权持有。运行时参数固定为：5 日窗口、两端各 10%、60 日上市期、成交额最低 20% 过滤、10% 单股成交容量。此包没有声称做过训练期网格搜索；参数按提交规格固定，结果为 2025–2026 评估区间的实跑结果。

对照组使用相同区间、股票池、调仓节奏、资金、费率、容量及涨跌停/T+1 规则：系统默认 5DR 反转策略（20 只）和合格股票等权基准。另运行主策略 5 bp 滑点敏感性。

这里的默认 5DR 对照按系统策略模板定义为筛选值最低的 20 只股票；不同代码对 5DR 的方向、持仓数或执行口径定义不同，结果不能直接横向照搬。

## 实跑结果

| 策略 | 总收益 | 年化收益 | 最大回撤 | 夏普 | 期末净值 | 期末持仓市值 |
|---|---:|---:|---:|---:|---:|---:|
| 主策略（剔除两端极值） | {metrics_by_name['主策略']['total_return']:.2%} | {metrics_by_name['主策略']['annualized_return']:.2%} | {metrics_by_name['主策略']['max_drawdown']:.2%} | {metrics_by_name['主策略']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['主策略']['final_nav']:,.0f} | ¥{metrics_by_name['主策略']['ending_market_value']:,.0f} |
| 系统默认 5DR（20 只） | {five_dr_return:.2%} | {metrics_by_name['5DR基准']['annualized_return']:.2%} | {metrics_by_name['5DR基准']['max_drawdown']:.2%} | {metrics_by_name['5DR基准']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['5DR基准']['final_nav']:,.0f} | ¥{metrics_by_name['5DR基准']['ending_market_value']:,.0f} |
| 合格股票等权基准 | {benchmark_return:.2%} | {metrics_by_name['等权基准']['annualized_return']:.2%} | {metrics_by_name['等权基准']['max_drawdown']:.2%} | {metrics_by_name['等权基准']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['等权基准']['final_nav']:,.0f} | ¥{metrics_by_name['等权基准']['ending_market_value']:,.0f} |
| 主策略（5 bp 滑点） | {metrics_by_name['主策略5bp滑点']['total_return']:.2%} | {metrics_by_name['主策略5bp滑点']['annualized_return']:.2%} | {metrics_by_name['主策略5bp滑点']['max_drawdown']:.2%} | {metrics_by_name['主策略5bp滑点']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['主策略5bp滑点']['final_nav']:,.0f} | ¥{metrics_by_name['主策略5bp滑点']['ending_market_value']:,.0f} |

相对等权基准超额：{main_return - benchmark_return:.2%}。主策略 5 bp 滑点相对基础情形少 {slip_difference:.2%}。调仓日平均合格证券 {mean_eligible:,.0f} 只，主策略平均持有 {mean_selected:,.0f} 只。

五层因子诊断期末毛收益：Q1 {layer_returns['Q1']:.2%}、Q2 {layer_returns['Q2']:.2%}、Q3 {layer_returns['Q3']:.2%}、Q4 {layer_returns['Q4']:.2%}、Q5 {layer_returns['Q5']:.2%}。Q1 到 Q5 按复权 5 日收益从低到高分组。

![策略净值对比](comparison_nav.png)

![Q1–Q5 五日收益分层](layers.png)

## 实现与可复核材料

- `quantbt.py`：撮合引擎，支持 T+1、整手、涨跌停、受阻订单重试、容量约束、费率和期末清仓。
- `assignment_backtest.py`：按以上固定口径运行三组策略和 5 bp 敏感性对照。
- 每个结果子目录提供 `nav.csv`、`trades.csv`、`trade_summary.csv`、`metrics.json`、`run.json` 和净值图。
- `comparison.csv` 汇总四组结果；`run.json` 记录参数、数据文件快照、ST 推断和每次调仓股票数量。逐笔委托文件按每次尝试记录，受阻订单会重试，因此它的行数不是唯一委托数。
- `layers.csv` 和 `layers.png` 展示同一过滤股票池中，按 5 日收益排序后每 5 个交易日更新的 Q1–Q5 毛收益分层；分层不计费用、容量及涨跌停撮合。

## 结果限制

ST 识别是 OHLCV 价格行为推断，可能漏判或误判；它不能替代官方逐日 ST 名单。退市和长期停牌股票会按原始日线保持持仓，订单无法成交时继续记录阻塞。日线数据不含盘口队列，涨跌停和容量撮合仍是近似。固定费率不代表历史费率。策略参数按用户给定规格固定，未做 2020–2023 调参验证，也不应把这组回测结果解释为显著超额或收益保证。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    with open(output / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics_by_name, "st_inference": inference_info}, handle, ensure_ascii=False, indent=2)
    print(report)
    print(f"\n结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()
