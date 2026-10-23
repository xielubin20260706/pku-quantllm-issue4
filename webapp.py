#!/usr/bin/env python3
"""Local browser UI for the Feather A-share backtester."""

from __future__ import annotations

import argparse
import ast
import json
import operator
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from quantbt import (Backtester, EngineConfig, Fees, MATRICES, MarketData, PortfolioSnapshot,
                     load_limit_rules, write_outputs)


FIELDS = {"open", "high", "low", "close", "amt", "num_trades", "cumadj", "five_dr", "is_st", "limit_pct"}
BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def parse_factor(expression: str) -> tuple[ast.Expression, int]:
    """Validate a small arithmetic factor language and return its lookback."""
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("因子表达式语法有误") from exc

    lookback = 0

    def visit(node: ast.AST) -> None:
        nonlocal lookback
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise ValueError(f"不认识字段 {node.id}；可用字段：{', '.join(sorted(FIELDS))}")
            if node.id == "five_dr":
                lookback = max(lookback, 5)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError("表达式里只允许数字常量")
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARYOPS:
            visit(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn not in {"delay", "mean"} or len(node.args) != 2 or node.keywords:
                raise ValueError("仅支持 delay(字段, 天数) 和 mean(字段, 天数)")
            field, period = node.args
            if not isinstance(field, ast.Name) or field.id not in FIELDS:
                raise ValueError(f"{fn} 的第一个参数必须是一个可用字段")
            if not isinstance(period, ast.Constant) or not isinstance(period.value, int) or isinstance(period.value, bool):
                raise ValueError(f"{fn} 的天数必须是正整数")
            if not 1 <= period.value <= 1000:
                raise ValueError("窗口天数必须在 1 到 1000 之间")
            field_lookback = 5 if field.id == "five_dr" else 0
            needed = period.value + field_lookback if fn == "delay" else period.value - 1 + field_lookback
            lookback = max(lookback, needed)
        else:
            raise ValueError("表达式包含不支持的语法；请使用字段、四则运算、幂运算、delay 或 mean")

    visit(root)
    return root, lookback


class FormulaTopN:
    """Monthly top-N strategy driven by a safe arithmetic cross-sectional factor."""

    def __init__(self, expression: str, top_n: int):
        self.expression = expression
        self.tree, self.lookback = parse_factor(expression)
        if top_n < 1:
            raise ValueError("持仓只数必须为正整数")
        self.top_n = top_n
        self.last_month: tuple[int, int] | None = None

    def _value(self, node: ast.AST, data: MarketData, day: int):
        if isinstance(node, ast.Expression):
            return self._value(node.body, data, day)
        if isinstance(node, ast.Name):
            return _field_value(data, node.id, day)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return BINOPS[type(node.op)](self._value(node.left, data, day), self._value(node.right, data, day))
        if isinstance(node, ast.UnaryOp):
            return UNARYOPS[type(node.op)](self._value(node.operand, data, day))
        if isinstance(node, ast.Call):
            field = node.args[0].id
            period = int(node.args[1].value)
            if node.func.id == "delay":
                index = day - period
                if index < 0:
                    return np.full(len(data.symbols), np.nan)
                return _field_value(data, field, index)
            start = max(0, day - period + 1)
            window = np.stack([_field_value(data, field, i) for i in range(start, day + 1)])
            valid = np.isfinite(window)
            count = valid.sum(axis=0)
            total = np.where(valid, window, 0.0).sum(axis=0)
            return np.divide(total, count, out=np.full(len(data.symbols), np.nan), where=count > 0)
        raise ValueError("不支持的因子表达式")

    def score(self, data: MarketData, day: int) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.broadcast_to(
                np.asarray(self._value(self.tree, data, day), dtype=np.float64),
                (len(data.symbols),),
            )

    def target_weights(self, data: MarketData, day: int, previous_weights: np.ndarray) -> np.ndarray | None:
        current_month = (data.dates[day].year, data.dates[day].month)
        signal_ready = day >= self.lookback
        if not signal_ready:
            return None
        rebalance = current_month != self.last_month or (previous_weights.sum() == 0 and signal_ready)
        if not rebalance:
            return None
        self.last_month = current_month
        weights = np.zeros(len(data.symbols), dtype=np.float64)
        score = self.score(data, day)
        liquid = (data.values["amt"][day] > 0) & (data.values["num_trades"][day] > 0)
        status = data.values.get("is_st")
        not_st = np.ones(len(data.symbols), dtype=bool) if status is None else ~(np.isfinite(status[day]) & (status[day] >= 0.5))
        valid = np.isfinite(score) & liquid & not_st
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return weights
        count = min(self.top_n, len(candidates))
        best = candidates[np.argpartition(score[candidates], -count)[-count:]]
        weights[best] = 1.0 / len(best)
        return weights


def calculate_factor_layers(data: MarketData, source: FormulaTopN | str, groups: int = 5,
                            ascending: bool = True, clear_if_empty: bool = False) -> pd.DataFrame:
    """Build gross, equal-weight factor buckets using next-session close returns."""
    start = data.simulation_start
    day_count, symbol_count = data.values["close"].shape
    columns = [f"Q{i + 1}" for i in range(groups)]
    layer_nav = np.ones((day_count - start, groups), dtype=np.float64)
    holdings = np.zeros((groups, symbol_count), dtype=np.float64)
    last_month: tuple[int, int] | None = None
    values = data.values

    for day in range(start, day_count):
        row = day - start
        if day > start:
            previous_close = values["close"][day - 1] * values["cumadj"][day - 1]
            current_close = values["close"][day] * values["cumadj"][day]
            valid = (np.isfinite(previous_close) & (previous_close > 0)
                     & np.isfinite(current_close) & (current_close > 0))
            daily_return = np.zeros(symbol_count, dtype=np.float64)
            daily_return[valid] = current_close[valid] / previous_close[valid] - 1.0
            layer_nav[row] = layer_nav[row - 1] * (1.0 + holdings @ daily_return)

        month = (data.dates[day].year, data.dates[day].month)
        if month == last_month:
            continue
        if isinstance(source, FormulaTopN):
            if day < source.lookback:
                continue
            score = source.score(data, day)
        else:
            field_lookback = 5 if source == "five_dr" else 0
            if day < field_lookback:
                continue
            score = _field_value(data, source, day)

        liquid = (values["amt"][day] > 0) & (values["num_trades"][day] > 0)
        status = values.get("is_st")
        not_st = np.ones(symbol_count, dtype=bool) if status is None else ~(np.isfinite(status[day]) & (status[day] >= 0.5))
        candidates = np.flatnonzero(np.isfinite(score) & liquid & not_st)
        if not len(candidates):
            if clear_if_empty:
                holdings.fill(0.0)
                last_month = month
            continue
        ordered = candidates[np.argsort(score[candidates], kind="stable")]
        if not ascending:
            ordered = ordered[::-1]
        holdings.fill(0.0)
        for bucket, selected in enumerate(np.array_split(ordered, groups)):
            if len(selected):
                holdings[bucket, selected] = 1.0 / len(selected)
        last_month = month

    return pd.DataFrame(layer_nav, index=data.dates[start:], columns=columns)


class StrategyContext:
    """Read-only, point-in-time view passed to user Python strategies."""

    def __init__(self, data: MarketData, day: int, previous_weights: np.ndarray,
                 params: dict[str, object], portfolio: PortfolioSnapshot | None = None):
        self.date = data.dates[day]
        self.symbols = tuple(map(str, data.symbols))
        self.available_fields = tuple(dict.fromkeys((*data.values.keys(), "five_dr")))
        self.previous_weights = previous_weights.copy()
        self.params = dict(params)
        if portfolio is None:
            portfolio = PortfolioSnapshot(
                positions=np.zeros(len(data.symbols)), average_cost=np.zeros(len(data.symbols)),
                mark_prices=np.full(len(data.symbols), np.nan), cash=0.0, market_value=0.0, nav=0.0,
            )
        self.positions = portfolio.positions.copy()
        self.average_cost = portfolio.average_cost.copy()
        self.mark_prices = portfolio.mark_prices.copy()
        self.cash = portfolio.cash
        self.market_value = portfolio.market_value
        self.nav = portfolio.nav
        self._data = data
        self._day = day

    def current(self, field: str) -> np.ndarray:
        """Return today's cross-section for a core or additional Feather field."""
        if field not in self.available_fields:
            raise ValueError(f"未知行情字段：{field}")
        return _field_value(self._data, field, self._day).copy()

    def history(self, field: str, window: int) -> np.ndarray:
        """Return up to `window` rows ending today; no future rows are exposed."""
        if field not in self.available_fields:
            raise ValueError(f"未知行情字段：{field}")
        if not isinstance(window, (int, np.integer)) or window < 1 or window > 1001:
            raise ValueError("history 窗口必须是 1 到 1001 的整数")
        start = max(0, self._day - int(window) + 1)
        if field in self._data.values and field != "five_dr":
            return self._data.values[field][start : self._day + 1].copy()
        return np.stack([_field_value(self._data, field, i) for i in range(start, self._day + 1)])

    def tradable(self) -> np.ndarray:
        """Stocks with an opening price and reported trading activity today."""
        values = self._data.values
        tradable = (np.isfinite(values["open"][self._day]) & (values["open"][self._day] > 0)
                    & np.isfinite(values["amt"][self._day]) & (values["amt"][self._day] > 0)
                    & np.isfinite(values["num_trades"][self._day]) & (values["num_trades"][self._day] > 0))
        status = values.get("is_st")
        if status is not None:
            tradable &= ~(np.isfinite(status[self._day]) & (status[self._day] >= 0.5))
        return tradable.copy()


def _field_value(data: MarketData, field: str, day: int) -> np.ndarray:
    """Resolve a point-in-time field, including the built-in adjusted 5DR factor."""
    if field != "five_dr":
        if field not in data.values:
            raise ValueError(f"未知行情字段：{field}")
        return data.values[field][day]
    size = len(data.symbols)
    result = np.full(size, np.nan, dtype=np.float64)
    if day < 5:
        return result
    close = data.values["close"]
    adjustment = data.values["cumadj"]
    prior = close[day - 5 : day] * adjustment[day - 5 : day]
    valid = np.isfinite(prior) & (prior > 0)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, prior, 0.0).sum(axis=0), counts,
                      out=np.full(size, np.nan), where=counts == 5)
    today = close[day] * adjustment[day]
    valid_today = np.isfinite(today) & (today > 0) & np.isfinite(means) & (means > 0)
    np.divide(today, means, out=result, where=valid_today)
    result[valid_today] -= 1.0
    return result


class PythonStrategy:
    """Stateful user strategy whose callback runs after each daily close."""

    def __init__(self, source: str, params: dict[str, object]):
        namespace = {"np": np, "pd": pd}
        exec(compile(source, "<local-strategy>", "exec"), namespace)
        strategy_class = namespace.get("Strategy")
        if not isinstance(strategy_class, type):
            raise ValueError("策略代码需要定义 class Strategy")
        self.instance = strategy_class()
        if not callable(getattr(self.instance, "target_weights", None)):
            raise ValueError("Strategy 类需要实现 target_weights(ctx)")
        self.params = params
        self.name = str(getattr(self.instance, "name", "Python 策略"))
        self.stratification_field = str(getattr(self.instance, "stratification_field", "five_dr")).strip()
        self.stratification_ascending = bool(getattr(self.instance, "stratification_ascending", True))
        try:
            self.lookback = int(getattr(self.instance, "lookback", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Strategy.lookback 必须是 0 到 1000 的整数") from exc
        if not 0 <= self.lookback <= 1000:
            raise ValueError("Strategy.lookback 必须在 0 到 1000 之间")

    def target_weights(self, data: MarketData, day: int, previous_weights: np.ndarray,
                       portfolio: PortfolioSnapshot) -> np.ndarray | None:
        context = StrategyContext(data, day, previous_weights, self.params, portfolio)
        return self.instance.target_weights(context)


ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"
DATA_SOURCE: Path
OUTPUT_DIR: Path
_data: MarketData | None = None
_data_signature: tuple[tuple[str, int, int], ...] | None = None
_data_lock = threading.Lock()
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quantbt")


def _source_signature() -> tuple[tuple[str, int, int], ...]:
    if DATA_SOURCE.is_dir():
        direct_matrices = all((DATA_SOURCE / f"{name}.feather").exists() for name in MATRICES)
        if direct_matrices:
            files = sorted([*DATA_SOURCE.glob("*.feather"),
                            *[DATA_SOURCE / name for name in ("security_status.csv", "limit_rules.csv")
                              if (DATA_SOURCE / name).is_file()]])
        else:
            archives = ([DATA_SOURCE / "feathers.zip"] if (DATA_SOURCE / "feathers.zip").exists()
                        else sorted(DATA_SOURCE.glob("*.zip")))
            files = [*archives[:1], *[DATA_SOURCE / name for name in ("security_status.csv", "limit_rules.csv")
                                      if (DATA_SOURCE / name).is_file()]]
    else:
        files = [DATA_SOURCE, *[DATA_SOURCE.parent / name for name in ("security_status.csv", "limit_rules.csv")
                                 if (DATA_SOURCE.parent / name).is_file()]]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files if path.is_file())


def _load_data() -> MarketData:
    global _data, _data_signature
    signature = _source_signature()
    if _data is None or signature != _data_signature:
        with _data_lock:
            signature = _source_signature()
            if _data is None or signature != _data_signature:
                _data = MarketData.load(DATA_SOURCE)
                _data_signature = signature
    return _data


def _series(nav: pd.DataFrame) -> list[dict[str, object]]:
    normalized = nav["nav"].astype(float) / float(nav["nav"].iloc[0])
    return [
        {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 6)}
        for date, value in zip(nav.index, normalized)
    ]


def _read_artifact(run_id: str, filename: str) -> dict[str, object]:
    folder = OUTPUT_DIR.parent / "momentum" if run_id == "sample" else OUTPUT_DIR / run_id
    nav = pd.read_csv(folder / "nav.csv", parse_dates=["date"], index_col="date")
    with open(folder / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(folder / "run.json", encoding="utf-8") as handle:
        run_info = json.load(handle)
    trades = pd.read_csv(folder / "trades.csv")
    blocked = int((trades["status"] == "blocked").sum()) if len(trades) else 0
    filled = int(trades["status"].isin(["filled", "partial"]).sum()) if len(trades) else 0
    partial = int((trades["status"] == "partial").sum()) if len(trades) else 0
    summary_path = folder / "trade_summary.csv"
    reason_breakdown = pd.read_csv(summary_path).to_dict(orient="records") if summary_path.exists() else []
    submitted = int(metrics.get("submitted_orders", len(trades)))
    result = {
        "run_id": run_id,
        "metrics": metrics,
        "series": _series(nav),
        "filled_orders": filled,
        "blocked_orders": blocked,
        "order_audit": {
            "submitted_orders": submitted,
            "filled_orders": int(metrics.get("filled_orders", max(0, filled - partial))),
            "partial_orders": int(metrics.get("partial_orders", partial)),
            "blocked_orders": int(metrics.get("blocked_orders", blocked)),
            "order_completion_rate": float(metrics.get("order_completion_rate", 0.0)),
            "order_execution_rate": float(metrics.get("order_execution_rate", 0.0)),
            "share_fill_rate": float(metrics.get("share_fill_rate", 0.0)),
            "reasons": reason_breakdown,
        },
        "period": [nav.index[0].strftime("%Y-%m-%d"), nav.index[-1].strftime("%Y-%m-%d")],
        "downloads": {"nav": f"/download?run={run_id}&file=nav.csv", "trades": f"/download?run={run_id}&file=trades.csv",
                       "trade_summary": f"/download?run={run_id}&file=trade_summary.csv"},
        "strategy_name": run_info.get("strategy_name", "20 日动量示例" if run_id == "sample" else "自定义策略"),
        "data_asof": run_info.get("data_asof", nav.index[-1].strftime("%Y-%m-%d")),
        "symbols": run_info.get("symbols", 0),
        "status_source": run_info.get("status_source", "missing"),
        "st_status_coverage": float(metrics.get("st_status_coverage", 0.0)),
        "average_limit_up_count": float(metrics.get("average_limit_up_count", 0.0)),
        "average_limit_down_count": float(metrics.get("average_limit_down_count", 0.0)),
        "data_files": run_info.get("data_files", []),
        "layer_factor": run_info.get("layer_factor", ""),
        "layer_order": run_info.get("layer_order", "低值到高值"),
        "source": filename,
    }
    layers_path = folder / "layers.csv"
    if layers_path.exists():
        layer_frame = pd.read_csv(layers_path, parse_dates=["date"], index_col="date")
        names = list(layer_frame.columns)
        result["layers"] = []
        for i, name in enumerate(names):
            if i == 0:
                edge = "低值" if result["layer_order"] == "低值到高值" else "高值"
            elif i == len(names) - 1:
                edge = "高值" if result["layer_order"] == "低值到高值" else "低值"
            else:
                edge = ""
            result["layers"].append({
                "label": f"{name} {edge}组".strip() if edge else name,
                "total_return": float(layer_frame[name].iloc[-1] - 1.0),
                "series": [
                    {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 6)}
                    for date, value in layer_frame[name].items()
                ],
            })
    if (folder / "strategy.py").exists():
        result["downloads"]["strategy"] = f"/download?run={run_id}&file=strategy.py"
    if layers_path.exists():
        result["downloads"]["layers"] = f"/download?run={run_id}&file=layers.csv"
    if not summary_path.exists():
        result["downloads"].pop("trade_summary", None)
    if run_id == "sample":
        result["downloads"].pop("trade_summary", None)
    if run_id != "sample" and (folder / "run.json").exists():
        result["downloads"]["run"] = f"/download?run={run_id}&file=run.json"
    return result


def _execute(run_id: str, settings: dict[str, object]) -> None:
    try:
        with _jobs_lock:
            _jobs[run_id]["message"] = "载入 7 个行情矩阵…首次运行可能需要一点时间"
        mode = str(settings.get("mode", "formula"))
        params = {"top_n": int(settings["top_n"]), "initial_cash": float(settings["initial_cash"])}
        formula = str(settings.get("formula", "")).strip()
        source = str(settings.get("strategy_code", ""))
        if mode == "python":
            strategy = PythonStrategy(source, params)
            strategy_name = strategy.name
            layer_source: FormulaTopN | str = strategy.stratification_field
            layer_ascending = strategy.stratification_ascending
        else:
            strategy = FormulaTopN(formula, params["top_n"])
            strategy_name = "5DR 五日反转" if formula == "-five_dr" else "因子 Top N"
            layer_source = strategy
            layer_ascending = True
        start = str(settings.get("start") or "").strip() or None
        end = str(settings.get("end") or "").strip() or None
        source_signature = _source_signature()
        layer_lookback = layer_source.lookback if isinstance(layer_source, FormulaTopN) else (5 if layer_source == "five_dr" else 0)
        if start is None and end is None:
            data = _load_data()
            source_signature = _data_signature or source_signature
        else:
            data = MarketData.load(DATA_SOURCE, start, end, warmup=max(strategy.lookback, layer_lookback) + 1)
        if isinstance(layer_source, str) and layer_source not in data.values and layer_source != "five_dr":
            raise ValueError(f"分层字段 {layer_source} 不在当前数据中；请检查 Feather 矩阵文件名")
        with _jobs_lock:
            _jobs[run_id]["message"] = f"正在计算 {len(data.dates) - data.simulation_start:,} 个交易日…"
        config = EngineConfig(
            initial_cash=float(settings["initial_cash"]),
            fees=Fees(0.0003, 5.0, 0.0005, 2.0),
        )
        sidecar_dir = DATA_SOURCE if DATA_SOURCE.is_dir() else DATA_SOURCE.parent
        limit_rule_path = sidecar_dir / "limit_rules.csv"
        limit_rules = load_limit_rules(limit_rule_path if limit_rule_path.is_file() else None)
        nav, trades, metrics = Backtester(data, strategy, config, limit_rules).run()
        with _jobs_lock:
            _jobs[run_id]["message"] = "正在计算因子五层收益…"
        layer_order = "低值到高值" if layer_ascending else "高值到低值"
        layer_frame = calculate_factor_layers(
            data, layer_source, groups=5, ascending=layer_ascending, clear_if_empty=mode == "formula"
        )
        source_files = [
            {"name": Path(path).name, "bytes": size, "modified_ns": modified_ns}
            for path, size, modified_ns in source_signature
        ]
        info = {
            "strategy": "python" if mode == "python" else "formula_top_n",
            "strategy_name": strategy_name,
            "formula": formula,
            "top_n": int(settings["top_n"]),
            "lookback": strategy.lookback,
            "start": str(nav.index[0].date()),
            "end": str(nav.index[-1].date()),
            "data_asof": str(data.dates[-1].date()),
            "data_files": source_files,
            "symbols": len(data.symbols),
            "status_source": data.status_source,
            "status_coverage": metrics.get("st_status_coverage", 0.0),
            "limit_rule_file": limit_rule_path.name if limit_rule_path.is_file() else None,
            "limit_rule_count": len(limit_rules),
            "layer_factor": formula if mode == "formula" else layer_source,
            "layer_order": layer_order,
            "layer_return_method": "每月分组后，下一交易日起按复权收盘到收盘收益等权持有；毛收益，不含费用与成交约束",
            "execution": "前一收盘因子信号，次日开盘交易；A 股涨跌停、T+1、整手、费率和滑点约束",
        }
        out = OUTPUT_DIR / run_id
        write_outputs(out, nav, trades, metrics, info)
        layer_frame.to_csv(out / "layers.csv", index_label="date", float_format="%.8f")
        if mode == "python":
            (out / "strategy.py").write_text(source, encoding="utf-8")
        result = _read_artifact(run_id, "web")
        if mode == "python":
            result["strategy_name"] = strategy_name
        else:
            result["formula"] = formula
        with _jobs_lock:
            _jobs[run_id].update({"status": "done", "message": "回测完成", "result": result})
    except Exception as exc:
        with _jobs_lock:
            _jobs[run_id].update({"status": "error", "message": str(exc)})


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantBT/1.0"

    def _json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            raw = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        static_downloads = {
            "/security-status-template.csv": ROOT / "security_status_template.csv",
            "/limit-rules-template.csv": ROOT / "limit_rules_template.csv",
            "/strategy_template.py": ROOT / "strategy_template.py",
        }
        if parsed.path in static_downloads:
            target = static_downloads[parsed.path]
            raw = target.read_bytes()
            content_type = "text/x-python; charset=utf-8" if target.suffix == ".py" else "text/csv; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/latest":
            recent_runs = [
                path for path in OUTPUT_DIR.iterdir()
                if path.is_dir() and all((path / name).is_file() for name in ("nav.csv", "metrics.json", "trades.csv"))
            ]
            if recent_runs:
                latest_run = max(recent_runs, key=lambda path: path.stat().st_mtime_ns)
                try:
                    self._json(_read_artifact(latest_run.name, "最近回测"))
                except Exception as exc:
                    self._json({"available": False, "message": str(exc)})
                return
            latest = OUTPUT_DIR.parent / "momentum"
            if (latest / "nav.csv").exists() and (latest / "metrics.json").exists() and (latest / "trades.csv").exists():
                try:
                    result = _read_artifact("sample", "示例回测")
                    result["downloads"] = {
                        "nav": "/download?run=sample&file=nav.csv",
                        "trades": "/download?run=sample&file=trades.csv",
                    }
                    self._json(result)
                except Exception as exc:
                    self._json({"available": False, "message": str(exc)})
            else:
                self._json({"available": False})
            return
        if parsed.path == "/api/run":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            with _jobs_lock:
                job = _jobs.get(run_id)
            self._json(job if job else {"status": "missing", "message": "找不到这次回测"}, 200 if job else 404)
            return
        if parsed.path == "/download":
            query = parse_qs(parsed.query)
            run_id = query.get("run", [""])[0]
            filename = query.get("file", [""])[0]
            if filename not in {"nav.csv", "trades.csv", "trade_summary.csv", "layers.csv", "strategy.py", "run.json"} or (run_id != "sample" and not run_id.replace("-", "").isalnum()):
                self.send_error(404)
                return
            if run_id == "sample" and filename in {"strategy.py", "run.json"}:
                self.send_error(404)
                return
            target = (OUTPUT_DIR.parent / "momentum" / filename) if run_id == "sample" else (OUTPUT_DIR / run_id / filename)
            if not target.is_file():
                self.send_error(404)
                return
            raw = target.read_bytes()
            self.send_response(200)
            content_type = {".csv": "text/csv; charset=utf-8", ".py": "text/x-python; charset=utf-8",
                            ".json": "application/json; charset=utf-8"}.get(target.suffix, "application/octet-stream")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            settings = json.loads(self.rfile.read(length))
            mode = str(settings.get("mode", "formula"))
            formula = str(settings.get("formula", "")).strip()
            strategy_code = str(settings.get("strategy_code", ""))
            if mode == "python":
                if not strategy_code.strip():
                    raise ValueError("请先填写 Python 策略代码")
                if len(strategy_code.encode("utf-8")) > 200_000:
                    raise ValueError("策略代码不能超过 200 KB")
            elif mode == "formula":
                if not formula:
                    raise ValueError("请先填写因子表达式")
                parse_factor(formula)
            else:
                raise ValueError("未知策略模式")
            top_n = int(settings.get("top_n", 20))
            initial_cash = float(settings.get("initial_cash", 1_000_000))
            if not 1 <= top_n <= 500:
                raise ValueError("持仓只数应在 1 到 500 之间")
            if not np.isfinite(initial_cash) or initial_cash <= 0:
                raise ValueError("初始资金必须是正数")
            settings["top_n"] = top_n
            settings["initial_cash"] = initial_cash
            run_id = uuid.uuid4().hex[:12]
            with _jobs_lock:
                _jobs[run_id] = {"status": "running", "message": "正在准备回测…"}
            _executor.submit(_execute, run_id, settings)
            self._json({"run_id": run_id, "status": "running"}, 202)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    global DATA_SOURCE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="启动本地网页回测工作台")
    parser.add_argument("--data", default="feathers", help="Feather 矩阵目录、feathers.zip 或含压缩包的目录")
    parser.add_argument("--output", default="runs/web", help="网页回测结果目录")
    parser.add_argument("--host", default="127.0.0.1", help="默认只在本机开放")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DATA_SOURCE = Path(args.data).resolve()
    OUTPUT_DIR = Path(args.output).resolve()
    if not DATA_SOURCE.exists():
        raise SystemExit(f"找不到行情数据：{DATA_SOURCE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"回测网页已启动：http://{args.host}:{args.port}")
    print(f"行情数据：{DATA_SOURCE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n网页服务已停止")
    finally:
        server.server_close()
        _executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
