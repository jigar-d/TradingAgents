"""Run the daily, selective swing-trade scan.

This script researches candidates only. It never places a brokerage order.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily TradingAgents strategy scan")
    parser.add_argument("--config", default="strategy.json")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = DEFAULT_CONFIG.copy()
    config["checkpoint_enabled"] = True
    config["data_cache_dir"] = str(Path(".tradingagents/cache").resolve())
    config["results_dir"] = settings["results_dir"]
    graph = TradingAgentsGraph(debug=False, config=config)

    results = []
    for symbol in settings["symbols"]:
        _, signal = graph.propagate(symbol, args.date)
        results.append({"symbol": symbol, "signal": signal, "trade_amount_usd": settings["trade_amount_usd"]})
        print(f"{symbol}: {signal}")

    output = Path(settings["results_dir"]) / f"scan_{args.date}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"date": args.date, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print("Research only: review the report and approve any Robinhood order manually.")


if __name__ == "__main__":
    main()
