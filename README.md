# IBKR AI Trading Bot

A Python-based trading bot that connects directly to Interactive Brokers (IBKR) for live market data and order execution, with optional OpenAI integration for advanced signal processing.

## Features

- **IBKR Connectivity:** Real-time data and order management via ib_insync
- **Entry Filters:** VWAP alignment, RSI thresholds, MACD crossover, volume-spike detection
- **Position Sizing:** 5% of account equity per trade
- **Exits:** Tiered profit targets (25%, 75%, 150%) with partial exits
- **Risk Management:** Configurable stop-loss (default 2%)
- **OpenAI Integration (Optional):** AI-driven signal evaluation and dynamic filter adjustment
