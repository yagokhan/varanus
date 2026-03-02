# Awesome AI in Finance — Reference

> Saved from [georgezouq/awesome-ai-in-finance](https://github.com/georgezouq/awesome-ai-in-finance) for future reference.

## Why This Is Here

During the Varanus v4.0 walk-forward validation review, several resources from this curated list were identified as relevant for improving the strategy's validation pipeline, data sourcing, and backtesting framework.

---

## Key Resources for Varanus

### Walk-Forward & Backtesting Frameworks
- [backtrader](https://github.com/backtrader/backtrader) — Mature walk-forward optimization framework with proper temporal splitting.
- [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library) — DRL-based trading library with rolling-window retraining across multiple assets.
- [QuantResearch](https://github.com/letianzj/QuantResearch) — Practical quant backtesting examples with proper train/test temporal separation.
- [zipline](https://github.com/quantopian/zipline) — Python algorithmic trading library.
- [lean](https://github.com/QuantConnect/Lean) — Algorithmic trading engine for strategy research, backtesting and live trading.

### Books & Courses
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos-ebook/dp/B079KLDW21/) — **Chapter 7: Cross-Validation in Finance** covers Combinatorial Purged Cross-Validation (CPCV), the gold standard for time-series model validation. Directly addresses the fold-overlap problem found in our WFV.
- [Advanced-Deep-Trading](https://github.com/Rachnog/Advanced-Deep-Trading) — Experiments based on the above book.

### Crypto Data Sources
- [CryptoInscriber](https://github.com/Optixal/CryptoInscriber) — Live crypto historical trade data blotter. Useful for extending Varanus data history beyond current cache.
- [Gekko-Datasets](https://github.com/xFFFFF/Gekko-Datasets) — Historical crypto data dumps in SQLite format.

### Risk & Performance Analytics
- [pyfolio](https://github.com/quantopian/pyfolio) — Portfolio and risk analytics in Python.
- [empyrical](https://github.com/quantopian/empyrical) — Common financial risk and performance metrics.
- [alphalens](https://github.com/quantopian/alphalens) — Performance analysis of predictive (alpha) stock factors.

### Crypto Trading Strategies (Reference)
- [LSTM-Crypto-Price-Prediction](https://github.com/SC4RECOIN/LSTM-Crypto-Price-Prediction) — LSTM-RNN for crypto price trend prediction.
- [catalyst](https://github.com/enigmampc/catalyst) — Algorithmic trading library for crypto assets.

### AI Agents & LLMs in Finance
- [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund) — Explore AI for trading decisions.
- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) — Open-source AI agent platform for financial analysis.
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) — LLMs and NLP in Finance playground.

### Research Tools
- [TensorTrade](https://github.com/tensortrade-org/tensortrade) — Trade efficiently with reinforcement learning.
- [OpenBB](https://github.com/OpenBB-finance/OpenBB) — AI-powered open-source research and analytics workspace.

---

## Full Source

The complete curated list is maintained at:
**https://github.com/georgezouq/awesome-ai-in-finance**