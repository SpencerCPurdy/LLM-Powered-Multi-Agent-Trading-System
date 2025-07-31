---
title: LLM-Powered Multi-Agent Trading System
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
license: mit
short_description: AI trading system with live LLM agents & a DQN coordinator
---

# LLM-Powered Multi-Agent Trading System

This project is a sophisticated simulation of a multi-agent algorithmic trading system. It demonstrates how a team of specialized, AI-powered agents can collaborate to analyze market conditions and make trading decisions. The system is architected around a central Deep Q-Network (DQN) coordinator that learns to weigh the signals from each agent, optimizing the overall strategy through reinforcement learning.

This is a **simulation environment** created for demonstration and educational purposes. It operates on synthetically generated market data and does not connect to live exchanges or execute real trades.

## Core Features

The system is composed of four distinct, specialized agents whose signals are integrated by a master reinforcement learning agent.

* **Fundamental Analyst Agent**: Leverages a Large Language Model (OpenAI's GPT-3.5-turbo) to perform genuine market analysis. It synthesizes a comprehensive report of price action, technical levels, and volume trends to generate a trading recommendation with detailed, human-like reasoning. If an API key is not provided, it uses a rule-based fallback.

* **Technical Analyst Agent**: Employs a custom-built Transformer neural network to identify complex patterns in price and indicator data. It complements this deep learning model with a suite of classical technical indicators (RSI, MACD, Bollinger Bands) to form its recommendation.

* **Sentiment Analyst Agent**: Performs real-time sentiment analysis on financial news. It uses the `ProsusAI/finbert` model, which is specifically fine-tuned for financial text. If a NewsAPI key is provided, it fetches real-world news headlines; otherwise, it operates on a set of contextual, pre-defined headlines.

* **Risk Manager Agent**: Provides a continuous, quantitative assessment of portfolio and market risk. It calculates critical metrics such as Value at Risk (VaR), Sharpe Ratio, and maximum drawdown. It also uses the Kelly Criterion to suggest an optimal position size, ensuring that all trading decisions are made within a robust risk framework.

* **DQN Learning Coordinator**: A Deep Q-Network (DQN) acts as the central decision-maker. It takes the signals and metadata from all four agents as input and learns an optimal trading policy over time through reinforcement learning. It is trained to maximize returns while managing risk by learning how to best interpret the collective intelligence of the agent team.

## How It Works

The simulation follows a structured, event-driven process that mirrors a real-world quantitative trading system:

1.  **Market Simulation**: The system first generates a realistic, synthetic time series of market data with distinct regimes (e.g., bull market, bear market, high volatility).
2.  **Agent Analysis**: In each step of the simulation, the historical market data and current portfolio state are passed to each of the four agents. Each agent performs its specialized analysis and produces a `MarketSignal` containing a recommendation (buy, sell, or hold), a confidence score, and detailed reasoning.
3.  **State Vector Creation**: The signals from all agents, along with key market and portfolio metrics, are compiled into a numerical state vector.
4.  **DQN Decision**: This state vector is fed into the DQN Coordinator, which determines the final trading action (buy, sell, or hold).
5.  **Trade Execution**: The system simulates the execution of the trade, updating the portfolio's cash and positions. Position sizing is influenced by the Risk Manager's Kelly Criterion calculation and the overall confidence of the agent signals.
6.  **Learning and Adaptation**: A reward is calculated based on the change in portfolio value. The RL agent stores the state, action, and reward in its memory and periodically replays these experiences to train its Q-network, continuously improving its decision-making policy.
7.  **Visualization**: The entire process, including portfolio performance, allocation, agent signal distribution, and drawdown, is visualized in a comprehensive dashboard.

## Technical Stack

* **Machine Learning / AI**: PyTorch, OpenAI, Transformers, Gymnasium (for RL environment structure)
* **Data Science & Quantitative Analysis**: NumPy, Pandas, SciPy, StatsModels, scikit-learn
* **Technical Analysis**: `ta`
* **Web Interface & Dashboard**: Gradio
* **Visualization**: Plotly
* **News Data**: NewsAPI

## How to Use the Demo

1.  Adjust the **Simulation Parameters** on the left, such as initial capital, market volatility, and the number of days to simulate.
2.  (Optional but Recommended) Enter your **API keys** for OpenAI and NewsAPI to enable the full capabilities of the Fundamental and Sentiment agents. The system will function without them by using its built-in fallback logic.
3.  Click the **Run Simulation** button. The simulation may take a moment to complete.
4.  Analyze the results in the **Performance Analysis Dashboard** and the summary tables. The dashboard provides a visual breakdown of the portfolio's performance, while the tables offer detailed metrics and a log of recent trades.

## Disclaimer

This project is a simulation for demonstration purposes only and does not constitute financial advice. All market data is synthetically generated, and the strategies shown are not intended for live trading. Trading in financial markets involves significant risk.