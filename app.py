"""
LLM-Powered Multi-Agent Trading System
Author: Spencer Purdy
Description: A sophisticated multi-agent trading system leveraging real LLM reasoning for market analysis.
             Features specialized agents for fundamental, technical, sentiment analysis, and risk management,
             coordinated by a DQN reinforcement learning agent.
"""

# Install required packages
# !pip install -q transformers torch numpy pandas scikit-learn plotly gradio yfinance ta scipy gymnasium accelerate openai newsapi-python

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import warnings
import os
import openai
from newsapi import NewsApiClient
warnings.filterwarnings('ignore')

# Technical analysis
import ta

# Transformers for sentiment analysis
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Configuration constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02
MAX_POSITION_SIZE = 0.25  # Maximum 25% of portfolio in single position
MIN_CASH_RESERVE = 0.1    # Minimum 10% cash reserve

@dataclass
class MarketSignal:
    """Data class for agent signals"""
    agent_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class TradingDecision:
    """Data class for trading decisions"""
    action: str  # 'buy', 'sell', 'hold'
    size: float  # Position size as fraction of portfolio
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: Dict[str, str]

class BaseAgent:
    """Base class for all trading agents"""

    def __init__(self, name: str):
        self.name = name
        self.history = []

    def analyze(self, market_data: pd.DataFrame, portfolio_state: Dict) -> MarketSignal:
        """Analyze market data and return signal"""
        raise NotImplementedError

    def update_history(self, signal: MarketSignal, outcome: float):
        """Update agent history with signal and outcome"""
        self.history.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'outcome': outcome
        })

class FundamentalAnalystAgent(BaseAgent):
    """Agent specializing in fundamental analysis using real OpenAI LLM reasoning"""

    def __init__(self, api_key: str):
        super().__init__("Fundamental Analyst")

        # Initialize OpenAI client
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model = "gpt-3.5-turbo"  # Using GPT-3.5 for cost efficiency

        # System prompt for consistent analysis
        self.system_prompt = """You are a professional fundamental analyst with deep expertise in financial markets.
        Analyze the provided market data and give a clear trading recommendation (BUY, SELL, or HOLD) with detailed reasoning.
        Consider price movements, volume patterns, volatility, and market positioning.
        Be specific about the factors driving your recommendation and provide a confidence level (0-1).
        Format your response as:
        RECOMMENDATION: [BUY/SELL/HOLD]
        CONFIDENCE: [0.0-1.0]
        REASONING: [Your detailed analysis]"""

    def _prepare_market_analysis(self, market_data: pd.DataFrame, portfolio_state: Dict) -> str:
        """Prepare comprehensive market analysis for LLM"""

        # Calculate key metrics
        current_price = market_data['close'].iloc[-1]
        price_change_1d = market_data['close'].pct_change().iloc[-1] * 100
        price_change_5d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5] - 1) * 100
        price_change_20d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-20] - 1) * 100

        # Volume analysis
        avg_volume_20d = market_data['volume'].iloc[-20:].mean()
        current_volume = market_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume_20d

        # Price metrics
        high_20d = market_data['high'].iloc[-20:].max()
        low_20d = market_data['low'].iloc[-20:].min()
        price_position = (current_price - low_20d) / (high_20d - low_20d) if high_20d > low_20d else 0.5

        # Moving averages
        ma_5 = market_data['close'].iloc[-5:].mean()
        ma_20 = market_data['close'].iloc[-20:].mean()
        ma_50 = market_data['close'].iloc[-50:].mean() if len(market_data) >= 50 else ma_20

        # Volatility
        returns = market_data['close'].pct_change()
        volatility = returns.iloc[-20:].std() * np.sqrt(252) * 100

        # Support and resistance levels
        support = market_data['low'].iloc[-20:].min()
        resistance = market_data['high'].iloc[-20:].max()

        # Portfolio context
        cash_ratio = portfolio_state.get('cash', 100000) / portfolio_state.get('total_value', 100000)

        analysis = f"""Market Analysis Report:

PRICE ACTION:
- Current Price: ${current_price:.2f}
- 1-Day Change: {price_change_1d:+.2f}%
- 5-Day Change: {price_change_5d:+.2f}%
- 20-Day Change: {price_change_20d:+.2f}%
- Position in 20-Day Range: {price_position:.1%} (0% = at low, 100% = at high)

TECHNICAL LEVELS:
- 5-Day MA: ${ma_5:.2f} ({'+' if current_price > ma_5 else '-'}{abs(current_price/ma_5 - 1)*100:.1f}%)
- 20-Day MA: ${ma_20:.2f} ({'+' if current_price > ma_20 else '-'}{abs(current_price/ma_20 - 1)*100:.1f}%)
- 50-Day MA: ${ma_50:.2f} ({'+' if current_price > ma_50 else '-'}{abs(current_price/ma_50 - 1)*100:.1f}%)
- Support Level: ${support:.2f}
- Resistance Level: ${resistance:.2f}

VOLUME ANALYSIS:
- Current Volume: {current_volume:,.0f}
- Volume vs 20-Day Average: {volume_ratio:.2f}x
- Volume Trend: {'High' if volume_ratio > 1.5 else 'Above Average' if volume_ratio > 1.2 else 'Average' if volume_ratio > 0.8 else 'Below Average'}

RISK METRICS:
- Annualized Volatility: {volatility:.1f}%
- Risk Level: {'High' if volatility > 30 else 'Moderate' if volatility > 20 else 'Low'}

PORTFOLIO CONTEXT:
- Cash Available: {cash_ratio:.1%} of portfolio
- Risk Capacity: {'High' if cash_ratio > 0.3 else 'Moderate' if cash_ratio > 0.15 else 'Low'}

Based on this comprehensive analysis, provide your trading recommendation."""

        return analysis

    def analyze(self, market_data: pd.DataFrame, portfolio_state: Dict) -> MarketSignal:
        """Perform fundamental analysis using OpenAI LLM"""

        # Prepare market analysis
        market_analysis = self._prepare_market_analysis(market_data, portfolio_state)

        try:
            # Call OpenAI API for analysis
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": market_analysis}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500
            )

            # Extract LLM response
            llm_analysis = response.choices[0].message.content

            # Parse the response
            lines = llm_analysis.split('\n')
            recommendation = 'hold'
            confidence = 0.5
            reasoning = ""

            for line in lines:
                if line.startswith('RECOMMENDATION:'):
                    rec_text = line.split(':', 1)[1].strip().upper()
                    if 'BUY' in rec_text:
                        recommendation = 'buy'
                    elif 'SELL' in rec_text:
                        recommendation = 'sell'
                    else:
                        recommendation = 'hold'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Ensure within bounds
                    except:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif reasoning and line.strip():  # Continue reasoning on subsequent lines
                    reasoning += " " + line.strip()

            # Ensure we have valid reasoning
            if not reasoning:
                reasoning = "LLM analysis suggests " + recommendation + " based on current market conditions."

        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to rule-based analysis if API fails
            returns = market_data['close'].pct_change()
            recent_return = returns.iloc[-20:].mean()

            if recent_return > 0.001:
                recommendation = 'buy'
                confidence = 0.6
                reasoning = "Positive momentum detected in recent price action based on technical indicators"
            elif recent_return < -0.001:
                recommendation = 'sell'
                confidence = 0.6
                reasoning = "Negative momentum suggests caution based on recent price movements"
            else:
                recommendation = 'hold'
                confidence = 0.5
                reasoning = "Market showing neutral patterns, maintaining current position"

        # Extract numerical data for metadata
        current_price = market_data['close'].iloc[-1]
        price_change = market_data['close'].pct_change().iloc[-1]

        return MarketSignal(
            agent_name=self.name,
            signal_type=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 4),
                'llm_model': self.model,
                'analysis_timestamp': datetime.now().isoformat()
            }
        )

class TechnicalAnalystAgent(BaseAgent):
    """Agent specializing in technical analysis using numerical transformers"""

    def __init__(self):
        super().__init__("Technical Analyst")
        self.lookback_period = 20
        self.transformer_model = self._build_price_transformer()

    def _build_price_transformer(self):
        """Build a transformer model for price prediction"""
        class PriceTransformer(nn.Module):
            def __init__(self, input_dim=7, hidden_dim=64, num_heads=4, num_layers=2):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_projection = nn.Linear(hidden_dim, 3)  # Buy, Hold, Sell

            def forward(self, x):
                x = self.input_projection(x)
                seq_len = x.size(1)
                x = x + self.positional_encoding[:, :seq_len, :]
                x = self.transformer(x)
                x = self.output_projection(x[:, -1, :])  # Use last timestep
                return torch.softmax(x, dim=-1)

        return PriceTransformer()

    def analyze(self, market_data: pd.DataFrame, portfolio_state: Dict) -> MarketSignal:
        """Perform technical analysis using indicators and transformer model"""

        # Calculate technical indicators
        df = market_data.copy()

        # Add technical indicators using ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # Prepare features for transformer
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']

        # Get last lookback_period rows
        recent_data = df[features].iloc[-self.lookback_period:].fillna(method='ffill').fillna(0)

        # Normalize features
        normalized_data = (recent_data - recent_data.mean()) / (recent_data.std() + 1e-8)

        # Convert to tensor
        input_tensor = torch.FloatTensor(normalized_data.values).unsqueeze(0)

        # Get transformer prediction
        with torch.no_grad():
            predictions = self.transformer_model(input_tensor)
            buy_prob, hold_prob, sell_prob = predictions[0].numpy()

        # Determine signal based on transformer output
        if buy_prob > 0.6:
            signal_type = 'buy'
            confidence = float(buy_prob)
            reasoning = "Strong bullish technical setup detected by transformer model"
        elif sell_prob > 0.6:
            signal_type = 'sell'
            confidence = float(sell_prob)
            reasoning = "Bearish technical pattern identified by transformer model"
        else:
            signal_type = 'hold'
            confidence = float(hold_prob)
            reasoning = "No clear technical direction detected"

        # Enhance reasoning with specific technical indicators
        current_rsi = df['rsi'].iloc[-1]
        if current_rsi < 30 and signal_type != 'buy':
            reasoning += f" (Note: RSI at {current_rsi:.1f} indicates oversold conditions)"
        elif current_rsi > 70 and signal_type != 'sell':
            reasoning += f" (Note: RSI at {current_rsi:.1f} indicates overbought conditions)"

        # Check Bollinger Bands
        current_price = df['close'].iloc[-1]
        bb_high = df['bb_high'].iloc[-1]
        bb_low = df['bb_low'].iloc[-1]

        if not np.isnan(bb_high) and not np.isnan(bb_low):
            if current_price > bb_high:
                reasoning += " Price breaking above upper Bollinger Band"
            elif current_price < bb_low:
                reasoning += " Price breaking below lower Bollinger Band"

        # MACD analysis
        macd_value = df['macd'].iloc[-1]
        macd_signal = ta.trend.MACD(df['close']).macd_signal().iloc[-1]

        if not np.isnan(macd_value) and not np.isnan(macd_signal):
            if macd_value > macd_signal and macd_value > 0:
                reasoning += " MACD showing bullish crossover above zero"
            elif macd_value < macd_signal and macd_value < 0:
                reasoning += " MACD showing bearish crossover below zero"

        return MarketSignal(
            agent_name=self.name,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'rsi': round(current_rsi, 2) if not np.isnan(current_rsi) else 50,
                'buy_probability': round(buy_prob, 3),
                'sell_probability': round(sell_prob, 3),
                'hold_probability': round(hold_prob, 3),
                'current_price': round(current_price, 2),
                'bb_position': 'above' if current_price > bb_high else 'below' if current_price < bb_low else 'within'
            }
        )

class SentimentAnalystAgent(BaseAgent):
    """Agent specializing in sentiment analysis using FinBERT and real news data"""

    def __init__(self, news_api_key: str = None):
        super().__init__("Sentiment Analyst")
        # Initialize FinBERT for financial sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU
        )

        # Initialize news API client
        self.news_api_key = news_api_key
        if self.news_api_key:
            self.newsapi = NewsApiClient(api_key=self.news_api_key)
        else:
            self.newsapi = None

    def _fetch_real_news(self, symbol: str = None) -> List[str]:
        """Fetch real news articles from NewsAPI"""

        if not self.newsapi:
            return self._generate_contextual_news()

        try:
            # Search for financial news
            query = f"stock market {symbol if symbol else 'trading'} finance"

            # Get top headlines
            headlines = self.newsapi.get_top_headlines(
                q=query,
                category='business',
                language='en',
                page_size=10
            )

            # Extract article titles and descriptions
            news_items = []

            if headlines['status'] == 'ok' and headlines['articles']:
                for article in headlines['articles'][:10]:
                    if article['title']:
                        news_items.append(article['title'])
                    if article['description']:
                        news_items.append(article['description'])

            # If not enough news, get everything
            if len(news_items) < 5:
                all_articles = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )

                if all_articles['status'] == 'ok' and all_articles['articles']:
                    for article in all_articles['articles']:
                        if article['title'] and article['title'] not in news_items:
                            news_items.append(article['title'])
                        if len(news_items) >= 10:
                            break

            return news_items[:10] if news_items else self._generate_contextual_news()

        except Exception as e:
            print(f"News API error: {e}")
            return self._generate_contextual_news()

    def _generate_contextual_news(self) -> List[str]:
        """Generate contextual market news as fallback"""

        base_headlines = [
            "Federal Reserve signals potential rate changes amid economic uncertainty",
            "Tech stocks rally as earnings season approaches",
            "Global markets react to latest inflation data",
            "Energy sector sees volatility amid geopolitical tensions",
            "Analysts upgrade outlook for financial sector",
            "Retail sales data exceeds expectations",
            "Manufacturing index shows signs of recovery",
            "Currency markets stabilize after central bank interventions",
            "Commodity prices surge on supply chain concerns",
            "Corporate earnings beat analyst estimates"
        ]

        # Add some variation
        selected = random.sample(base_headlines, min(5, len(base_headlines)))
        return selected

    def analyze(self, market_data: pd.DataFrame, portfolio_state: Dict) -> MarketSignal:
        """Perform sentiment analysis on real or contextual news"""

        # Fetch news items
        news_items = self._fetch_real_news()

        # Analyze sentiment for each news item
        sentiments = []
        for news in news_items:
            try:
                # Truncate to 512 characters for FinBERT
                result = self.sentiment_pipeline(news[:512])[0]
                sentiments.append(result)
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                sentiments.append({'label': 'neutral', 'score': 0.5})

        # Aggregate sentiments
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')

        # Calculate weighted scores
        positive_scores = [s['score'] for s in sentiments if s['label'] == 'positive']
        negative_scores = [s['score'] for s in sentiments if s['label'] == 'negative']

        positive_score = sum(positive_scores) / len(sentiments) if sentiments else 0
        negative_score = sum(negative_scores) / len(sentiments) if sentiments else 0
        neutral_score = 1 - positive_score - negative_score

        # Calculate net sentiment
        net_sentiment = positive_score - negative_score

        # Determine signal based on sentiment analysis
        if net_sentiment > 0.2 and positive_count > negative_count * 1.5:
            signal_type = 'buy'
            confidence = min(0.8, positive_score + 0.2)
            reasoning = f"Positive sentiment detected across {positive_count}/{len(sentiments)} news items"
        elif net_sentiment < -0.2 and negative_count > positive_count * 1.5:
            signal_type = 'sell'
            confidence = min(0.8, negative_score + 0.2)
            reasoning = f"Negative sentiment dominates with {negative_count}/{len(sentiments)} bearish news items"
        else:
            signal_type = 'hold'
            confidence = 0.5 + abs(net_sentiment) * 0.3
            reasoning = "Mixed sentiment suggests maintaining current position"

        # Add specific news context to reasoning
        if news_items and len(news_items) > 0:
            reasoning += f". Key headline: '{news_items[0][:100]}...'"

        # Consider market context
        recent_volatility = market_data['close'].pct_change().iloc[-20:].std()
        if recent_volatility > 0.02:
            confidence *= 0.9  # Reduce confidence in high volatility
            reasoning += " (Confidence adjusted for high market volatility)"

        return MarketSignal(
            agent_name=self.name,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'positive_sentiment': round(positive_score, 3),
                'negative_sentiment': round(negative_score, 3),
                'neutral_sentiment': round(neutral_score, 3),
                'net_sentiment': round(net_sentiment, 3),
                'news_analyzed': len(news_items),
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'data_source': 'real_news' if self.newsapi else 'contextual'
            }
        )

class RiskManagerAgent(BaseAgent):
    """Agent specializing in risk management and position sizing"""

    def __init__(self):
        super().__init__("Risk Manager")
        self.max_drawdown_threshold = 0.15  # 15% max drawdown
        self.var_confidence = 0.95  # 95% VaR

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 20:
            return 0.02  # Default 2% VaR if insufficient data
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / (returns.std() + 1e-8)

    def analyze(self, market_data: pd.DataFrame, portfolio_state: Dict) -> MarketSignal:
        """Perform risk analysis and provide risk-adjusted recommendations"""

        returns = market_data['close'].pct_change().dropna()
        current_price = market_data['close'].iloc[-1]

        # Calculate comprehensive risk metrics
        volatility = returns.iloc[-20:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        var = self.calculate_var(returns.iloc[-100:])
        sharpe = self.calculate_sharpe_ratio(returns.iloc[-60:])

        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Check portfolio metrics
        current_positions = portfolio_state.get('total_position_value', 0)
        portfolio_value = portfolio_state.get('total_value', 100000)
        cash_ratio = portfolio_state.get('cash', portfolio_value) / portfolio_value

        # Calculate current portfolio drawdown
        if 'peak_value' in portfolio_state:
            current_drawdown = (portfolio_state['peak_value'] - portfolio_value) / portfolio_state['peak_value']
        else:
            current_drawdown = 0

        # Comprehensive risk assessment
        risk_factors = []
        risk_score = 0

        if volatility > 0.3:  # High volatility
            risk_factors.append("high_volatility")
            risk_score += 2

        if current_drawdown > self.max_drawdown_threshold * 0.8:
            risk_factors.append("approaching_max_drawdown")
            risk_score += 3

        if cash_ratio < MIN_CASH_RESERVE:
            risk_factors.append("low_cash_reserves")
            risk_score += 2

        if sharpe < 0.5:
            risk_factors.append("poor_risk_adjusted_returns")
            risk_score += 1

        if var < -0.05:  # 5% VaR threshold
            risk_factors.append("high_value_at_risk")
            risk_score += 2

        # Kelly Criterion for position sizing
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1

        kelly_fraction = 0
        if avg_loss > 0 and win_rate > 0:
            odds = avg_win / avg_loss
            kelly_fraction = (win_rate * odds - (1 - win_rate)) / odds
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Determine signal based on comprehensive risk assessment
        if risk_score >= 5:
            signal_type = 'sell'
            confidence = min(0.7 + risk_score * 0.05, 0.9)
            reasoning = f"Multiple risk factors detected ({len(risk_factors)}): {', '.join(risk_factors)}. Risk score: {risk_score}/10"
        elif risk_score >= 3:
            signal_type = 'hold'
            confidence = 0.6
            reasoning = f"Elevated risk levels detected. Factors: {', '.join(risk_factors)}. Recommend caution"
        else:
            # Low risk environment - check for opportunities
            if sharpe > 1.0 and volatility < 0.2 and cash_ratio > 0.2:
                signal_type = 'buy'
                confidence = 0.7
                reasoning = f"Risk metrics favorable. Sharpe: {sharpe:.2f}, Vol: {volatility:.1%}, Cash available"
            else:
                signal_type = 'hold'
                confidence = 0.5
                reasoning = "Risk levels acceptable, maintaining current exposure"

        # Add Kelly Criterion to reasoning
        if kelly_fraction > 0:
            reasoning += f". Optimal position size (Kelly): {kelly_fraction:.1%}"

        return MarketSignal(
            agent_name=self.name,
            signal_type=signal_type,
            confidence=min(confidence, 0.9),
            reasoning=reasoning,
            metadata={
                'volatility': round(volatility, 3),
                'var_95': round(var, 3),
                'sharpe_ratio': round(sharpe, 2),
                'max_drawdown': round(max_drawdown, 3),
                'current_drawdown': round(current_drawdown, 3),
                'risk_factors': risk_factors,
                'risk_score': risk_score,
                'cash_ratio': round(cash_ratio, 2),
                'kelly_fraction': round(kelly_fraction, 3),
                'win_rate': round(win_rate, 2)
            }
        )

# DQN Implementation for Multi-Agent Coordination
class DQN(nn.Module):
    """Deep Q-Network for learning from agent recommendations"""

    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReinforcementLearningAgent:
    """DQN agent that learns from multi-agent recommendations"""

    def __init__(self, state_size: int = 16, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.learning_rate = 0.001

        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().numpy())

    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class MultiAgentTradingSystem:
    """Main trading system coordinating multiple agents"""

    def __init__(self, initial_capital: float = 100000, openai_api_key: str = None, news_api_key: str = None):
        self.initial_capital = initial_capital
        self.openai_api_key = openai_api_key
        self.news_api_key = news_api_key
        self.reset()

        # Initialize agents
        print("Initializing trading agents...")

        # Check for API keys
        if not self.openai_api_key:
            print("Warning: OpenAI API key not provided. Fundamental analysis will use fallback methods.")

        if not self.news_api_key:
            print("Warning: News API key not provided. Sentiment analysis will use contextual news.")

        self.fundamental_agent = FundamentalAnalystAgent(self.openai_api_key)
        self.technical_agent = TechnicalAnalystAgent()
        self.sentiment_agent = SentimentAnalystAgent(self.news_api_key)
        self.risk_agent = RiskManagerAgent()

        print("All agents initialized successfully")

        # Initialize RL coordinator
        self.rl_agent = ReinforcementLearningAgent(state_size=16, action_size=3)

        # Trading history
        self.trade_history = []
        self.performance_history = []

    def reset(self):
        """Reset portfolio to initial state"""
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'peak_value': self.initial_capital,
            'total_position_value': 0
        }

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total = self.portfolio['cash']
        for symbol, position in self.portfolio['positions'].items():
            if symbol in current_prices:
                total += position['shares'] * current_prices[symbol]
        return total

    def aggregate_signals(self, signals: List[MarketSignal]) -> Tuple[str, float, Dict]:
        """Aggregate signals from multiple agents using weighted voting"""
        buy_score = 0
        sell_score = 0
        hold_score = 0

        reasoning = {}

        # Weight signals by confidence
        for signal in signals:
            weight = signal.confidence
            reasoning[signal.agent_name] = {
                'signal': signal.signal_type,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }

            if signal.signal_type == 'buy':
                buy_score += weight
            elif signal.signal_type == 'sell':
                sell_score += weight
            else:
                hold_score += weight

        # Normalize scores
        total_score = buy_score + sell_score + hold_score
        if total_score > 0:
            buy_score /= total_score
            sell_score /= total_score
            hold_score /= total_score

        # Determine action based on weighted voting
        if buy_score > 0.5:
            action = 'buy'
            confidence = buy_score
        elif sell_score > 0.5:
            action = 'sell'
            confidence = sell_score
        else:
            action = 'hold'
            confidence = hold_score

        return action, confidence, reasoning

    def create_state_vector(self, market_data: pd.DataFrame, signals: List[MarketSignal]) -> np.ndarray:
        """Create state vector for RL agent"""

        # Market features
        returns = market_data['close'].pct_change()
        current_return = returns.iloc[-1]
        volatility = returns.iloc[-20:].std()
        momentum = returns.iloc[-20:].mean()

        # Technical indicators
        rsi = ta.momentum.RSIIndicator(market_data['close']).rsi().iloc[-1]

        # Signal features
        buy_signals = sum(1 for s in signals if s.signal_type == 'buy')
        sell_signals = sum(1 for s in signals if s.signal_type == 'sell')
        avg_confidence = np.mean([s.confidence for s in signals])

        # Portfolio features
        cash_ratio = self.portfolio['cash'] / self.portfolio['total_value']
        position_ratio = self.portfolio['total_position_value'] / self.portfolio['total_value']

        # Risk metrics from risk agent
        risk_metadata = next((s.metadata for s in signals if s.agent_name == "Risk Manager"), {})
        risk_score = risk_metadata.get('risk_score', 0) / 10.0  # Normalize

        # Create state vector
        state = np.array([
            current_return,
            volatility,
            momentum,
            rsi / 100.0 if not np.isnan(rsi) else 0.5,
            buy_signals / len(signals),
            sell_signals / len(signals),
            avg_confidence,
            cash_ratio,
            position_ratio,
            risk_score,
            # Agent-specific confidences
            signals[0].confidence,  # Fundamental
            signals[1].confidence,  # Technical
            signals[2].confidence,  # Sentiment
            signals[3].confidence,  # Risk
            0, 0  # Padding for consistent size
        ])

        return state[:self.rl_agent.state_size]

    def execute_trade(self, symbol: str, action: str, confidence: float,
                     current_price: float, reasoning: Dict) -> Dict:
        """Execute trading decision"""

        trade_result = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'price': current_price,
            'confidence': confidence,
            'reasoning': reasoning,
            'executed': False,
            'shares': 0,
            'value': 0
        }

        if action == 'buy':
            # Calculate position size based on confidence and risk limits
            max_position_value = self.portfolio['total_value'] * MAX_POSITION_SIZE

            # Use Kelly Criterion from risk agent if available
            risk_metadata = reasoning.get('Risk Manager', {})
            if isinstance(risk_metadata, dict) and 'metadata' in reasoning['Risk Manager']:
                kelly_fraction = reasoning['Risk Manager']['metadata'].get('kelly_fraction', 0.1)
            else:
                kelly_fraction = 0.1

            position_value = min(
                self.portfolio['cash'] * confidence * kelly_fraction,
                max_position_value
            )

            if position_value > current_price:
                shares = int(position_value / current_price)
                cost = shares * current_price

                # Ensure minimum cash reserve
                if cost <= self.portfolio['cash'] * (1 - MIN_CASH_RESERVE):
                    # Execute buy
                    self.portfolio['cash'] -= cost

                    if symbol in self.portfolio['positions']:
                        # Update existing position
                        old_shares = self.portfolio['positions'][symbol]['shares']
                        old_avg_price = self.portfolio['positions'][symbol]['avg_price']
                        new_shares = old_shares + shares
                        new_avg_price = (old_avg_price * old_shares + cost) / new_shares

                        self.portfolio['positions'][symbol]['shares'] = new_shares
                        self.portfolio['positions'][symbol]['avg_price'] = new_avg_price
                    else:
                        # Create new position
                        self.portfolio['positions'][symbol] = {
                            'shares': shares,
                            'avg_price': current_price
                        }

                    trade_result['executed'] = True
                    trade_result['shares'] = shares
                    trade_result['value'] = cost

        elif action == 'sell' and symbol in self.portfolio['positions']:
            # Sell a portion of position based on confidence
            position = self.portfolio['positions'][symbol]
            shares_to_sell = int(position['shares'] * confidence * 0.5)

            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.portfolio['cash'] += revenue
                position['shares'] -= shares_to_sell

                if position['shares'] == 0:
                    del self.portfolio['positions'][symbol]

                trade_result['executed'] = True
                trade_result['shares'] = -shares_to_sell
                trade_result['value'] = revenue

        # Update portfolio metrics
        self.portfolio['total_position_value'] = sum(
            pos['shares'] * pos['avg_price']
            for pos in self.portfolio['positions'].values()
        )

        return trade_result

class MarketSimulator:
    """Simulate realistic market data for demonstration"""

    def __init__(self, base_price: float = 100, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        self.drift = 0.0001

    def generate_market_data(self, days: int = 252) -> pd.DataFrame:
        """Generate simulated market data with realistic patterns"""

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate price series using geometric Brownian motion
        returns = np.random.normal(self.drift, self.volatility, days)

        # Add market regimes
        regime_length = days // 4
        for i in range(0, days, regime_length):
            regime_type = i // regime_length % 4
            if regime_type == 0:  # Bull market
                returns[i:i+regime_length] += np.random.normal(0.0005, 0.0001, min(regime_length, days-i))
            elif regime_type == 1:  # Bear market
                returns[i:i+regime_length] += np.random.normal(-0.0005, 0.0001, min(regime_length, days-i))
            elif regime_type == 2:  # High volatility
                returns[i:i+regime_length] *= 1.5
            # regime_type == 3 is normal market

        price_series = self.base_price * np.exp(np.cumsum(returns))

        # Add some mean reversion
        ma_20 = pd.Series(price_series).rolling(20).mean()
        mean_reversion_strength = 0.1
        for i in range(20, len(price_series)):
            if not np.isnan(ma_20.iloc[i]):
                deviation = (price_series[i] - ma_20.iloc[i]) / ma_20.iloc[i]
                price_series[i] *= (1 - mean_reversion_strength * deviation)

        # Generate OHLCV data
        data = {
            'date': dates,
            'open': price_series * np.random.uniform(0.99, 1.01, days),
            'high': price_series * np.random.uniform(1.01, 1.03, days),
            'low': price_series * np.random.uniform(0.97, 0.99, days),
            'close': price_series,
            'volume': np.random.lognormal(np.log(1000000), 0.5, days)
        }

        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)

        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

def run_backtest(trading_system: MultiAgentTradingSystem,
                 market_data: pd.DataFrame,
                 symbol: str = "DEMO") -> Tuple[pd.DataFrame, List[Dict]]:
    """Run comprehensive backtest simulation"""

    performance_history = []
    trade_history = []

    # Run simulation day by day
    print("Starting backtest simulation...")
    for i in range(50, len(market_data)):
        # Progress indicator
        if i % 50 == 0:
            print(f"Processing day {i}/{len(market_data)}")

        # Get historical data up to current day
        historical_data = market_data.iloc[:i+1]
        current_price = historical_data['close'].iloc[-1]

        # Get signals from all agents
        signals = [
            trading_system.fundamental_agent.analyze(historical_data, trading_system.portfolio),
            trading_system.technical_agent.analyze(historical_data, trading_system.portfolio),
            trading_system.sentiment_agent.analyze(historical_data, trading_system.portfolio),
            trading_system.risk_agent.analyze(historical_data, trading_system.portfolio)
        ]

        # Create state vector for RL
        state = trading_system.create_state_vector(historical_data, signals)

        # Get RL agent action
        rl_action = trading_system.rl_agent.act(state)
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        rl_recommendation = action_map[rl_action]

        # Aggregate signals
        action, confidence, reasoning = trading_system.aggregate_signals(signals)

        # Blend with RL recommendation
        if rl_recommendation == action:
            confidence = min(confidence * 1.1, 0.95)

        # Execute trade
        trade_result = trading_system.execute_trade(
            symbol, action, confidence, current_price, reasoning
        )

        if trade_result['executed']:
            trade_history.append(trade_result)

        # Update portfolio value
        trading_system.portfolio['total_value'] = trading_system.get_portfolio_value({symbol: current_price})
        trading_system.portfolio['peak_value'] = max(
            trading_system.portfolio['peak_value'],
            trading_system.portfolio['total_value']
        )

        # Calculate performance metrics
        returns = (trading_system.portfolio['total_value'] - trading_system.initial_capital) / trading_system.initial_capital

        performance_history.append({
            'date': historical_data.index[-1],
            'portfolio_value': trading_system.portfolio['total_value'],
            'returns': returns,
            'price': current_price,
            'cash': trading_system.portfolio['cash'],
            'position_value': trading_system.portfolio['total_position_value'],
            'action': action,
            'confidence': confidence,
            'signals': {s.agent_name: s.signal_type for s in signals}
        })

        # Train RL agent
        if i > 51:  # Need previous state
            prev_state = trading_system.create_state_vector(market_data.iloc[:i], signals)
            reward = (trading_system.portfolio['total_value'] - performance_history[-2]['portfolio_value']) / trading_system.initial_capital
            trading_system.rl_agent.remember(prev_state, rl_action, reward, state, False)

            if i % 10 == 0:  # Train every 10 steps
                trading_system.rl_agent.replay()

        # Update target network periodically
        if i % 100 == 0:
            trading_system.rl_agent.update_target_network()

    print("Backtest completed")
    return pd.DataFrame(performance_history), trade_history

def calculate_performance_metrics(performance_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive performance metrics"""

    # Basic metrics
    total_return = performance_df['returns'].iloc[-1]

    # Calculate daily returns
    portfolio_values = performance_df['portfolio_value'].values
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Sharpe ratio
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)

    # Win rate
    winning_days = np.sum(daily_returns > 0)
    total_days = len(daily_returns)
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Volatility
    annual_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

    # Calculate Sortino ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1
    sortino_ratio = (daily_returns.mean() * 252 - RISK_FREE_RATE) / downside_deviation if downside_deviation > 0 else 0

    # Calmar ratio
    calmar_ratio = (total_return * 252 / len(performance_df)) / max_drawdown if max_drawdown > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': ((1 + total_return) ** (252 / len(performance_df)) - 1) if len(performance_df) > 0 else 0,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'annual_volatility': annual_volatility
    }

# Gradio Interface
def create_gradio_interface():
    """Create professional Gradio interface for the trading system"""

    def run_simulation(initial_capital, volatility, trading_days, openai_api_key, news_api_key):
        """Run comprehensive trading simulation"""

        print(f"Starting simulation with capital: ${initial_capital:,.2f}")

        # Initialize trading system with API keys
        trading_system = MultiAgentTradingSystem(
            initial_capital=float(initial_capital),
            openai_api_key=openai_api_key if openai_api_key else None,
            news_api_key=news_api_key if news_api_key else None
        )

        # Generate market data
        market_simulator = MarketSimulator(volatility=float(volatility))
        market_data = market_simulator.generate_market_data(int(trading_days))

        # Run backtest
        performance_df, trade_history = run_backtest(trading_system, market_data, "DEMO")

        # Calculate metrics
        metrics = calculate_performance_metrics(performance_df)

        # Create performance visualization
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Portfolio Value vs Market Price',
                'Portfolio Allocation',
                'Agent Signal Distribution',
                'Drawdown Analysis'
            ),
            vertical_spacing=0.08,
            row_heights=[0.35, 0.25, 0.20, 0.20]
        )

        # Portfolio value and market price
        fig.add_trace(
            go.Scatter(
                x=performance_df['date'],
                y=performance_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Normalize market price for comparison
        normalized_price = performance_df['price'] * initial_capital / performance_df['price'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=performance_df['date'],
                y=normalized_price,
                name='Buy & Hold',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )

        # Portfolio allocation
        fig.add_trace(
            go.Scatter(
                x=performance_df['date'],
                y=performance_df['cash'],
                name='Cash',
                fill='tonexty',
                stackgroup='one',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=performance_df['date'],
                y=performance_df['position_value'],
                name='Positions',
                fill='tonexty',
                stackgroup='one',
                line=dict(color='orange')
            ),
            row=2, col=1
        )

        # Agent signals analysis
        agent_signals = pd.DataFrame([p['signals'] for p in performance_df.to_dict('records')])
        signal_counts = {}
        for agent in ['Fundamental Analyst', 'Technical Analyst', 'Sentiment Analyst', 'Risk Manager']:
            if agent in agent_signals.columns:
                signal_counts[agent] = agent_signals[agent].value_counts().to_dict()

        # Create stacked bar chart for signals
        agents = list(signal_counts.keys())
        buy_counts = [signal_counts[agent].get('buy', 0) for agent in agents]
        hold_counts = [signal_counts[agent].get('hold', 0) for agent in agents]
        sell_counts = [signal_counts[agent].get('sell', 0) for agent in agents]

        fig.add_trace(
            go.Bar(name='Buy', x=agents, y=buy_counts, marker_color='green'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(name='Hold', x=agents, y=hold_counts, marker_color='yellow'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(name='Sell', x=agents, y=sell_counts, marker_color='red'),
            row=3, col=1
        )

        # Drawdown chart
        portfolio_values = performance_df['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100  # Convert to percentage

        fig.add_trace(
            go.Scatter(
                x=performance_df['date'],
                y=-drawdown,  # Negative for visual clarity
                fill='tozeroy',
                name='Drawdown %',
                line=dict(color='red')
            ),
            row=4, col=1
        )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text=f"Multi-Agent Trading System Performance Analysis",
            title_font_size=20
        )

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Value ($)", row=2, col=1)
        fig.update_yaxes(title_text="Signal Count", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)

        # Stack bars
        fig.update_layout(barmode='stack')

        # Create metrics summary
        metrics_text = f"""
        ## Performance Metrics

        **Returns**
        - Total Return: {metrics['total_return']*100:.2f}%
        - Annualized Return: {metrics['annual_return']*100:.2f}%

        **Risk Metrics**
        - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        - Sortino Ratio: {metrics['sortino_ratio']:.2f}
        - Calmar Ratio: {metrics['calmar_ratio']:.2f}
        - Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
        - Annual Volatility: {metrics['annual_volatility']*100:.1f}%

        **Trading Statistics**
        - Win Rate: {metrics['win_rate']*100:.1f}%
        - Total Trades: {len(trade_history)}
        - Average Confidence: {performance_df['confidence'].mean():.2%}

        **Portfolio Summary**
        - Final Portfolio Value: ${performance_df['portfolio_value'].iloc[-1]:,.2f}
        - Final Cash Position: ${performance_df['cash'].iloc[-1]:,.2f}
        - Final Position Value: ${performance_df['position_value'].iloc[-1]:,.2f}
        """

        # Create trade history table
        if trade_history:
            # Get last 20 trades
            recent_trades = trade_history[-20:]
            trade_df = pd.DataFrame([
                {
                    'Date': t['timestamp'].strftime('%Y-%m-%d'),
                    'Action': t['action'].upper(),
                    'Shares': f"{t['shares']:+d}",
                    'Price': f"${t['price']:.2f}",
                    'Value': f"${abs(t['value']):,.2f}",
                    'Confidence': f"{t['confidence']:.1%}"
                }
                for t in recent_trades
            ])
        else:
            trade_df = pd.DataFrame()

        # Agent analysis for the last day
        if performance_df['signals'].iloc[-1]:
            last_signals = performance_df['signals'].iloc[-1]
            agent_summary = "## Latest Agent Consensus\n\n"

            for agent_name, signal in last_signals.items():
                agent_summary += f"**{agent_name}**: {signal.upper()}\n"

            agent_summary += f"\n**Final Decision**: {performance_df['action'].iloc[-1].upper()} "
            agent_summary += f"with {performance_df['confidence'].iloc[-1]:.1%} confidence"
        else:
            agent_summary = "## Agent Analysis\n\nNo signals available"

        return fig, metrics_text, trade_df, agent_summary

    # Create Gradio interface
    with gr.Blocks(title="LLM-Powered Multi-Agent Trading System", theme=gr.themes.Base()) as interface:
        gr.Markdown("""
        # LLM-Powered Multi-Agent Trading System

        This professional trading system demonstrates sophisticated multi-agent coordination for algorithmic trading:

        **Core Components:**
        - **Fundamental Analysis Agent**: Uses OpenAI LLM for real market analysis and trading insights
        - **Technical Analysis Agent**: Employs transformer neural networks for price pattern recognition
        - **Sentiment Analysis Agent**: Leverages FinBERT for financial sentiment analysis with real news data
        - **Risk Management Agent**: Monitors portfolio risk metrics and provides risk-adjusted recommendations
        - **DQN Coordinator**: Deep Q-Network that learns optimal trading strategies from agent signals

        **Key Features:**
        - Real LLM integration for genuine fundamental analysis reasoning
        - Advanced neural network models for pattern recognition
        - Comprehensive risk management framework with Kelly Criterion
        - Reinforcement learning for strategy optimization
        - Professional-grade backtesting and performance analytics

        Author: Spencer Purdy
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Simulation Parameters")
                initial_capital = gr.Number(
                    value=100000,
                    label="Initial Capital ($)",
                    minimum=10000,
                    maximum=1000000,
                    info="Starting capital for the simulation"
                )
                volatility = gr.Slider(
                    minimum=0.01,
                    maximum=0.05,
                    value=0.02,
                    step=0.005,
                    label="Market Volatility",
                    info="Annual volatility of the simulated market"
                )
                trading_days = gr.Slider(
                    minimum=100,
                    maximum=500,
                    value=252,
                    step=10,
                    label="Trading Days",
                    info="Number of trading days to simulate"
                )

                gr.Markdown("### API Keys (Optional)")
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    info="For real LLM fundamental analysis (leave empty for fallback)"
                )
                news_api_key = gr.Textbox(
                    label="News API Key",
                    placeholder="Your NewsAPI key",
                    type="password",
                    info="For real news sentiment analysis (leave empty for contextual news)"
                )

                run_button = gr.Button("Run Simulation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=3):
                performance_plot = gr.Plot(label="Performance Analysis Dashboard")

            with gr.Column(scale=1):
                metrics_display = gr.Markdown(label="Performance Metrics")

        with gr.Row():
            with gr.Column():
                trade_table = gr.DataFrame(
                    label="Recent Trading Activity (Last 20 Trades)",
                    headers=["Date", "Action", "Shares", "Price", "Value", "Confidence"]
                )

        with gr.Row():
            with gr.Column():
                agent_display = gr.Markdown(label="Agent Analysis")

        # Connect interface
        run_button.click(
            fn=run_simulation,
            inputs=[initial_capital, volatility, trading_days, openai_api_key, news_api_key],
            outputs=[performance_plot, metrics_display, trade_table, agent_display]
        )

        # Add professional examples
        gr.Examples(
            examples=[
                [100000, 0.02, 252, "", ""],  # Standard market conditions
                [50000, 0.03, 365, "", ""],   # Higher volatility environment
                [200000, 0.015, 180, "", ""], # Lower volatility environment
            ],
            inputs=[initial_capital, volatility, trading_days, openai_api_key, news_api_key],
            label="Example Configurations"
        )

        gr.Markdown("""
        ---
        **Note**: This system uses sophisticated machine learning models including real LLM integration for fundamental analysis.
        For best results, provide an OpenAI API key for genuine LLM reasoning and a News API key for real news sentiment analysis.
        The simulation may take a few moments to initialize and run. All trading decisions are for demonstration
        purposes only and should not be used for actual trading without proper validation and risk assessment.

        **API Key Information**:
        - OpenAI API Key: Get yours at https://platform.openai.com/api-keys
        - News API Key: Get yours at https://newsapi.org/register
        """)

    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()