import json
import websockets
from datetime import datetime
from threading import Thread, Lock
import uuid
import asyncio
import os
import sqlite3
import sys
import gc
from contextlib import contextmanager
from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import math
import numpy as np
import pandas as pd
import random

# === LOGGING CONFIGURATION ===
LOG_DIR = os.environ.get('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'multiplier_bot.log'),
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s'
))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s'
))
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger('MultiplierBot')
trade_logger = logging.getLogger('TradeExecution')
db_logger = logging.getLogger('Database')
api_logger = logging.getLogger('API')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify

# Database setup - ENHANCED with new columns
DB_PATH = os.environ.get('DB_PATH', 'multiplier_trades.db')

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                app_id TEXT,
                status TEXT,
                success INTEGER,
                contract_id TEXT,
                profit REAL,
                final_balance REAL,
                initial_balance REAL,
                error TEXT,
                parameters TEXT,
                volatility REAL,
                direction TEXT,
                multiplier INTEGER,
                entry_price REAL,
                exit_price REAL,
                exit_reason TEXT,
                duration_seconds REAL,
                atr REAL,
                ema REAL,
                rsi REAL,
                macd TEXT,
                mtf_aligned INTEGER DEFAULT 0,
                tick_volume REAL,
                spread_pct REAL,
                optimal_stake REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_sessions (
                session_date TEXT PRIMARY KEY,
                trades_count INTEGER DEFAULT 0,
                consecutive_losses INTEGER DEFAULT 0,
                total_profit_loss REAL DEFAULT 0,
                stopped INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                component TEXT,
                message TEXT,
                details TEXT
            )
        ''')
        conn.commit()
        conn.close()
        db_logger.info("Database initialized successfully with enhanced schema")
    except Exception as e:
        db_logger.error(f"Database initialization failed: {e}")

@contextmanager
def get_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        db_logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        db_logger.error(f"Unexpected database error: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def log_system_event(level, component, message, details=None):
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_logs (timestamp, level, component, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    level,
                    component,
                    message,
                    json.dumps(details) if details else None
                ))
    except Exception as e:
        db_logger.error(f"Failed to log system event: {e}")

def save_trade(trade_id, trade_data):
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trades
                    (trade_id, timestamp, app_id, status, success, contract_id, profit,
                     final_balance, initial_balance, error, parameters, volatility, direction,
                     multiplier, entry_price, exit_price, exit_reason, duration_seconds,
                     atr, ema, rsi, macd, mtf_aligned, tick_volume, spread_pct, optimal_stake)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    trade_data.get('timestamp'),
                    trade_data.get('app_id'),
                    trade_data.get('status'),
                    1 if trade_data.get('success') else 0,
                    trade_data.get('contract_id'),
                    trade_data.get('profit'),
                    trade_data.get('final_balance'),
                    trade_data.get('initial_balance'),
                    trade_data.get('error'),
                    json.dumps(trade_data.get('parameters', {})),
                    trade_data.get('volatility'),
                    trade_data.get('direction'),
                    trade_data.get('multiplier'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('exit_reason'),
                    trade_data.get('duration_seconds'),
                    trade_data.get('atr'),
                    trade_data.get('ema'),
                    trade_data.get('rsi'),
                    trade_data.get('macd'),
                    1 if trade_data.get('mtf_aligned') else 0,
                    trade_data.get('tick_volume'),
                    trade_data.get('spread_pct'),
                    trade_data.get('optimal_stake')
                ))
                db_logger.debug(f"Trade {trade_id} saved successfully")
    except Exception as e:
        db_logger.error(f"Failed to save trade {trade_id}: {e}")

def get_trade(trade_id):
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
    except Exception as e:
        db_logger.error(f"Failed to get trade {trade_id}: {e}")
    return None

def get_all_trades():
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC')
                return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        db_logger.error(f"Failed to get all trades: {e}")
    return []

def get_recent_trades(limit=20):
    """Get last N completed trades for Kelly calculation"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT profit FROM trades 
                WHERE status = 'completed' 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [row['profit'] > 0 for row in rows]  # List of wins (True/False)
    except Exception as e:
        db_logger.error(f"Failed to get recent trades: {e}")
        return []

def get_session_data(session_date):
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trading_sessions WHERE session_date = ?', (session_date,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
    except Exception as e:
        db_logger.error(f"Failed to get session data: {e}")
    return None

def update_session_data(session_date, trades_count, consecutive_losses, total_profit_loss, stopped):
    try:
        with get_db() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trading_sessions
                    (session_date, trades_count, consecutive_losses, total_profit_loss, stopped, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_date, trades_count, consecutive_losses, total_profit_loss, stopped,
                      datetime.now().isoformat()))
    except Exception as e:
        db_logger.error(f"Failed to update session data: {e}")

try:
    init_db()
except Exception as e:
    logger.error(f"Database initialization error: {e}")

trade_results = {}
MAX_CONCURRENT_TRADES = 100
active_trades_lock = Lock()
active_trade_count = 0

def can_start_trade():
    global active_trade_count
    try:
        with active_trades_lock:
            if active_trade_count >= MAX_CONCURRENT_TRADES:
                logger.warning(f"Max concurrent trades reached: {active_trade_count}/{MAX_CONCURRENT_TRADES}")
                return False
            active_trade_count += 1
            logger.info(f"Trade slot acquired. Active: {active_trade_count}/{MAX_CONCURRENT_TRADES}")
            return True
    except Exception as e:
        logger.error(f"Error in can_start_trade: {e}")
        return False

def trade_completed():
    global active_trade_count
    try:
        with active_trades_lock:
            active_trade_count = max(0, active_trade_count - 1)
            logger.info(f"Trade slot released. Active: {active_trade_count}/{MAX_CONCURRENT_TRADES}")
    except Exception as e:
        logger.error(f"Error in trade_completed: {e}")

class TechnicalIndicators:
    @staticmethod
    def calculate_sma(prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def calculate_ema(prices, period):
        if len(prices) < period:
            return None
        prices_series = pd.Series(prices)
        return prices_series.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi[-1]

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        if len(prices) < slow + signal:
            return 0, 0, 0
        df = pd.Series(prices)
        exp1 = df.ewm(span=fast, adjust=False).mean()
        exp2 = df.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - exp3
        return macd.iloc[-1], exp3.iloc[-1], histogram.iloc[-1]

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        if len(prices) < period:
            return None, None, None
        sma = sum(prices[-period:]) / period
        variance = sum((x - sma) ** 2 for x in prices[-period:]) / period
        std = math.sqrt(variance)
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def calculate_atr(prices, period=14):
        if len(prices) < period + 1:
            return 0.0
        high_low = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return sum(high_low[-period:]) / period if high_low else 0.0

class VolatilityAnalyzer:
    @staticmethod
    def calculate_standard_deviation(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    @staticmethod
    def is_low_volatility(prices, threshold=0.20):
        if len(prices) < 20:
            return False, 0.0
        recent = prices[-20:]
        mean = sum(recent) / len(recent)
        pct_vol = (VolatilityAnalyzer.calculate_standard_deviation(recent) / mean) * 100 if mean > 0 else 0
        return pct_vol < threshold, pct_vol

class SignalGenerator:
    def __init__(self, prices, params):
        self.prices = prices
        self.params = params
        
    def get_signal(self):
        if len(self.prices) < 50:
            return "neutral", 0, {}

        ema_fast = TechnicalIndicators.calculate_ema(self.prices, self.params.get('ema_period', 12))
        ema_slow = TechnicalIndicators.calculate_ema(self.prices, 26)
        rsi = TechnicalIndicators.calculate_rsi(self.prices, 14)
        macd, macd_signal, hist = TechnicalIndicators.calculate_macd(self.prices)
        upper, mid, lower = TechnicalIndicators.calculate_bollinger_bands(self.prices)
        
        current_price = self.prices[-1]
        
        bullish_score = 0
        bearish_score = 0
        
        # Trend Score
        if ema_fast > ema_slow: bullish_score += 1
        else: bearish_score += 1
            
        # MACD Score
        if macd > macd_signal and hist > 0: bullish_score += 1
        elif macd < macd_signal and hist < 0: bearish_score += 1
            
        # RSI Score - Improved for volatility indices
        if rsi < 35: bullish_score += 2      # Slightly oversold
        elif rsi > 65: bearish_score += 2    # Slightly overbought
        elif rsi < 50: bullish_score += 1
        elif rsi > 50: bearish_score += 1
            
        # Bollinger Score
        if current_price < lower: bullish_score += 1
        elif current_price > upper: bearish_score += 1
        
        indicators = {
            "rsi": round(rsi, 1),
            "macd": f"{macd:.4f}/{macd_signal:.4f}",
            "ema": f"{ema_fast:.2f}/{ema_slow:.2f}",
            "bb": f"{lower:.2f}/{upper:.2f}"
        }

        # Relaxed threshold for more entries
        if bullish_score >= 2 and bearish_score <= 1:
            return "up", bullish_score, indicators
        elif bearish_score >= 2 and bullish_score <= 1:
            return "down", bearish_score, indicators
            
        return "neutral", 0, indicators

class DerivMultiplierBot:
    def __init__(self, api_token, app_id, trade_id, parameters):
        self.api_token = api_token
        self.app_id = app_id
        self.trade_id = trade_id
        self.params = parameters
        
        # Core parameters
        self.stake_per_trade = parameters.get('stake', 1.0)
        self.symbol = parameters.get('symbol', 'vol100_1s')  # Best for multipliers
        self.multiplier = parameters.get('multiplier', 300)
        self.risk_per_trade_pct = parameters.get('risk_per_trade_pct', 0.01)
        
        # New advanced features
        self.use_multi_timeframe = parameters.get('use_multi_timeframe', True)
        self.min_tick_volume = parameters.get('min_tick_volume', 0.001)
        self.max_spread_pct = parameters.get('max_spread_pct', 0.5)
        
        # Risk management
        self.stop_loss_amount = parameters.get('stop_loss_amount', 1.0)
        self.take_profit_amount = parameters.get('take_profit_amount', 8.0)  # High RR
        
        # Session controls
        self.max_daily_trades = parameters.get('max_daily_trades', 10)
        self.max_consecutive_losses = parameters.get('max_consecutive_losses', 2)
        self.daily_loss_limit_pct = parameters.get('daily_loss_limit_pct', 0.03)
        
        # Technical
        self.volatility_threshold = parameters.get('volatility_threshold', 0.80)
        self.atr_threshold = parameters.get('atr_threshold', 0.20)
        
        # State
        self.current_direction = None
        self.effective_multiplier = None
        self.mtf_aligned = False
        self.tick_volume = 0.0
        self.spread_pct = 0.0
        self.optimal_stake = 0.0
        
        self.ws_urls = [
            f"wss://ws.derivws.com/websockets/v3?app_id={app_id}",
            f"wss://wscluster1.deriv.com/websockets/v3?app_id={app_id}",
        ]
        self.ws_url = self.ws_urls[0]
        self.ws = None
        self.request_id = 0
        self.account_balance = 0.0
        self.initial_balance = 0.0
        self.symbol_available = False
        
        self.price_history = deque(maxlen=500)  # Increased for MTF
        self.trade_start_time = None
        self.entry_price = None
        self.exit_price = None
        
        trade_logger.info(f"Quantum Multiplier Pro initialized - ID: {trade_id}")
        trade_logger.info(f"MTF: {self.use_multi_timeframe} | Symbol: {self.symbol} | Multiplier: {self.multiplier}")
        
    def get_next_request_id(self):
        self.request_id += 1
        return self.request_id
        
    async def connect(self, retry_count=0, max_retries=3):
        for ws_url in self.ws_urls:
            self.ws_url = ws_url
            try:
                trade_logger.info(f"Connecting to {ws_url}")
                self.ws = await asyncio.wait_for(
                    websockets.connect(self.ws_url, ping_interval=None, close_timeout=5),
                    timeout=15.0
                )
                auth_success = await self.authorize()
                if not auth_success:
                    await self.ws.close()
                    continue
                return True
            except Exception as e:
                trade_logger.error(f"Connect failed: {e}")
                if self.ws: await self.ws.close()
                continue
        
        if retry_count < max_retries:
            await asyncio.sleep(10)
            return await self.connect(retry_count + 1, max_retries)
        return False
    
    async def authorize(self):
        try:
            auth_request = {"authorize": self.api_token, "req_id": self.get_next_request_id()}
            await self.ws.send(json.dumps(auth_request))
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)
            if "authorize" in data:
                self.account_balance = float(data['authorize']['balance'])
                self.initial_balance = self.account_balance
                trade_logger.info(f"Authorized | Balance: ${self.account_balance:.2f}")
                return True
        except Exception as e:
            trade_logger.error(f"Auth error: {e}")
        return False
    
    async def get_balance(self):
        try:
            balance_request = {"balance": 1, "subscribe": 0, "req_id": self.get_next_request_id()}
            response_data = await self.send_request(balance_request)
            if response_data and "balance" in response_data:
                self.account_balance = float(response_data["balance"]["balance"])
                return self.account_balance
        except Exception as e:
            trade_logger.error(f"Failed to get balance: {e}")
        return self.account_balance
    
    async def send_request(self, request):
        req_id = self.get_next_request_id()
        request["req_id"] = req_id
        
        try:
            await self.ws.send(json.dumps(request))
            while True:
                response_text = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                data = json.loads(response_text)
                if data.get("req_id") == req_id:
                    return data
                if "subscription" in data:
                    continue
        except Exception as e:
            trade_logger.error(f"Request failed: {e}")
        return None

    async def fetch_ticks(self, count=200):
        try:
            ticks_request = {
                "ticks_history": self.symbol,
                "count": count,
                "end": "latest",
                "style": "ticks",
                "granularity": 1,
                "req_id": self.get_next_request_id()
            }
            
            trade_logger.info(f"Fetching {count} ticks for {self.symbol}...")
            await self.ws.send(json.dumps(ticks_request))
            
            # Add explicit timeout
            response_text = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
            response = json.loads(response_text)
            
            if "error" in response:
                trade_logger.error(f"Tick history error: {response['error']['message']}")
                return []
                
            if "history" not in response or not response["history"]["prices"]:
                trade_logger.warning("No tick history received")
                return []
                
            prices = [float(p) for p in response["history"]["prices"]]
            trade_logger.info(f"Fetched {len(prices)} ticks successfully")
            return prices
            
        except asyncio.TimeoutError:
            trade_logger.error("Tick history request timed out")
            return []
        except Exception as e:
            trade_logger.error(f"Tick fetch failed: {e}")
            return []
    async def analyze_market(self):
        prices = await self.fetch_ticks(200)
        if len(prices) < 50:
            trade_logger.warning(f"Insufficient data ({len(prices)} ticks) - retrying later")
            return None, 0, {}, 0, False, 0, 0
        
        self.price_history.extend(prices)
        
        # Volume & Spread Analysis
        recent_prices = prices[-20:]
        self.tick_volume = abs(recent_prices[-1] - recent_prices[0])
        high = max(recent_prices)
        low = min(recent_prices)
        spread = high - low
        self.spread_pct = (spread / recent_prices[-1]) * 100 if recent_prices[-1] > 0 else 0
        
        # Multi-Timeframe Analysis
        mtf_aligned = False
        if self.use_multi_timeframe and len(prices) >= 200:
            short_gen = SignalGenerator(prices[-50:], self.params)
            medium_gen = SignalGenerator(prices[-100:], self.params)
            long_gen = SignalGenerator(prices[-200:], self.params)
            
            short_dir, short_score, _ = short_gen.get_signal()
            medium_dir, medium_score, _ = medium_gen.get_signal()
            long_dir, long_score, _ = long_gen.get_signal()
            
            if short_dir == medium_dir == long_dir and short_dir != "neutral":
                mtf_aligned = True
                trade_logger.info(f"MTF ALIGNMENT ACHIEVED: {short_dir.upper()} across all timeframes")
        
        # Primary signal from full history
        gen = SignalGenerator(prices, self.params)
        trend, score, indicators = gen.get_signal()
        
        atr = TechnicalIndicators.calculate_atr(prices, 14)
        
        return trend, score, indicators, atr, mtf_aligned, self.tick_volume, self.spread_pct

    def calculate_kelly_stake(self):
        recent_wins = get_recent_trades(20)
        if len(recent_wins) < 10:
            return self.stake_per_trade  # Not enough data
        
        win_rate = sum(recent_wins) / len(recent_wins)
        avg_win = self.take_profit_amount
        avg_loss = self.stop_loss_amount
        
        if win_rate == 0 or avg_loss == 0:
            return self.stake_per_trade * 0.5
        
        # Kelly formula: f = (bp - q) / b
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
        
        kelly = (b * p - q) / b if b > 0 else 0
        fractional_kelly = max(0, kelly * 0.25)  # 25% of full Kelly
        
        optimal = self.account_balance * fractional_kelly
        stake = min(optimal, self.stake_per_trade * 2)  # Cap at 2x base
        stake = max(stake, self.stake_per_trade * 0.5)  # Min 50%
        
        trade_logger.info(f"Kelly sizing: WinRate={win_rate:.1%} → Optimal stake=${stake:.2f}")
        return round(stake, 2)

    async def wait_for_entry_signal(self, max_wait_time=45):
        trade_logger.info("Starting entry signal search...")
        start_time = datetime.now()
        fallback_triggered = False
        
        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            trend, score, indicators, atr, mtf_aligned, tick_vol, spread_pct = await self.analyze_market()
            
            if trend is None:
                await asyncio.sleep(3)
                continue
            
            # Volume & Spread filter
            if tick_vol < self.min_tick_volume:
                trade_logger.debug(f"Low volume ({tick_vol:.5f}) - skipping")
                await asyncio.sleep(3)
                continue
            if spread_pct > self.max_spread_pct:
                trade_logger.debug(f"High spread ({spread_pct:.2f}%) - skipping")
                await asyncio.sleep(3)
                continue
            
            # MTF bonus
            if mtf_aligned:
                score += 2  # Strong confirmation bonus
            
            # Primary entry condition
            if trend != "neutral" and score >= 2 and atr < self.atr_threshold:
                self.current_direction = "MULTUP" if trend == "up" else "MULTDOWN"
                self.effective_multiplier = self.multiplier
                self.mtf_aligned = mtf_aligned
                self.tick_volume = tick_vol
                self.spread_pct = spread_pct
                
                if mtf_aligned:
                    trade_logger.info(f"ELITE SIGNAL: {trend.upper()} | Score: {score} | MTF ALIGNED")
                else:
                    trade_logger.info(f"STRONG SIGNAL: {trend.upper()} | Score: {score}")
                
                return True, "Elite signal confirmed"
            
            # Fallback after 30s
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 30 and not fallback_triggered and score >= 2:
                self.current_direction = "MULTUP" if trend == "up" else "MULTDOWN"
                self.effective_multiplier = self.multiplier // 2
                self.mtf_aligned = mtf_aligned
                self.tick_volume = tick_vol
                self.spread_pct = spread_pct
                fallback_triggered = True
                trade_logger.info(f"FALLBACK ENTRY: {trend.upper()} | Reduced multiplier {self.effective_multiplier}")
                return True, "Fallback entry"
            
            await asyncio.sleep(3)
        
        return False, "Timeout: No valid signal"

    async def check_trading_conditions(self):
        today = datetime.now().date().isoformat()
        session = get_session_data(today)
        
        if not session:
            update_session_data(today, 0, 0, 0.0, 0)
            return True, "New session"
        
        if session['stopped']:
            return False, "Trading stopped"
        
        if session['trades_count'] >= self.max_daily_trades:
            return False, f"Max trades reached ({self.max_daily_trades})"
        
        if session['consecutive_losses'] >= self.max_consecutive_losses:
            update_session_data(today, session['trades_count'], session['consecutive_losses'], 
                              session['total_profit_loss'], 1)
            return False, "Max losses - stopped"
        
        if session['total_profit_loss'] <= -(self.initial_balance * self.daily_loss_limit_pct):
            update_session_data(today, session['trades_count'], session['consecutive_losses'], 
                              session['total_profit_loss'], 1)
            return False, "Daily loss limit"
        
        return True, "OK"
    
    def update_session_after_trade(self, profit):
        today = datetime.now().date().isoformat()
        session = get_session_data(today) or {
            'trades_count': 0, 'consecutive_losses': 0, 'total_profit_loss': 0.0, 'stopped': 0
        }
        
        new_trades_count = session['trades_count'] + 1
        new_total_pl = session['total_profit_loss'] + profit
        new_consecutive = 0 if profit > 0 else session['consecutive_losses'] + 1
        
        update_session_data(today, new_trades_count, new_consecutive, new_total_pl, session['stopped'])
        trade_logger.info(f"Session: {new_trades_count} trades | {new_consecutive} losses | P/L: ${new_total_pl:.2f}")

    async def place_multiplier_trade(self):
        can_enter, reason = await self.wait_for_entry_signal()
        if not can_enter:
            return None, reason
        
        # Kelly Criterion sizing
        self.optimal_stake = self.calculate_kelly_stake()
        stake = self.optimal_stake
        
        proposal_request = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": self.current_direction,
            "currency": "USD",
            "symbol": self.symbol,
            "multiplier": self.effective_multiplier,
            "limit_order": {
                "stop_loss": self.stop_loss_amount,
                "take_profit": self.take_profit_amount
            }
        }
        
        try:
            response = await self.send_request(proposal_request)
            if "error" in response:
                return None, response["error"]["message"]
            
            proposal_id = response["proposal"]["id"]
            buy_request = {"buy": proposal_id, "price": response["proposal"]["ask_price"]}
            buy_response = await self.send_request(buy_request)
            
            if "error" in buy_response:
                return None, buy_response["error"]["message"]
            
            contract_id = buy_response["buy"]["contract_id"]
            self.entry_price = float(buy_response["buy"]["buy_price"])
            
            trade_logger.info(f"TRADE EXECUTED | Stake: ${stake:.2f} | Direction: {self.current_direction} | MTF: {'YES' if self.mtf_aligned else 'NO'}")
            return contract_id, None
        except Exception as e:
            return None, str(e)

    async def monitor_contract(self, contract_id):
        self.trade_start_time = datetime.now()
        subscribe_request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
            "req_id": self.get_next_request_id()
        }
        await self.ws.send(json.dumps(subscribe_request))
        
        while True:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                data = json.loads(response)
                
                if "proposal_open_contract" in data:
                    contract = data["proposal_open_contract"]
                    if contract.get("is_sold"):
                        profit = float(contract.get("profit", 0))
                        return {
                            "profit": profit,
                            "status": "win" if profit > 0 else "loss",
                            "exit_reason": contract.get("exit_reason", "limit_order"),
                            "duration_seconds": (datetime.now() - self.trade_start_time).total_seconds(),
                            "entry_price": self.entry_price,
                            "exit_price": float(contract.get("sell_price", 0)),
                            "mtf_aligned": self.mtf_aligned,
                            "tick_volume": self.tick_volume,
                            "spread_pct": self.spread_pct,
                            "optimal_stake": self.optimal_stake
                        }
            except asyncio.TimeoutError:
                trade_logger.error("Monitoring timeout")
                return {"profit": 0, "status": "error", "error": "timeout"}
            except Exception as e:
                trade_logger.error(f"Monitor error: {e}")
                return {"profit": 0, "status": "error", "error": str(e)}

    async def execute_trade_async(self):
        try:
            trade_results[self.trade_id] = {'status': 'running'}
            
            await self.connect()
            if not await self.authorize():
                return
            
            can_trade, reason = await self.check_trading_conditions()
            if not can_trade:
                trade_results[self.trade_id] = {"status": "skipped", "error": reason}
                return
            
            contract_id, error = await self.place_multiplier_trade()
            if error:
                trade_results[self.trade_id] = {"status": "error", "error": error}
                return
            
            result = await self.monitor_contract(contract_id)
            balance = await self.get_balance()
            
            self.update_session_after_trade(result["profit"])
            
            final_result = {
                "success": result["profit"] > 0,
                "trade_id": self.trade_id,
                "contract_id": contract_id,
                "profit": result["profit"],
                "status": "completed",
                "final_balance": balance,
                "timestamp": datetime.now().isoformat(),
                "direction": self.current_direction,
                "multiplier": self.effective_multiplier,
                "entry_price": result["entry_price"],
                "exit_price": result["exit_price"],
                "exit_reason": result["exit_reason"],
                "duration_seconds": result["duration_seconds"],
                "mtf_aligned": result.get("mtf_aligned", False),
                "tick_volume": result.get("tick_volume", 0),
                "spread_pct": result.get("spread_pct", 0),
                "optimal_stake": result.get("optimal_stake", self.stake_per_trade),
                "parameters": self.params
            }
            
            save_trade(self.trade_id, final_result)
            trade_results[self.trade_id] = final_result
            
        except Exception as e:
            trade_logger.error(f"Critical error: {e}")
        finally:
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            gc.collect()

# Flask Routes
app = Flask(__name__)
app.logger.disabled = True

@app.route('/trade/<app_id>/<api_token>', methods=['POST'])
def execute_trade(app_id, api_token):
    if not can_start_trade():
        return jsonify({"error": "System busy"}), 429
    
    data = request.get_json() or {}
    new_trade_id = str(uuid.uuid4())
    
    thread = Thread(target=lambda: asyncio.run(
        DerivMultiplierBot(api_token, app_id, new_trade_id, data).execute_trade_async()
    ))
    thread.start()
    
    return jsonify({"trade_id": new_trade_id, "status": "initiated"}), 202

@app.route('/trades', methods=['GET'])
def get_trades():
    return jsonify(get_all_trades())

@app.route('/session', methods=['GET'])
def get_session():
    today = datetime.now().date().isoformat()
    session = get_session_data(today)
    if not session:
        return jsonify({'trades_count': 0, 'consecutive_losses': 0, 'total_profit_loss': 0.0, 'stopped': 0})
    return jsonify(session)

@app.route('/session/resume', methods=['POST'])
def resume_session():
    today = datetime.now().date().isoformat()
    session = get_session_data(today)
    if session:
        update_session_data(today, session['trades_count'], session['consecutive_losses'],
                          session['total_profit_loss'], 0)
    return jsonify({"success": True, "message": "Trading resumed"})

@app.route('/session/reset', methods=['POST'])
def reset_session():
    today = datetime.now().date().isoformat()
    update_session_data(today, 0, 0, 0.0, 0)
    return jsonify({"success": True, "message": "Session reset"})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Multiplier Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #05070a;
            --card: #0c1017;
            --accent: #3b82f6;
            --success: #10b981;
            --danger: #ef4444;
            --text: #e2e8f0;
        }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .card { background-color: var(--card); border: 1px solid #1e293b; border-radius: 12px; }
        .stat-card { transition: transform 0.2s ease; }
        .stat-card:hover { transform: translateY(-2px); border-color: var(--accent); }
        .indicator-dot { height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .status-win { color: var(--success); }
        .status-loss { color: var(--danger); }
        .status-running { color: var(--accent); animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
            <div>
                <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                    QUANTUM MULTIPLIER <span class="bg-blue-500/10 text-blue-500 text-[10px] px-2 py-0.5 rounded border border-blue-500/20">PRO v2.0</span>
                </h1>
                <p id="last-update" class="text-slate-500 text-xs mt-1 mono uppercase tracking-tighter">System Idle | Syncing...</p>
            </div>
            <div class="flex gap-3">
                <button onclick="window.location.reload()" class="bg-slate-800 hover:bg-slate-700 text-white text-xs px-4 py-2 rounded-lg font-semibold transition-colors">REFRESH</button>
                <button onclick="fetch('/session/reset', {method:'POST'}).then(() => window.location.reload())" class="bg-red-500/10 hover:bg-red-500/20 text-red-500 text-xs px-4 py-2 rounded-lg font-semibold border border-red-500/20 transition-colors">RESET SESSION</button>
            </div>
        </header>

        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div class="card p-5 stat-card">
                <p class="text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-1">Total P/L</p>
                <p id="total-pl" class="text-2xl font-bold mono">$0.00</p>
            </div>
            <div class="card p-5 stat-card">
                <p class="text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-1">Win Rate</p>
                <p id="win-rate" class="text-2xl font-bold mono">0%</p>
            </div>
            <div class="card p-5 stat-card">
                <p class="text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-1">Trades</p>
                <div class="flex items-baseline gap-2">
                    <span id="trades-count" class="text-2xl font-bold mono">0</span>
                    <span id="wins-losses" class="text-xs text-slate-500">0W / 0L</span>
                </div>
            </div>
            <div class="card p-5 stat-card">
                <p class="text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-1">Session</p>
                <p id="session-status" class="text-2xl font-bold text-emerald-500">ACTIVE</p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 space-y-6">
                <div class="card overflow-hidden">
                    <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                        <h3 class="text-sm font-bold text-white uppercase tracking-wider">Active Execution Log</h3>
                        <div class="flex items-center gap-2">
                            <span class="indicator-dot bg-emerald-500"></span>
                            <span class="text-[10px] text-slate-500 font-bold uppercase">Live Terminal</span>
                        </div>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm">
                            <thead class="bg-slate-900/40 text-slate-500 text-[10px] font-bold uppercase tracking-widest">
                                <tr>
                                    <th class="px-6 py-4">Timestamp</th>
                                    <th class="px-6 py-4">Asset</th>
                                    <th class="px-6 py-4">Direction</th>
                                    <th class="px-6 py-4">Indicators</th>
                                    <th class="px-6 py-4 text-right">Result</th>
                                </tr>
                            </thead>
                            <tbody id="trade-table" class="divide-y divide-slate-800">
                                <!-- Trades injected here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="space-y-6">
                <div class="card p-6">
                    <h3 class="text-xs font-bold text-blue-500 uppercase tracking-widest mb-4">Market Conditions</h3>
                    <div class="space-y-4">
                        <div class="flex justify-between items-center pb-3 border-b border-slate-800/50">
                            <span class="text-xs text-slate-400">RSI (14)</span>
                            <span id="market-rsi" class="mono text-xs font-bold">--</span>
                        </div>
                        <div class="flex justify-between items-center pb-3 border-b border-slate-800/50">
                            <span class="text-xs text-slate-400">MACD Histogram</span>
                            <span id="market-macd" class="mono text-xs font-bold">--</span>
                        </div>
                        <div class="flex justify-between items-center pb-3 border-b border-slate-800/50">
                            <span class="text-xs text-slate-400">Volatility (ATR)</span>
                            <span id="market-atr" class="mono text-xs font-bold">--</span>
                        </div>
                    </div>
                </div>

                <div class="card p-6 bg-blue-500/5 border-blue-500/20">
                    <h3 class="text-xs font-bold text-white uppercase tracking-widest mb-2">Technical Summary</h3>
                    <p class="text-xs text-slate-400 leading-relaxed italic">
                        The Quantum Multiplier Pro utilizes a proprietary triple-confirmation engine combining Bollinger mean reversion, RSI strength, and MACD trend convergence.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function updateUI() {
            try {
                const response = await fetch('/trades');
                const trades = await response.json();
                const sessionRes = await fetch('/session');
                const session = await sessionRes.json();

                // Stats Update
                const completed = trades.filter(t => t.status === 'completed');
                const wins = completed.filter(t => t.profit > 0);
                const totalPL = completed.reduce((sum, t) => sum + t.profit, 0);
                const winRate = completed.length > 0 ? ((wins.length / completed.length) * 100).toFixed(1) : 0;

                document.getElementById('total-pl').innerText = `${totalPL >= 0 ? '+' : ''}$${totalPL.toFixed(2)}`;
                document.getElementById('total-pl').className = `text-2xl font-bold mono ${totalPL >= 0 ? 'text-emerald-500' : 'text-red-500'}`;
                document.getElementById('win-rate').innerText = `${winRate}%`;
                document.getElementById('trades-count').innerText = completed.length;
                document.getElementById('wins-losses').innerText = `${wins.length}W / ${completed.length - wins.length}L`;
                document.getElementById('last-update').innerText = `Last sync: ${new Date().toLocaleTimeString()} | SYSTEM STABLE`;
                document.getElementById('session-status').innerText = session.stopped ? 'STOPPED' : 'ACTIVE';
                document.getElementById('session-status').className = `text-2xl font-bold ${session.stopped ? 'text-red-500' : 'text-emerald-500'}`;

                // Table Update
                const tbody = document.getElementById('trade-table');
                tbody.innerHTML = '';
                
                trades.slice(0, 10).forEach(t => {
                    const profit = parseFloat(t.profit || 0);
                    const isRunning = t.status === 'running';
                    const resultText = isRunning ? 'RUNNING' : (profit > 0 ? '+$' + profit.toFixed(2) : '-$' + Math.abs(profit).toFixed(2));
                    const resultClass = isRunning ? 'status-running' : (profit > 0 ? 'status-win' : 'status-loss');
                    const indicators = `RSI:${t.rsi?.toFixed(0) || '--'} MACD:${t.macd || '--'}`;

                    tbody.innerHTML += `
                        <tr class="hover:bg-slate-800/30 transition-colors">
                            <td class="px-6 py-4 text-[10px] text-slate-500 mono">${t.timestamp.split('T')[1].substring(0,8)}</td>
                            <td class="px-6 py-4 font-semibold text-slate-200">${t.parameters?.symbol || t.symbol || 'R_100'}</td>
                            <td class="px-6 py-4">
                                <span class="text-[10px] font-bold px-2 py-0.5 rounded ${t.direction === 'MULTUP' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 'bg-red-500/10 text-red-500 border border-red-500/20'}">
                                    ${t.direction === 'MULTUP' ? 'MULTUP' : 'MULTDOWN'}
                                </span>
                            </td>
                            <td class="px-6 py-4 text-[10px] text-slate-500 mono">${indicators}</td>
                            <td class="px-6 py-4 text-right font-bold mono ${resultClass}">${resultText}</td>
                        </tr>
                    `;
                    
                    if (!isRunning && t === trades[0]) {
                        document.getElementById('market-rsi').innerText = t.rsi?.toFixed(2) || '--';
                        document.getElementById('market-macd').innerText = t.macd || '--';
                        document.getElementById('market-atr').innerText = t.atr?.toFixed(4) || '--';
                    }
                });
            } catch (e) { console.error('UI Sync Failed:', e); }
        }

        setInterval(updateUI, 5000);
        updateUI();
    </script>
</body>
</html>
"""
    return html, 200
if __name__ == '__main__':
    trade_logger.info("Quantum Multiplier Pro v3.0 - Elite Edition Starting...")
    trade_logger.info("Features: MTF + Kelly Sizing + Volume Filter + Enhanced Risk")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)