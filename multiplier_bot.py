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

# Database setup
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
                ema REAL
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
        db_logger.info("Database initialized successfully")
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
                     atr, ema)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    trade_data.get('ema')
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
MAX_CONCURRENT_TRADES = 2
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

class VolatilityAnalyzer:
    @staticmethod
    def calculate_standard_deviation(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    @staticmethod
    def calculate_atr(prices, period=14):
        if len(prices) < period + 1:
            return 0.0
        high_low = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return sum(high_low[-period:]) / period if high_low else 0.0
    
    @staticmethod
    def calculate_ema(prices, period=20):
        prices_series = pd.Series(prices)
        return prices_series.ewm(span=period, adjust=False).mean().iloc[-1]
    
    @staticmethod
    def detect_trend(prices, ema_period=20):
        if len(prices) < ema_period:
            return "neutral"
        ema = VolatilityAnalyzer.calculate_ema(prices, ema_period)
        current_price = prices[-1]
        if current_price > ema * 1.0005:
            return "up"
        elif current_price < ema * 0.9995:
            return "down"
        return "neutral"
    
    @staticmethod
    def is_low_volatility(prices, threshold=0.20):
        if len(prices) < 20:
            return False, 0.0
        recent = prices[-20:]
        mean = sum(recent) / len(recent)
        pct_vol = (VolatilityAnalyzer.calculate_standard_deviation(recent) / mean) * 100 if mean > 0 else 0
        return pct_vol < threshold, pct_vol

class EnhancedSafetyChecks:
    @staticmethod
    def is_safe_entry(volatility_pct, atr, max_vol=0.20, max_atr=0.05):
        return volatility_pct < max_vol and atr < max_atr

class DerivMultiplierBot:
    def __init__(self, api_token, app_id, trade_id, parameters):
        self.api_token = api_token
        self.app_id = app_id
        self.trade_id = trade_id
        self.stake_per_trade = parameters.get('stake', 10.0)
        self.symbol = parameters.get('symbol', 'R_100')
        self.multiplier = parameters.get('multiplier', 20)
        self.risk_per_trade_pct = parameters.get('risk_per_trade_pct', 0.01)
        self.ema_period = parameters.get('ema_period', 20)
        self.atr_period = parameters.get('atr_period', 14)
        self.stop_loss_multiplier = parameters.get('stop_loss_multiplier', 1.5)
        self.take_profit_multiplier = parameters.get('take_profit_multiplier', 3.0)
        
        self.max_daily_trades = parameters.get('max_daily_trades', 5)
        self.max_consecutive_losses = parameters.get('max_consecutive_losses', 2)
        self.daily_loss_limit_pct = parameters.get('daily_loss_limit_pct', 0.02)
        
        self.volatility_threshold = parameters.get('volatility_threshold', 0.20)
        self.atr_threshold = parameters.get('atr_threshold', 0.05)
        
        self.current_direction = None
        self.volatility = None
        self.atr = None
        self.ema = None
        
        self.ws_urls = [
            f"wss://ws.derivws.com/websockets/v3?app_id={app_id}",
            f"wss://wscluster1.deriv.com/websockets/v3?app_id={app_id}",
            f"wss://wscluster2.deriv.com/websockets/v3?app_id={app_id}",
        ]
        self.ws_url = self.ws_urls[0]
        self.ws = None
        self.request_id = 0
        self.account_balance = 0.0
        self.initial_balance = 0.0
        self.symbol_available = False
        
        self.price_history = deque(maxlen=300)
        self.trade_start_time = None
        self.entry_price = None
        self.exit_price = None
        
        trade_logger.info(f"Bot initialized - Trade ID: {trade_id}, Symbol: {self.symbol}, Multiplier: {self.multiplier}")
        
    def get_next_request_id(self):
        self.request_id += 1
        return self.request_id
        
    async def connect(self, retry_count=0, max_retries=3):
        for ws_url in self.ws_urls:
            self.ws_url = ws_url
            try:
                trade_logger.info(f"Attempting connection to {ws_url}")
                self.ws = await asyncio.wait_for(
                    websockets.connect(self.ws_url, ping_interval=None, close_timeout=5),
                    timeout=15.0
                )
                auth_success = await self.authorize()
                if not auth_success:
                    trade_logger.warning(f"Authorization failed on {ws_url}")
                    await self.ws.close()
                    self.ws = None
                    raise Exception("Authorization failed")
                trade_logger.info(f"Successfully connected and authorized on {ws_url}")
                return True
            except Exception as e:
                trade_logger.error(f"Connection failed to {ws_url}: {e}")
                if self.ws:
                    await self.ws.close()
                    self.ws = None
                continue
        
        if retry_count < max_retries:
            wait_time = 10 * (2 ** retry_count)
            trade_logger.info(f"Retry {retry_count + 1}/{max_retries} in {wait_time}s")
            await asyncio.sleep(wait_time)
            return await self.connect(retry_count + 1, max_retries)
        else:
            trade_logger.error("Failed to connect after all retries")
            raise Exception("Failed to connect after retries")
    
    async def authorize(self):
        if not self.api_token:
            trade_logger.error("No API token provided")
            return False
        
        try:
            auth_request = {
                "authorize": self.api_token,
                "req_id": self.get_next_request_id()
            }
            await self.ws.send(json.dumps(auth_request))
            
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "error" in data:
                trade_logger.error(f"Authorization error: {data['error']}")
                return False
            
            if "authorize" in data:
                self.account_balance = float(data['authorize']['balance'])
                self.initial_balance = self.account_balance
                trade_logger.info(f"Authorized successfully. Balance: ${self.account_balance:.2f}")
                return True
        except Exception as e:
            trade_logger.error(f"Authorization exception: {e}")
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
    
    async def analyze_ticks(self, periods=100):
        try:
            ticks_request = {
                "ticks_history": self.symbol,
                "count": periods,
                "end": "latest",
                "style": "ticks",
                "req_id": self.get_next_request_id()
            }
            response = await self.send_request(ticks_request)
            
            if not response or "error" in response:
                trade_logger.warning(f"Tick history request failed: {response.get('error', 'No response') if response else 'None'}")
                return None, None, None, None, None
                
            if "history" not in response or "prices" not in response["history"]:
                trade_logger.warning(f"Incomplete tick history response: {response}")
                return None, None, None, None, None
                
            prices = [float(p) for p in response["history"]["prices"]]
            if len(prices) < 30:
                trade_logger.warning(f"Insufficient ticks received: {len(prices)} < 30")
                return None, None, None, None, None
                
            self.price_history.extend(prices)
            
            vol_pct = (VolatilityAnalyzer.calculate_standard_deviation(prices[-20:]) / 
                       np.mean(prices[-20:])) * 100 if np.mean(prices[-20:]) > 0 else 0
            atr = VolatilityAnalyzer.calculate_atr(prices, self.atr_period)
            ema = VolatilityAnalyzer.calculate_ema(prices, self.ema_period)
            trend = VolatilityAnalyzer.detect_trend(prices, self.ema_period)
            
            trade_logger.info(f"📊 Analysis - Ticks: {len(prices)}, Vol: {vol_pct:.4f}%, ATR: {atr:.6f}, EMA: {ema:.2f}, Trend: {trend.upper()}")
            
            return vol_pct, atr, ema, trend, prices
            
        except Exception as e:
            trade_logger.error(f"Exception in analyze_ticks: {e}")
            return None, None, None, None, None
    
    async def wait_for_entry_signal(self, max_wait_time=30, check_interval=1):
        trade_logger.info("🔍 STEP 1/3: Waiting for entry signal...")
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            vol_pct, atr, ema, trend, prices = await self.analyze_ticks()
            if vol_pct is None:
                await asyncio.sleep(check_interval)
                continue
            
            is_low_vol, pct_vol = VolatilityAnalyzer.is_low_volatility(prices, self.volatility_threshold)
            is_safe = EnhancedSafetyChecks.is_safe_entry(pct_vol, atr, self.volatility_threshold, self.atr_threshold)
            
            if is_safe and trend != "neutral":
                self.volatility = pct_vol
                self.atr = atr
                self.ema = ema
                self.current_direction = "MULTUP" if trend == "up" else "MULTDOWN"
                trade_logger.info(f"✅ ENTRY SIGNAL FOUND - Direction: {trend.upper()}, Vol: {pct_vol:.4f}%")
                return True, pct_vol, trend, "Signal approved"
            else:
                if not is_safe:
                    trade_logger.debug(f"⏳ Waiting - High volatility/ATR ({pct_vol:.4f}% / {atr:.6f})")
                if trend == "neutral":
                    trade_logger.debug(f"⏳ Waiting - No clear trend (price near EMA)")
            
            await asyncio.sleep(check_interval)
        
        trade_logger.info("⏰ Max wait time reached - no valid entry signal")
        return False, None, None, "Timeout: No safe entry signal"
    
    async def check_trading_conditions(self):
        today = datetime.now().date().isoformat()
        session = get_session_data(today)
        
        if not session:
            update_session_data(today, 0, 0, 0.0, 0)
            trade_logger.info("New trading session started")
            return True, "New session started"
        
        if session['stopped']:
            trade_logger.warning("Trading stopped for today")
            return False, "Trading stopped for today"
        
        if session['trades_count'] >= self.max_daily_trades:
            update_session_data(today, session['trades_count'],
                              session['consecutive_losses'],
                              session['total_profit_loss'], 1)
            trade_logger.warning(f"Max daily trades reached: {self.max_daily_trades}")
            return False, f"Max daily trades reached ({self.max_daily_trades})"
        
        if session['consecutive_losses'] >= self.max_consecutive_losses:
            update_session_data(today, session['trades_count'],
                              session['consecutive_losses'],
                              session['total_profit_loss'], 1)
            trade_logger.warning(f"Max consecutive losses reached: {self.max_consecutive_losses}")
            return False, f"Max consecutive losses reached ({self.max_consecutive_losses})"
        
        daily_loss_limit = self.initial_balance * self.daily_loss_limit_pct
        if session['total_profit_loss'] <= -daily_loss_limit:
            update_session_data(today, session['trades_count'],
                              session['consecutive_losses'],
                              session['total_profit_loss'], 1)
            trade_logger.warning(f"Daily loss limit reached: ${daily_loss_limit:.2f}")
            return False, f"Daily loss limit reached ({daily_loss_limit:.2f})"
        
        trade_logger.info("Trading conditions check passed")
        return True, "Trading conditions OK"
    
    def update_session_after_trade(self, profit):
        today = datetime.now().date().isoformat()
        session = get_session_data(today) or {
            'trades_count': 0,
            'consecutive_losses': 0,
            'total_profit_loss': 0.0,
            'stopped': 0
        }
        
        new_trades_count = session['trades_count'] + 1
        new_total_pl = session['total_profit_loss'] + profit
        
        if profit <= 0:
            new_consecutive_losses = session['consecutive_losses'] + 1
        else:
            new_consecutive_losses = 0
        
        update_session_data(today, new_trades_count, new_consecutive_losses,
                          new_total_pl, session['stopped'])
        
        trade_logger.info(f"Session updated: Trades={new_trades_count}, Losses={new_consecutive_losses}, P/L=${new_total_pl:.2f}")
    
    async def validate_symbol(self):
        try:
            spec_request = {
                "contracts_for": self.symbol,
                "req_id": self.get_next_request_id()
            }
            response = await self.send_request(spec_request)
            
            if not response or "error" in response:
                trade_logger.error(f"Symbol validation failed for {self.symbol}")
                return False
            
            if "contracts_for" in response:
                contracts = response["contracts_for"].get("available", [])
                has_multiplier = any(c.get("contract_category") == "multipliers" for c in contracts)
                if has_multiplier:
                    self.symbol_available = True
                    trade_logger.info(f"Symbol {self.symbol} validated successfully")
                    return True
        except Exception as e:
            trade_logger.error(f"Symbol validation exception: {e}")
        return False
    
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
    
    async def place_multiplier_trade(self):
        try:
            balance = await self.get_balance()
            if balance < self.stake_per_trade:
                trade_logger.error(f"❌ Insufficient balance: ${balance:.2f}")
                return None, "Insufficient balance"
            
            if not self.symbol_available:
                if not await self.validate_symbol():
                    return None, "Symbol validation failed"
            
            can_enter, vol, trend, reason = await self.wait_for_entry_signal()
            if not can_enter:
                trade_logger.info(f"🚫 SKIP: {reason}")
                return None, reason
            
            trade_logger.info("🔍 STEP 2/3: Calculating risk...")
            risk_amount = balance * self.risk_per_trade_pct
            stake = min(self.stake_per_trade, risk_amount)
            
            stop_loss = self.atr * self.stop_loss_multiplier
            take_profit = self.atr * self.take_profit_multiplier
            
            trade_logger.info("🔍 STEP 3/3: Final safety check...")
            is_safe = EnhancedSafetyChecks.is_safe_entry(vol, self.atr, self.volatility_threshold, self.atr_threshold)
            if not is_safe:
                trade_logger.warning("⚠️ Final safety check failed - aborting")
                return None, "Final safety check failed"
            
            trade_logger.info(f"✅ All checks passed - placing {self.current_direction} trade")
            
            proposal_request = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": self.current_direction,
                "currency": "USD",
                "symbol": self.symbol,
                "multiplier": self.multiplier,
                "limit_order": {
                    "stop_loss": {"value": stop_loss},
                    "take_profit": {"value": take_profit}
                }
            }
            
            proposal_response = await self.send_request(proposal_request)
            if not proposal_response or "error" in proposal_response:
                trade_logger.error(f"❌ Proposal failed: {proposal_response.get('error') if proposal_response else 'None'}")
                return None, "Proposal failed"
            
            proposal_id = proposal_response["proposal"]["id"]
            ask_price = proposal_response["proposal"]["ask_price"]
            
            buy_request = {"buy": proposal_id, "price": ask_price}
            buy_response = await self.send_request(buy_request)
            
            if not buy_response or "error" in buy_response:
                trade_logger.error(f"❌ Buy failed: {buy_response.get('error') if buy_response else 'None'}")
                return None, "Buy failed"
            
            contract_id = buy_response["buy"]["contract_id"]
            self.entry_price = float(buy_response["buy"]["buy_price"])
            trade_logger.info(f"✅ TRADE PLACED - Contract: {contract_id}, Stake: ${stake:.2f}")
            
            return contract_id, None
            
        except Exception as e:
            trade_logger.error(f"❌ Exception in place_multiplier_trade: {e}")
            return None, str(e)
    
    async def monitor_contract(self, contract_id):
        try:
            self.trade_start_time = datetime.now()
            req_id = self.get_next_request_id()
            proposal_request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1,
                "req_id": req_id
            }
            await self.ws.send(json.dumps(proposal_request))
            
            while True:
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    data = json.loads(response)
                    
                    if "proposal_open_contract" in data:
                        contract = data["proposal_open_contract"]
                        
                        if contract.get("is_sold") or contract.get("status") == "sold":
                            profit = float(contract.get("profit", 0))
                            self.exit_price = float(contract.get("sell_price", 0))
                            duration = (datetime.now() - self.trade_start_time).total_seconds()
                            exit_reason = contract.get("exit_reason", "unknown")
                            
                            trade_logger.info(f"Trade closed - Profit: ${profit:.2f}, Duration: {duration:.1f}s, Reason: {exit_reason}")
                            
                            try:
                                forget_request = {
                                    "forget": data.get("subscription", {}).get("id"),
                                    "req_id": self.get_next_request_id()
                                }
                                await self.ws.send(json.dumps(forget_request))
                            except:
                                pass
                            
                            return {
                                "profit": profit,
                                "status": "win" if profit > 0 else "loss",
                                "contract_id": contract_id,
                                "exit_reason": exit_reason,
                                "duration_seconds": duration,
                                "entry_price": self.entry_price,
                                "exit_price": self.exit_price
                            }
                        
                        current_profit = float(contract.get("profit", 0))
                        trade_logger.debug(f"Current P/L: ${current_profit:.2f}")
                        
                    elif "error" in data:
                        trade_logger.error(f"Contract error: {data['error']['message']}")
                        return {
                            "profit": 0,
                            "status": "error",
                            "error": data['error']['message'],
                            "exit_reason": "error",
                            "duration_seconds": (datetime.now() - self.trade_start_time).total_seconds()
                        }
                except asyncio.TimeoutError:
                    trade_logger.error("Monitoring timeout")
                    return {
                        "profit": 0,
                        "status": "error",
                        "error": "Timeout",
                        "exit_reason": "timeout",
                        "duration_seconds": (datetime.now() - self.trade_start_time).total_seconds()
                    }
        except Exception as e:
            trade_logger.error(f"Monitor exception: {e}")
            return {
                "profit": 0,
                "status": "error",
                "error": str(e),
                "exit_reason": "monitor_failed",
                "duration_seconds": 0
            }
    
    async def execute_trade_async(self):
        try:
            trade_results[self.trade_id] = {'status': 'running'}
            save_trade(self.trade_id, {
                'timestamp': datetime.now().isoformat(),
                'app_id': self.app_id,
                'status': 'running',
                'success': 0,
                'initial_balance': self.initial_balance,
                'parameters': {
                    'stake': self.stake_per_trade,
                    'symbol': self.symbol,
                    'multiplier': self.multiplier
                }
            })
            
            await self.connect()
            
            can_trade, reason = await self.check_trading_conditions()
            if not can_trade:
                result = {
                    "success": False,
                    "error": reason,
                    "trade_id": self.trade_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "skipped"
                }
                save_trade(self.trade_id, result)
                trade_results[self.trade_id] = result
                return result
            
            contract_id, error = await self.place_multiplier_trade()
            if error:
                result = {
                    "success": False,
                    "error": error,
                    "trade_id": self.trade_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "skipped"
                }
                save_trade(self.trade_id, result)
                trade_results[self.trade_id] = result
                return result
            
            monitor_result = await self.monitor_contract(contract_id)
            balance = await self.get_balance()
            
            self.update_session_after_trade(monitor_result.get("profit", 0))
            
            result = {
                "success": True,
                "trade_id": self.trade_id,
                "contract_id": contract_id,
                "profit": monitor_result.get("profit", 0),
                "status": "completed",
                "final_balance": balance,
                "initial_balance": self.initial_balance,
                "timestamp": datetime.now().isoformat(),
                "app_id": self.app_id,
                "volatility": self.volatility,
                "direction": self.current_direction,
                "multiplier": self.multiplier,
                "entry_price": monitor_result.get("entry_price"),
                "exit_price": monitor_result.get("exit_price"),
                "exit_reason": monitor_result.get("exit_reason"),
                "duration_seconds": monitor_result.get("duration_seconds"),
                "atr": self.atr,
                "ema": self.ema,
                "parameters": {
                    'stake': self.stake_per_trade,
                    'symbol': self.symbol,
                    'multiplier': self.multiplier
                }
            }
            save_trade(self.trade_id, result)
            trade_results[self.trade_id] = result
            
            return result
        except Exception as e:
            trade_logger.error(f"Execute trade exception: {e}")
            result = {
                "success": False,
                "error": str(e),
                "trade_id": self.trade_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
            save_trade(self.trade_id, result)
            trade_results[self.trade_id] = result
            return result
        finally:
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            gc.collect()

def run_async_trade_in_thread(api_token, app_id, parameters, trade_id):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        bot = DerivMultiplierBot(api_token, app_id, trade_id, parameters)
        loop.run_until_complete(bot.execute_trade_async())
        loop.close()
    except Exception as e:
        logger.error(f"Thread execution error: {e}")
    finally:
        trade_completed()
        gc.collect()

app = Flask(__name__)
app.logger.disabled = True

@app.route('/trade/<app_id>/<api_token>', methods=['POST'])
def execute_trade(app_id, api_token):
    try:
        if not can_start_trade():
            return jsonify({
                "success": False,
                "error": "Too many concurrent trades",
                "max_concurrent": MAX_CONCURRENT_TRADES
            }), 429
        
        data = request.get_json(silent=True) or {}
        
        parameters = {
            'stake': float(data.get('stake', 10.0)),
            'symbol': data.get('symbol', 'R_100'),
            'multiplier': int(data.get('multiplier', 20)),
            'risk_per_trade_pct': float(data.get('risk_per_trade_pct', 0.01)),
            'ema_period': int(data.get('ema_period', 20)),
            'atr_period': int(data.get('atr_period', 14)),
            'stop_loss_multiplier': float(data.get('stop_loss_multiplier', 1.5)),
            'take_profit_multiplier': float(data.get('take_profit_multiplier', 3.0)),
            'max_daily_trades': int(data.get('max_daily_trades', 5)),
            'max_consecutive_losses': int(data.get('max_consecutive_losses', 2)),
            'daily_loss_limit_pct': float(data.get('daily_loss_limit_pct', 0.02)),
            'volatility_threshold': float(data.get('volatility_threshold', 0.20)),
            'atr_threshold': float(data.get('atr_threshold', 0.05))
        }
        
        new_trade_id = str(uuid.uuid4())
        initial_data = {
            "status": "pending",
            "timestamp": datetime.now().isoformat(),
            "app_id": app_id,
            "parameters": parameters
        }
        save_trade(new_trade_id, initial_data)
        trade_results[new_trade_id] = initial_data
        
        api_logger.info(f"Trade initiated - ID: {new_trade_id}")
        thread = Thread(
            target=run_async_trade_in_thread,
            args=(api_token, app_id, parameters, new_trade_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "initiated", "trade_id": new_trade_id}), 202
    except Exception as e:
        api_logger.error(f"Endpoint error: {e}")
        trade_completed()
        return jsonify({"success": False, "error": "Internal Server Error"}), 500

@app.route('/trade/<trade_id>', methods=['GET'])
def get_trade_result(trade_id):
    result = trade_results.get(trade_id) or get_trade(trade_id)
    
    if not result:
        return jsonify({"success": False, "error": "Trade ID not found"}), 404
    
    response = {
        "trade_id": trade_id,
        "status": result.get('status', 'unknown'),
        "timestamp": result.get('timestamp')
    }
    
    if result.get('status') in ['completed', 'skipped', 'error']:
        profit = result.get('profit', 0)
        response.update({
            "success": result.get('success', False),
            "profit_loss": profit,
            "result": "PROFIT" if profit > 0 else "LOSS" if profit < 0 else "SKIPPED",
            "amount": abs(profit),
            "contract_id": result.get('contract_id'),
            "final_balance": result.get('final_balance'),
            "volatility": result.get('volatility'),
            "direction": result.get('direction'),
            "exit_reason": result.get('exit_reason'),
            "duration_seconds": result.get('duration_seconds'),
            "error_details": result.get('error')
        })
    
    return jsonify(response), 200

@app.route('/session', methods=['GET'])
def get_session_status():
    today = datetime.now().date().isoformat()
    session = get_session_data(today)
    
    if not session:
        return jsonify({
            "session_date": today,
            "trades_count": 0,
            "consecutive_losses": 0,
            "total_profit_loss": 0.0,
            "stopped": False,
            "message": "No trades today"
        }), 200
    
    return jsonify({
        "session_date": today,
        "trades_count": session['trades_count'],
        "consecutive_losses": session['consecutive_losses'],
        "total_profit_loss": round(session['total_profit_loss'], 2),
        "stopped": bool(session['stopped']),
        "can_trade": not bool(session['stopped'])
    }), 200

@app.route('/session/reset', methods=['POST'])
def reset_session():
    today = datetime.now().date().isoformat()
    update_session_data(today, 0, 0, 0.0, 0)
    api_logger.info("Session reset")
    return jsonify({"message": "Session reset successfully", "date": today}), 200

@app.route('/trades', methods=['GET'])
def get_all_trades_endpoint():
    all_trades = get_all_trades()
    filter_by = request.args.get('filter', 'today')
    
    from datetime import date
    today = date.today()
    
    if filter_by == 'today':
        filtered_trades = [t for t in all_trades if t.get('timestamp', '').startswith(today.isoformat())]
    elif filter_by == 'all':
        filtered_trades = all_trades
    else:
        filtered_trades = [t for t in all_trades if t.get('timestamp', '').startswith(today.isoformat())]
    
    completed = [t for t in filtered_trades if t.get('status') == 'completed']
    skipped = [t for t in filtered_trades if t.get('status') == 'skipped']
    
    wins = [t for t in completed if t.get('profit', 0) > 0]
    losses = [t for t in completed if t.get('profit', 0) <= 0]
    total_profit = sum(t.get('profit', 0) for t in completed)
    
    trades_dict = {t['trade_id']: dict(t) for t in filtered_trades}
    
    win_rate = f"{(len(wins)/len(completed)*100):.2f}%" if completed else "0%"
    
    return jsonify({
        "filter": filter_by,
        "date": today.isoformat(),
        "total_trades": len(filtered_trades),
        "completed_trades": len(completed),
        "skipped_trades": len(skipped),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_profit_loss": round(total_profit, 2),
        "trades": trades_dict
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Deriv Multiplier Bot v1.1 - Improved Logging & Reliability",
        "timestamp": datetime.now().isoformat(),
        "active_trades": active_trade_count,
        "max_concurrent": MAX_CONCURRENT_TRADES,
        "mode": "TREND_FOLLOWING",
        "features": [
            "Pure Python indicators (no TA-Lib)",
            "Enhanced logging for entry decisions",
            "Robust tick fetching with fallbacks",
            "Clear skip vs place distinction",
            "24/7 ready with auto-reconnect"
        ]
    }), 200

@app.route('/config/optimal', methods=['GET'])
def get_optimal_config():
    return jsonify({
        "recommended_setup": {
            "description": "Safe trend-following multipliers on volatility indices",
            "parameters": {
                "stake": 10.0,
                "symbol": "R_100",
                "multiplier": 20,
                "risk_per_trade_pct": 0.01,
                "ema_period": 20,
                "atr_period": 14,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 3.0,
                "volatility_threshold": 0.25,
                "atr_threshold": 0.06
            },
            "notes": [
                "Relaxed thresholds for more entries",
                "Use R_100 or 1HZ100V for steady trends",
                "Cron every 10-15 minutes",
                "Monitor dashboard for signal frequency"
            ]
        }
    }), 200

@app.route('/dashboard', methods=['GET'])
def dashboard():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiplier Bot Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0b0e14; color: #e2e8f0; }
        .glass { background: rgba(23, 27, 34, 0.8); backdrop-filter: blur(8px); border: 1px solid #1e293b; }
        .status-badge { padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
        .win { background: rgba(16, 185, 129, 0.1); color: #10b981; }
        .loss { background: rgba(239, 68, 68, 0.1); color: #ef4444; }
        .running { background: rgba(59, 130, 246, 0.1); color: #3b82f6; animation: pulse 2s infinite; }
        .skipped { background: rgba(251, 191, 36, 0.1); color: #f59e0b; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body class="p-6">
    <div class="max-w-6xl mx-auto">
        <header class="flex justify-between items-center mb-8 border-b border-slate-800 pb-6">
            <div>
                <h1 class="text-xl font-bold">MULTIPLIER BOT <span class="text-blue-500 text-xs">v1.1</span></h1>
                <p id="sync-status" class="text-slate-500 text-xs mt-1 italic">Last update: --:--:--</p>
            </div>
            <div class="text-right">
                <p class="text-slate-500 text-[10px] uppercase tracking-widest">Session P/L</p>
                <h2 id="session-pl" class="text-2xl font-bold">$0.00</h2>
            </div>
        </header>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="glass p-5 rounded-xl">
                <p class="text-slate-500 text-xs font-semibold mb-1 uppercase">Total Attempts</p>
                <p id="stat-total" class="text-xl font-bold">0</p>
            </div>
            <div class="glass p-5 rounded-xl">
                <p class="text-slate-500 text-xs font-semibold mb-1 uppercase">Win Rate</p>
                <p id="stat-winrate" class="text-xl font-bold">0%</p>
            </div>
            <div class="glass p-5 rounded-xl">
                <p class="text-slate-500 text-xs font-semibold mb-1 uppercase">Skipped</p>
                <p id="stat-skipped" class="text-xl font-bold text-amber-500">0</p>
            </div>
            <div class="glass p-5 rounded-xl">
                <p class="text-slate-500 text-xs font-semibold mb-1 uppercase">Status</p>
                <p id="safety-label" class="text-xl font-bold text-emerald-500">Running</p>
            </div>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 glass rounded-xl overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-900/50">
                    <h3 class="font-bold text-sm">Trade Log</h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="bg-slate-900/30 text-slate-500 text-[10px] uppercase">
                            <tr>
                                <th class="px-6 py-3">Time</th>
                                <th class="px-6 py-3">Symbol</th>
                                <th class="px-6 py-3">Direction</th>
                                <th class="px-6 py-3">Status</th>
                                <th class="px-6 py-3 text-right">P/L</th>
                            </tr>
                        </thead>
                        <tbody id="trade-body"></tbody>
                    </table>
                </div>
            </div>
            <div class="space-y-6">
                <div class="glass p-6 rounded-xl">
                    <h3 class="text-amber-500 text-xs font-bold uppercase mb-4">Recent Skips</h3>
                    <div id="skip-log" class="space-y-3 max-h-96 overflow-y-auto text-xs">
                        <p class="text-slate-600 italic">No skips recorded yet.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function updateDashboard() {
            try {
                const [tradesRes, sessionRes] = await Promise.all([
                    fetch('/trades'),
                    fetch('/session')
                ]);
                const tradesData = await tradesRes.json();
                const session = await sessionRes.json();
                
                document.getElementById('stat-total').innerText = tradesData.total_trades;
                document.getElementById('stat-skipped').innerText = tradesData.skipped_trades;
                document.getElementById('stat-winrate').innerText = tradesData.win_rate;
                document.getElementById('session-pl').innerText = `$${tradesData.total_profit_loss}`;
                document.getElementById('session-pl').className = `text-2xl font-bold ${tradesData.total_profit_loss >= 0 ? 'text-emerald-500' : 'text-red-500'}`;
                document.getElementById('sync-status').innerText = `Last update: ${new Date().toLocaleTimeString()}`;
                
                const tbody = document.getElementById('trade-body');
                const skipLog = document.getElementById('skip-log');
                tbody.innerHTML = '';
                skipLog.innerHTML = '';
                
                Object.values(tradesData.trades).reverse().forEach(t => {
                    const profit = parseFloat(t.profit || 0);
                    const status = t.status === 'completed' ? (profit > 0 ? 'win' : 'loss') : 
                                 t.status === 'skipped' ? 'skipped' : t.status;
                    const statusClass = status === 'win' ? 'win' : status === 'loss' ? 'loss' : 
                                      status === 'skipped' ? 'skipped' : 'running';
                    
                    if (t.status === 'skipped' || t.status === 'error') {
                        skipLog.innerHTML += `<div class="p-2 bg-amber-500/10 rounded">
                            <span class="text-xs text-slate-500">${t.timestamp.split('T')[1].split('.')[0]}</span>
                            <p class="text-amber-300">${t.error || 'No signal'}</p>
                        </div>`;
                    } else {
                        tbody.innerHTML += `
                            <tr class="hover:bg-slate-800/20">
                                <td class="px-6 py-4 text-xs text-slate-500">${t.timestamp.split('T')[1].split('.')[0]}</td>
                                <td class="px-6 py-4">${t.parameters?.symbol || 'R_100'}</td>
                                <td class="px-6 py-4">${t.direction || '-'}</td>
                                <td class="px-6 py-4"><span class="status-badge ${statusClass}">${status}</span></td>
                                <td class="px-6 py-4 text-right font-bold ${profit > 0 ? 'text-emerald-500' : 'text-red-500'}">
                                    ${t.status === 'running' ? '...' : '$' + profit.toFixed(2)}
                                </td>
                            </tr>`;
                    }
                });
            } catch (e) { console.error(e); }
        }
        setInterval(updateDashboard, 8000);
        updateDashboard();
    </script>
</body>
</html>
"""
    return html, 200

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Deriv Multiplier Bot v1.1 - Enhanced Logging & Reliability")
    logger.info("Trend-following strategy with EMA + ATR on volatility indices")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Server starting on port {port}")
    logger.info("Ready for cron triggers every 10-15 minutes")
    
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False, threaded=True)