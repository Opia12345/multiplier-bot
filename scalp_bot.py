import MetaTrader5 as mt5
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scalping_bot_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class BotPosition:
    ticket: int
    instrument: str
    type: PositionType
    entry_price: float
    volume: float
    tp_price: float
    sl_price: float
    open_time: datetime
    spread_at_entry: float
    current_price: float = 0.0
    pnl: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0   # For trailing stop
    trailing_activated: bool = False

@dataclass
class MarketCondition:
    symbol: str
    spread: float
    volatility: float
    atr: float
    trend_strength: float
    is_tradeable: bool
    rejection_reason: str = ""

class MT5ScalpingBot:
    def __init__(self, config: Dict):
        self.config = config
        self.positions: List[BotPosition] = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        
        # Performance tracking
        self.drawdown = 0.0
        self.max_equity = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Spread history (in pips)
        self.spread_history = {symbol: [] for symbol in config['symbols']}
        
        # Symbol-specific points per pip
        self.points_per_pip = config.get('points_per_pip', {'default': 10})
        
        # Volatility configuration per symbol
        self.volatility_config = config.get('volatility_limits', {
            'XAUUSDm': {'min': 0.08, 'max': 0.5},
            'BTCUSDm': {'min': 0.10, 'max': 0.6},
            'default': {'min': 0.05, 'max': 0.4}
        })
        
        # Initialize MT5
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            quit()
        
        # Login
        if not mt5.login(
            login=config['account'],
            password=config['password'],
            server=config['server']
        ):
            logging.error(f"Login failed: {mt5.last_error()}")
            mt5.shutdown()
            quit()
        
        account_info = mt5.account_info()
        self.initial_balance = account_info.balance
        self.max_equity = account_info.equity
        
        logging.info(f"Connected to MT5 Account: {config['account']}")
        logging.info(f"Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
        logging.info(f"Broker: {account_info.company} | Leverage: 1:{account_info.leverage}")
    
    def get_points_per_pip(self, symbol: str) -> int:
        return self.points_per_pip.get(symbol, self.points_per_pip.get('default', 10))
    
    def get_symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.warning(f"Symbol {symbol} not found")
            return None
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logging.warning(f"Failed to select {symbol}")
                return None
        return info
    
    def get_current_spread(self, symbol: str) -> float:
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = self.get_symbol_info(symbol)
        
        if tick is None or symbol_info is None:
            return 999.0
        
        spread_points = (tick.ask - tick.bid)
        spread_pips = spread_points / symbol_info.point / self.get_points_per_pip(symbol)
        
        self.spread_history[symbol].append(spread_pips)
        if len(self.spread_history[symbol]) > 100:
            self.spread_history[symbol] = self.spread_history[symbol][-100:]
        
        return spread_pips
    
    def get_average_spread(self, symbol: str) -> float:
        if not self.spread_history[symbol]:
            return 0.0
        return np.mean(self.spread_history[symbol])
    
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return 0.0
        
        df = pd.DataFrame(rates)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr
    
    def calculate_volatility(self, symbol: str) -> float:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 30)
        if rates is None or len(rates) < 20:
            return 0.0
        
        df = pd.DataFrame(rates)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100
        return volatility
    
    def is_good_trading_time(self) -> bool:
        """Check if current time is good for trading"""
        if not self.config.get('enable_time_filter', False):
            return True
        
        current_hour = datetime.utcnow().hour
        
        # Avoid dead zones: 22:00-01:00 UTC (NY close to Asian open)
        # and 12:00-13:00 UTC (Asian close before London)
        dead_zones = [(22, 1), (12, 13)]
        
        for start, end in dead_zones:
            if start < end:
                if start <= current_hour < end:
                    return False
            else:  # Wraps midnight
                if current_hour >= start or current_hour < end:
                    return False
        
        return True
    
    def analyze_market_condition(self, symbol: str) -> MarketCondition:
        spread = self.get_current_spread(symbol)
        volatility = self.calculate_volatility(symbol)
        atr = self.calculate_atr(symbol)
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)
        if rates is None or len(rates) < 50:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=0.0, is_tradeable=False, rejection_reason="Insufficient data")
        
        df = pd.DataFrame(rates)
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # Calculate trend strength with alignment check
        if df['ema8'].iloc[-1] > df['ema21'].iloc[-1] > df['ema50'].iloc[-1]:
            # Calculate strength based on separation
            sep1 = (df['ema8'].iloc[-1] - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]
            sep2 = (df['ema21'].iloc[-1] - df['ema50'].iloc[-1]) / df['ema50'].iloc[-1]
            trend_score = min(1.0, (sep1 + sep2) * 100)
        elif df['ema8'].iloc[-1] < df['ema21'].iloc[-1] < df['ema50'].iloc[-1]:
            sep1 = (df['ema21'].iloc[-1] - df['ema8'].iloc[-1]) / df['ema21'].iloc[-1]
            sep2 = (df['ema50'].iloc[-1] - df['ema21'].iloc[-1]) / df['ema50'].iloc[-1]
            trend_score = -min(1.0, (sep1 + sep2) * 100)
        else:
            trend_score = 0.0
        
        # Symbol-specific spread check
        max_spread = self.config.get('max_spread_pips', {}).get(symbol, self.config.get('max_spread_pips_default', 15))
        avg_spread = self.get_average_spread(symbol)
        max_acceptable_spread = self.config['max_spread_multiplier'] * (avg_spread if avg_spread > 0 else max_spread)
        
        if spread > min(max_acceptable_spread, max_spread):
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Spread too high: {spread:.1f}p")
        
        # Symbol-specific volatility limits
        vol_limits = self.volatility_config.get(symbol, self.volatility_config['default'])
        
        if volatility < vol_limits['min']:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Volatility too low: {volatility:.4f}%")
        
        if volatility > vol_limits['max']:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Volatility too high: {volatility:.4f}%")
        
        if abs(trend_score) < 0.3:  # Reduced from requiring full 1.0 alignment
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason="Weak trend alignment")
        
        # Check time filter
        if not self.is_good_trading_time():
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason="Outside trading hours")
        
        return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                              trend_strength=trend_score, is_tradeable=True)
    
    def calculate_advanced_signals(self, symbol: str, market_condition: MarketCondition) -> tuple:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) < 100:
            return False, False
        
        df = pd.DataFrame(rates)
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        
        # Simplified signals - less restrictive for more opportunities
        long_signal = (
            market_condition.trend_strength > 0.3 and
            current_price > df['ema8'].iloc[-1] and
            current_price > df['bb_middle'].iloc[-1] and
            rsi > 45 and rsi < 70 and
            macd_hist > 0
        )
        
        short_signal = (
            market_condition.trend_strength < -0.3 and
            current_price < df['ema8'].iloc[-1] and
            current_price < df['bb_middle'].iloc[-1] and
            rsi < 55 and rsi > 30 and
            macd_hist < 0
        )
        
        return long_signal, short_signal
    
    def calculate_dynamic_tp_sl(self, symbol: str, market_condition: MarketCondition, position_type: PositionType) -> tuple:
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return self.config['min_tp_pips'], self.config['min_sl_pips']
        
        ppp = self.get_points_per_pip(symbol)
        atr_pips = market_condition.atr / symbol_info.point / ppp
        
        abs_trend = abs(market_condition.trend_strength)
        
        # Adjusted multipliers for better R:R
        tp_multiplier = 2.0 + abs_trend * 0.8
        sl_multiplier = 1.0 + abs_trend * 0.3
        
        tp_pips = max(self.config['min_tp_pips'], min(self.config['max_tp_pips'], atr_pips * tp_multiplier))
        sl_pips = max(self.config['min_sl_pips'], min(self.config['max_sl_pips'], atr_pips * sl_multiplier))
        
        # Ensure minimum risk-reward ratio
        if tp_pips / sl_pips < self.config['min_risk_reward']:
            tp_pips = sl_pips * self.config['min_risk_reward']
        
        return round(tp_pips, 1), round(sl_pips, 1)
    
    def calculate_optimal_lot_size(self, symbol: str, sl_pips: float) -> float:
        account_info = mt5.account_info()
        symbol_info = self.get_symbol_info(symbol)
        
        if account_info is None or symbol_info is None or sl_pips <= 0:
            return self.config['min_lot']
        
        # Reduce risk after consecutive losses
        risk_percent = self.config['risk_percent']
        if self.consecutive_losses >= 3:
            risk_percent = risk_percent * 0.5
            logging.info(f"Risk reduced to {risk_percent}% due to {self.consecutive_losses} consecutive losses")
        
        risk_amount = account_info.equity * (risk_percent / 100)
        ppp = self.get_points_per_pip(symbol)
        
        pip_value_per_lot = symbol_info.trade_tick_value * ppp
        
        if pip_value_per_lot <= 0:
            return self.config['min_lot']
        
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        lot_size = max(lot_size, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        lot_size = max(lot_size, self.config['min_lot'])
        lot_size = min(lot_size, self.config['max_lot'])
        
        step = symbol_info.volume_step
        if step > 0:
            lot_size = round(lot_size / step) * step
        
        return round(lot_size, 2)
    
    def open_position(self, symbol: str, position_type: PositionType, market_condition: MarketCondition) -> Optional[int]:
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        tp_pips, sl_pips = self.calculate_dynamic_tp_sl(symbol, market_condition, position_type)
        lot_size = self.calculate_optimal_lot_size(symbol, sl_pips)
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"No tick data for {symbol}")
            return None
        
        point = symbol_info.point
        ppp = self.get_points_per_pip(symbol)
        
        if position_type == PositionType.LONG:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
            sl = price - sl_pips * ppp * point
            tp = price + tp_pips * ppp * point
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
            sl = price + sl_pips * ppp * point
            tp = price - tp_pips * ppp * point
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config['max_slippage_points'],
            "magic": self.config['magic_number'],
            "comment": f"ScalpBot_RR{tp_pips/sl_pips:.1f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.warning(f"Order failed {symbol} {position_type.value}: {result.retcode if result else 'None'} - {result.comment if result else 'No result'}")
            return None
        
        position = BotPosition(
            ticket=result.order,
            instrument=symbol,
            type=position_type,
            entry_price=result.price,
            volume=lot_size,
            tp_price=tp,
            sl_price=sl,
            open_time=datetime.now(),
            spread_at_entry=market_condition.spread,
            highest_price=result.price if position_type == PositionType.LONG else 0,
            lowest_price=result.price if position_type == PositionType.SHORT else float('inf')
        )
        self.positions.append(position)
        self.daily_trades += 1
        
        logging.info(f"[OPEN] {position_type.value} {symbol} | Lot: {lot_size:.2f} | Price: {result.price:.5f} | "
                     f"SL: {sl:.5f} ({sl_pips:.1f}p) | TP: {tp:.5f} ({tp_pips:.1f}p) | RR: {tp_pips/sl_pips:.2f}")
        
        return result.order
    
    def apply_trailing_stop(self, position: BotPosition):
        """Apply trailing stop to protect profits"""
        symbol_info = self.get_symbol_info(position.instrument)
        if not symbol_info:
            return
        
        tick = mt5.symbol_info_tick(position.instrument)
        if not tick:
            return
        
        ppp = self.get_points_per_pip(position.instrument)
        trailing_distance_pips = self.config.get('trailing_stop_pips', 20)
        activation_pips = self.config.get('trailing_activation_pips', 25)
        
        if position.type == PositionType.LONG:
            current_price = tick.bid
            position.highest_price = max(position.highest_price, current_price)
            
            profit_pips = (current_price - position.entry_price) / symbol_info.point / ppp
            
            # Activate trailing after reaching activation threshold
            if profit_pips >= activation_pips:
                new_sl = position.highest_price - (trailing_distance_pips * ppp * symbol_info.point)
                
                if new_sl > position.sl_price:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp_price
                    }
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        if not position.trailing_activated:
                            logging.info(f"[TRAIL ACTIVATED] #{position.ticket} at {profit_pips:.1f} pips profit")
                            position.trailing_activated = True
                        position.sl_price = new_sl
        
        else:  # SHORT
            current_price = tick.ask
            position.lowest_price = min(position.lowest_price, current_price)
            
            profit_pips = (position.entry_price - current_price) / symbol_info.point / ppp
            
            if profit_pips >= activation_pips:
                new_sl = position.lowest_price + (trailing_distance_pips * ppp * symbol_info.point)
                
                if new_sl < position.sl_price:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp_price
                    }
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        if not position.trailing_activated:
                            logging.info(f"[TRAIL ACTIVATED] #{position.ticket} at {profit_pips:.1f} pips profit")
                            position.trailing_activated = True
                        position.sl_price = new_sl
    
    def close_position(self, ticket: int, reason: str = "Manual close"):
        """Manually close a position"""
        for pos in self.positions:
            if pos.ticket == ticket:
                symbol_info = self.get_symbol_info(pos.instrument)
                if not symbol_info:
                    return
                
                tick = mt5.symbol_info_tick(pos.instrument)
                if not tick:
                    return
                
                if pos.type == PositionType.LONG:
                    price = tick.bid
                    order_type = mt5.ORDER_TYPE_SELL
                else:
                    price = tick.ask
                    order_type = mt5.ORDER_TYPE_BUY
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": pos.instrument,
                    "volume": pos.volume,
                    "type": order_type,
                    "price": price,
                    "deviation": self.config['max_slippage_points'],
                    "magic": self.config['magic_number'],
                    "comment": f"Close: {reason}",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"[CLOSED] #{ticket} - {reason}")
                break
    
    def update_positions(self):
        mt5_positions = mt5.positions_get(magic=self.config['magic_number'])
        if mt5_positions is None:
            mt5_positions = []
        
        mt5_tickets = {pos.ticket for pos in mt5_positions}
        
        closed = [pos for pos in self.positions if pos.ticket not in mt5_tickets]
        for pos in closed:
            deals = mt5.history_deals_get(position=pos.ticket)
            profit = sum(deal.profit for deal in deals) if deals else 0
            self.total_pnl += profit
            self.daily_pnl += profit
            self.total_trades += 1
            
            if profit > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
            duration = (datetime.now() - pos.open_time).total_seconds() / 60
            outcome = "WIN" if profit > 0 else "LOSS"
            logging.info(f"[CLOSE-{outcome}] {pos.instrument} #{pos.ticket} | P&L: ${profit:+.2f} | Dur: {duration:.1f}min")
            self.positions.remove(pos)
        
        # Update current positions
        for pos in self.positions:
            for mt5_pos in mt5_positions:
                if mt5_pos.ticket == pos.ticket:
                    pos.current_price = mt5_pos.price_current
                    pos.pnl = mt5_pos.profit
                    break
        
        # Check for positions exceeding max hold time
        max_hold_minutes = self.config.get('max_hold_time_minutes', 120)
        for pos in self.positions[:]:  # Copy list to avoid modification during iteration
            hold_time = (datetime.now() - pos.open_time).total_seconds() / 60
            if hold_time > max_hold_minutes:
                self.close_position(pos.ticket, f"Max hold time {max_hold_minutes}min exceeded")
    
    def check_daily_limits(self) -> bool:
        if datetime.now().date() != self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = datetime.now().date()
            self.consecutive_losses = 0  # Reset daily
            logging.info("=== NEW TRADING DAY - Counters Reset ===")
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        if self.daily_pnl <= -self.config['max_daily_loss']:
            logging.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        if self.config.get('daily_profit_target') and self.daily_pnl >= self.config['daily_profit_target']:
            logging.info(f"Daily profit target reached: ${self.daily_pnl:.2f}")
            return False
        return True
    
    def check_drawdown(self) -> bool:
        account_info = mt5.account_info()
        if account_info.equity > self.max_equity:
            self.max_equity = account_info.equity
        self.drawdown = (self.max_equity - account_info.equity) / self.max_equity * 100 if self.max_equity > 0 else 0
        if self.drawdown >= self.config['max_drawdown_percent']:
            logging.error(f"Max drawdown reached: {self.drawdown:.2f}%")
            return False
        return True
    
    def can_open_position(self, symbol: str) -> bool:
        if not self.check_daily_limits() or not self.check_drawdown():
            return False
        if len(self.positions) >= self.config['max_positions']:
            return False
        if len([p for p in self.positions if p.instrument == symbol]) >= self.config['max_positions_per_symbol']:
            return False
        
        account_info = mt5.account_info()
        if account_info.equity < self.config['min_equity']:
            return False
        
        # Stop trading after too many consecutive losses
        if self.consecutive_losses >= self.config.get('max_consecutive_losses', 5):
            logging.warning(f"Paused trading: {self.consecutive_losses} consecutive losses")
            return False
        
        return True
    
    def display_status(self):
        account_info = mt5.account_info()
        print("\n" + "="*100)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Balance: ${account_info.balance:.2f} | "
              f"Equity: ${account_info.equity:.2f} | Drawdown: {self.drawdown:.2f}%")
        print(f"Open: {len(self.positions)}/{self.config['max_positions']} | "
              f"Today: {self.daily_trades} trades, ${self.daily_pnl:+.2f} | "
              f"Consecutive Losses: {self.consecutive_losses}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_win = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
            print(f"Total: {self.total_trades} trades | Win Rate: {win_rate:.1f}% | "
                  f"Total P&L: ${self.total_pnl:+.2f} | Avg/Trade: ${avg_win:+.2f}")
        if self.positions:
            unrealized = sum(p.pnl for p in self.positions)
            print(f"\nOpen Positions (Unrealized: ${unrealized:+.2f}):")
            for pos in self.positions:
                trail_status = "TRAILING" if pos.trailing_activated else ""
                print(f"  #{pos.ticket} {pos.type.value} {pos.instrument} {trail_status} | Lot: {pos.volume:.2f} | "
                      f"Entry: {pos.entry_price:.5f} | P&L: ${pos.pnl:+.2f}")
        print("="*100)
    
    def run(self):
        logging.info("\n" + "="*100)
        logging.info("IMPROVED SCALPING BOT STARTED - ENHANCED VERSION")
        logging.info("="*100)
        
        iteration = 0
        last_status = datetime.now()
        
        try:
            while True:
                iteration += 1
                self.update_positions()
                
                # Apply trailing stops to all open positions
                for pos in self.positions:
                    self.apply_trailing_stop(pos)
                
                # Look for new trade opportunities
                for symbol in self.config['symbols']:
                    if not self.can_open_position(symbol):
                        continue
                    
                    condition = self.analyze_market_condition(symbol)
                    if not condition.is_tradeable:
                        if iteration % 20 == 0:
                            logging.info(f"{symbol}: REJECTED - {condition.rejection_reason}")
                        continue
                    
                    if iteration % 20 == 0:
                        logging.info(f"{symbol}: TRADEABLE | Spread: {condition.spread:.1f}p | "
                                   f"Vol: {condition.volatility:.4f}% | Trend: {condition.trend_strength:.3f}")
                    
                    long_sig, short_sig = self.calculate_advanced_signals(symbol, condition)
                    
                    if long_sig:
                        self.open_position(symbol, PositionType.LONG, condition)
                        time.sleep(0.5)
                    if short_sig:
                        self.open_position(symbol, PositionType.SHORT, condition)
                        time.sleep(0.5)
                
                # Display status periodically
                if (datetime.now() - last_status).total_seconds() >= 30:
                    self.display_status()
                    last_status = datetime.now()
                
                time.sleep(self.config['sleep_interval'])
                
        except KeyboardInterrupt:
            logging.info("\nStopped by user")
        except Exception as e:
            logging.error(f"Critical error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.display_status()
        account_info = mt5.account_info()
        print("\n" + "="*100)
        print("FINAL STATISTICS")
        print("="*100)
        net_pnl = account_info.equity - self.initial_balance
        print(f"Net P&L: ${net_pnl:+.2f} ({(net_pnl/self.initial_balance)*100:+.2f}%)")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_win = self.total_pnl / self.total_trades
            print(f"Total Trades: {self.total_trades} | Wins: {self.winning_trades} | Losses: {self.total_trades - self.winning_trades}")
            print(f"Win Rate: {win_rate:.1f}% | Avg P&L per Trade: ${avg_win:+.2f}")
            print(f"Max Consecutive Losses: {self.max_consecutive_losses}")
        print("="*100)
        mt5.shutdown()
        logging.info("Shutdown complete")

# ============= IMPROVED CONFIGURATION =============
config = {
    # Account credentials
    'account': 297846595,
    'password': 'Killindem22.',
    'server': 'Exness-MT5Trial9',
    
    # Trading instruments
    'symbols': ['XAUUSDm', 'BTCUSDm'],
    
    # Position limits
    'max_positions': 4,
    'max_positions_per_symbol': 1,
    'magic_number': 234567,
    
    # Risk management - CRITICAL IMPROVEMENTS
    'risk_percent': 0.5,  # Increased from 0.3 (better R:R allows this)
    'min_lot': 0.01,
    'max_lot': 0.05,
    'min_risk_reward': 1.5,  # FIXED: Was 0.5 (losing setup), now 1.5:1 minimum
    
    # Take Profit / Stop Loss - OPTIMIZED
    'min_tp_pips': 25,  # Increased from 15
    'max_tp_pips': 80,  # Increased from 60
    'min_sl_pips': 15,  # Decreased from 35 (tighter stops)
    'max_sl_pips': 50,  # Decreased from 120
    
    # Trailing stop configuration - NEW
    'trailing_stop_pips': 20,
    'trailing_activation_pips': 25,
    
    # Spread control - TIGHTER
    'max_spread_pips': {
        'XAUUSDm': 15,  # Symbol-specific
        'BTCUSDm': 20,
    },
    'max_spread_pips_default': 15,  # Tightened from 25
    'max_spread_multiplier': 1.3,  # Reduced from 1.5
    
    # Volatility limits - SYMBOL-SPECIFIC
    'volatility_limits': {
        'XAUUSDm': {'min': 0.08, 'max': 0.5},
        'BTCUSDm': {'min': 0.10, 'max': 0.6},
        'default': {'min': 0.05, 'max': 0.4}
    },
    
    # Time filtering - NOW ENABLED
    'enable_time_filter': True,
    
    # Daily limits - ADJUSTED
    'max_daily_trades': 15,  # Increased from 10
    'max_daily_loss': 50,  # Increased from 30
    'daily_profit_target': 100,  # New: stop after hitting target
    
    # Safety limits - ENHANCED
    'min_equity': 100,
    'max_drawdown_percent': 10,
    'max_slippage_points': 30,
    'max_consecutive_losses': 5,  # New: pause after 5 losses
    'max_hold_time_minutes': 120,  # New: max 2 hours per position
    
    # Bot behavior
    'sleep_interval': 4,
    
    # Points per pip (symbol-specific)
    'points_per_pip': {
        'default': 10,
        'XAUUSDm': 1,
        'BTCUSDm': 10
    },
}

# ============= RUN BOT =============
if __name__ == "__main__":
    print("="*100)
    print("IMPROVED MT5 SCALPING BOT - VERSION 2.0")
    print("="*100)
    print("\nKEY IMPROVEMENTS:")
    print("✓ Fixed Risk-Reward Ratio: 1.5:1 minimum (was 0.5:1)")
    print("✓ Trailing Stop System: Locks in profits automatically")
    print("✓ Time Filtering: Avoids low-liquidity periods")
    print("✓ Better Position Management: Max hold time, consecutive loss protection")
    print("✓ Optimized TP/SL: Tighter stops, larger targets")
    print("✓ Symbol-Specific Settings: Tailored for GOLD and BTC")
    print("\n" + "="*100)
    print("WARNING: This will trade on your Exness account!")
    print("="*100)
    
    response = input("\nType 'START' to launch bot: ").strip().upper()
    if response == 'START':
        print("\n🚀 Launching improved bot...\n")
        bot = MT5ScalpingBot(config)
        bot.run()
    else:
        print("Bot not started.")