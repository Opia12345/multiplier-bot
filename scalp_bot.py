import MetaTrader5 as mt5
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import logging

# Setup logging (safe for Windows console - no emojis in logs)
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
        
        # Spread history (in pips)
        self.spread_history = {symbol: [] for symbol in config['symbols']}
        
        # Symbol-specific points per pip (Exness tuning)
        self.points_per_pip = config.get('points_per_pip', {'default': 10})
        
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
        
        # Require strict full alignment for tradeable
        if df['ema8'].iloc[-1] > df['ema21'].iloc[-1] > df['ema50'].iloc[-1]:
            trend_score = 1.0
        elif df['ema8'].iloc[-1] < df['ema21'].iloc[-1] < df['ema50'].iloc[-1]:
            trend_score = -1.0
        else:
            trend_score = 0.0
        
        avg_spread = self.get_average_spread(symbol)
        max_acceptable_spread = self.config['max_spread_multiplier'] * (avg_spread if avg_spread > 0 else 20)
        
        if spread > max_acceptable_spread or spread > self.config['max_spread_pips']:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Spread too high: {spread:.1f}p")
        
        if volatility < self.config['min_volatility']:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Volatility too low: {volatility:.4f}%")
        
        if volatility > self.config['max_volatility']:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason=f"Volatility too high: {volatility:.4f}%")
        
        if abs(trend_score) < 1.0:
            return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                                  trend_strength=trend_score, is_tradeable=False,
                                  rejection_reason="Weak or no trend alignment")
        
        return MarketCondition(symbol=symbol, spread=spread, volatility=volatility, atr=atr,
                              trend_strength=trend_score, is_tradeable=True)
    
    def calculate_advanced_signals(self, symbol: str, market_condition: MarketCondition) -> tuple:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) < 100:
            return False, False
        
        df = pd.DataFrame(rates)
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
        volume_confirmation = df['tick_volume'].iloc[-1] > df['volume_ma'].iloc[-1] * 1.5
        
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        macd_hist_prev = df['macd_hist'].iloc[-2]
        
        long_signal = (
            market_condition.trend_strength == 1.0 and
            current_price > df['ema8'].iloc[-1] and
            current_price > df['bb_middle'].iloc[-1] and
            rsi > 50 and rsi < 70 and
            macd_hist > 0 and macd_hist > macd_hist_prev and
            volume_confirmation
        )
        
        short_signal = (
            market_condition.trend_strength == -1.0 and
            current_price < df['ema8'].iloc[-1] and
            current_price < df['bb_middle'].iloc[-1] and
            rsi < 50 and rsi > 30 and
            macd_hist < 0 and macd_hist < macd_hist_prev and
            volume_confirmation
        )
        
        return long_signal, short_signal
    
    def calculate_dynamic_tp_sl(self, symbol: str, market_condition: MarketCondition, position_type: PositionType) -> tuple:
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return self.config['min_tp_pips'], self.config['min_sl_pips']
        
        ppp = self.get_points_per_pip(symbol)
        atr_pips = market_condition.atr / symbol_info.point / ppp
        
        abs_trend = abs(market_condition.trend_strength)
        
        tp_multiplier = 1.2 + abs_trend * 0.5
        sl_multiplier = 2.5 + abs_trend * 0.8
        
        tp_pips = max(self.config['min_tp_pips'], min(self.config['max_tp_pips'], atr_pips * tp_multiplier))
        sl_pips = max(self.config['min_sl_pips'], min(self.config['max_sl_pips'], atr_pips * sl_multiplier))
        
        if tp_pips / sl_pips < self.config['min_risk_reward']:
            tp_pips = sl_pips * self.config['min_risk_reward']
        
        return round(tp_pips, 1), round(sl_pips, 1)
    
    def calculate_optimal_lot_size(self, symbol: str, sl_pips: float) -> float:
        account_info = mt5.account_info()
        symbol_info = self.get_symbol_info(symbol)
        
        if account_info is None or symbol_info is None or sl_pips <= 0:
            return self.config['min_lot']
        
        risk_amount = account_info.equity * (self.config['risk_percent'] / 100)
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
            spread_at_entry=market_condition.spread
        )
        self.positions.append(position)
        self.daily_trades += 1
        
        logging.info(f"[OPEN] {position_type.value} {symbol} | Lot: {lot_size:.2f} | Price: {result.price:.5f} | "
                     f"SL: {sl:.5f} ({sl_pips:.1f}p) | TP: {tp:.5f} ({tp_pips:.1f}p) | RR: {tp_pips/sl_pips:.2f}")
        
        return result.order
    
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
            
            duration = (datetime.now() - pos.open_time).total_seconds() / 60
            outcome = "WIN" if profit > 0 else "LOSS"
            logging.info(f"[CLOSE-{outcome}] {pos.instrument} #{pos.ticket} | P&L: ${profit:+.2f} | Dur: {duration:.1f}min")
            self.positions.remove(pos)
        
        for pos in self.positions:
            for mt5_pos in mt5_positions:
                if mt5_pos.ticket == pos.ticket:
                    pos.current_price = mt5_pos.price_current
                    pos.pnl = mt5_pos.profit
                    break
    
    def check_daily_limits(self) -> bool:
        if datetime.now().date() != self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = datetime.now().date()
            logging.info("New day - daily counters reset")
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        if self.daily_pnl <= -self.config['max_daily_loss']:
            return False
        if self.config.get('daily_profit_target') and self.daily_pnl >= self.config['daily_profit_target']:
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
        return True
    
    def display_status(self):
        account_info = mt5.account_info()
        print("\n" + "="*100)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Balance: ${account_info.balance:.2f} | "
              f"Equity: ${account_info.equity:.2f} | Drawdown: {self.drawdown:.2f}%")
        print(f"Open: {len(self.positions)}/{self.config['max_positions']} | Today: {self.daily_trades} trades, ${self.daily_pnl:+.2f}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"Total: {self.total_trades} trades | Win Rate: {win_rate:.1f}% | Total P&L: ${self.total_pnl:+.2f}")
        if self.positions:
            unrealized = sum(p.pnl for p in self.positions)
            print(f"\nOpen Positions (Unrealized: ${unrealized:+.2f}):")
            for pos in self.positions:
                print(f"  #{pos.ticket} {pos.type.value} {pos.instrument} | Lot: {pos.volume:.2f} | "
                      f"Entry: {pos.entry_price:.5f} | P&L: ${pos.pnl:+.2f}")
        print("="*100)
    
    def run(self):
        logging.info("\n" + "="*100)
        logging.info("SCALPING BOT STARTED - EXNESS ACCOUNT")
        logging.info("="*100)
        
        iteration = 0
        last_status = datetime.now()
        
        try:
            while True:
                iteration += 1
                self.update_positions()
                
                for symbol in self.config['symbols']:
                    if not self.can_open_position(symbol):
                        continue
                    
                    condition = self.analyze_market_condition(symbol)
                    if not condition.is_tradeable:
                        if iteration % 20 == 0:
                            logging.info(f"{symbol}: REJECTED - {condition.rejection_reason}")
                        continue
                    
                    if iteration % 20 == 0:
                        logging.info(f"{symbol}: TRADEABLE | Spread: {condition.spread:.1f}p | Trend: {condition.trend_strength:.3f}")
                    
                    long_sig, short_sig = self.calculate_advanced_signals(symbol, condition)
                    
                    if long_sig:
                        self.open_position(symbol, PositionType.LONG, condition)
                        time.sleep(0.5)
                    if short_sig:
                        self.open_position(symbol, PositionType.SHORT, condition)
                        time.sleep(0.5)
                
                if (datetime.now() - last_status).total_seconds() >= 30:
                    self.display_status()
                    last_status = datetime.now()
                
                time.sleep(self.config['sleep_interval'])
                
        except KeyboardInterrupt:
            logging.info("\nStopped by user")
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.display_status()
        account_info = mt5.account_info()
        print("\nFINAL STATS")
        print(f"Net P&L: ${account_info.equity - self.initial_balance:+.2f}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"Total Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")
        mt5.shutdown()
        logging.info("Shutdown complete")

# ============= CONFIGURATION (OPTIMIZED FOR HIGH WIN RATE) =============
config = {
    'account': 297846595,
    'password': 'Killindem22.',
    'server': 'Exness-MT5Trial9',
    
    'symbols': ['XAUUSDm', 'BTCUSDm'],
    
    'max_positions': 4,
    'max_positions_per_symbol': 1,
    'magic_number': 234567,
    
    'risk_percent': 0.3,
    'min_lot': 0.01,
    'max_lot': 0.05,
    'min_risk_reward': 0.5,
    
    'min_tp_pips': 15,
    'max_tp_pips': 60,
    'min_sl_pips': 35,
    'max_sl_pips': 120,
    
    'max_spread_pips': 25,
    'max_spread_multiplier': 1.5,
    
    'min_volatility': 0.05,
    'max_volatility': 0.4,
    
    'enable_time_filter': False,
    
    'max_daily_trades': 10,
    'max_daily_loss': 30,
    'daily_profit_target': None,
    
    'min_equity': 100,
    'max_drawdown_percent': 10,
    'max_slippage_points': 30,
    
    'sleep_interval': 4,
    
    'points_per_pip': {
        'default': 10,
        'XAUUSDm': 1,
        'BTCUSDm': 10
    },
}

# ============= RUN BOT =============
if __name__ == "__main__":
    print("="*100)
    print("EXNESS MT5 SCALPING BOT - OPTIMIZED FOR HIGH WIN RATE")
    print("="*100)
    print("WARNING: This will trade on your Exness account!")
    print("Fewer, higher-quality trades with small TP and wide SL")
    
    response = input("\nType 'START' to launch: ").strip().upper()
    if response == 'START':
        print("Launching bot...")
        bot = MT5ScalpingBot(config)
        bot.run()
    else:
        print("Bot not started.")
        
        # buy the dip, sell the peak