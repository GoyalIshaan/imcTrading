from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation
from typing import List, Dict, Tuple, Any
import jsonpickle
import math
import numpy as np
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # For Kelp we maintain a history of prices/VWAP values
        self.kelp_prices = []
        self.kelp_vwap = []
        self.last_ema = None  # Store the last EMA value
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.last_ema = None  # Store the last EMA value
        self.last_squid_ink_ema = None  # Store the last EMA value for SQUID_INK
        self.squid_ink_last_short_ema = None  # Short-term EMA (e.g., 5-period)
        self.squid_ink_last_long_ema = None   # Long-term EMA (e.g., 20-period)
        self.squid_ink_last_ema_values = []
        self.recent_squid_ink_pl = []
        
        # Set up squid ink ML logging
        self.squid_ink_log_file = "squid_ink_ml_data.csv"
        self.init_squid_ink_log()

        logger.print("Trader initialized with price tracking for KELP")

    def init_squid_ink_log(self):
        """Initialize the SQUID_INK ML data log file with headers"""
        try:
            with open(self.squid_ink_log_file, 'w') as f:
                headers = [
                    "timestamp", 
                    "best_bid", 
                    "best_ask", 
                    "mid_price", 
                    "vwap",
                    "current_ema", 
                    "short_ema", 
                    "long_ema", 
                    "trend_signal",
                    "dynamic_width",
                    "position", 
                    "orders_placed", 
                    "buy_volume", 
                    "sell_volume",
                    "order_book_imbalance",
                    "liquidity_score",
                    "momentum",
                    "recent_pl"
                ]
                f.write(",".join(headers) + "\n")
        except Exception as e:
            logger.print(f"Error initializing SQUID_INK log: {e}")
            
    def log_squid_ink_data(self, state, orders, timestamp, order_depth, 
                           current_ema=None, short_ema=None, long_ema=None, 
                           trend_signal=None, dynamic_width=None, buy_volume=0, sell_volume=0):
        """
        Log SQUID_INK trading data to a separate file for ML analysis
        """
        try:
            position = state.position.get("SQUID_INK", 0)
            
            # Calculate order book metrics
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
            
            # Calculate book imbalance
            buy_volume_total = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            sell_volume_total = abs(sum(order_depth.sell_orders.values())) if order_depth.sell_orders else 0
            total_volume = buy_volume_total + sell_volume_total
            order_book_imbalance = (buy_volume_total - sell_volume_total) / total_volume if total_volume > 0 else 0
            
            # Get liquidity score
            liquidity_score = self.assess_liquidity(order_depth)
            
            # Get VWAP if available
            vwap = 0
            if len(self.squid_ink_vwap) > 0:
                vwap = self.squid_ink_vwap[-1]["vwap"]
                
            # Calculate momentum
            momentum, _ = self.calculate_momentum()
            
            # Recent P&L
            recent_pl = sum(self.recent_squid_ink_pl[-5:]) / 5 if len(self.recent_squid_ink_pl) >= 5 else 0
            
            # Orders placed
            orders_placed = len(orders) if orders else 0
                
            # Prepare data row
            data = [
                timestamp,
                best_bid,
                best_ask,
                mid_price,
                vwap,
                current_ema if current_ema is not None else 0,
                short_ema if short_ema is not None else 0,
                long_ema if long_ema is not None else 0,
                trend_signal if trend_signal is not None else 0,
                dynamic_width if dynamic_width is not None else 0,
                position,
                orders_placed,
                buy_volume,
                sell_volume,
                order_book_imbalance,
                liquidity_score,
                momentum,
                recent_pl
            ]
            
            # Write to file
            with open(self.squid_ink_log_file, 'a') as f:
                f.write(",".join(map(str, data)) + "\n")
                
        except Exception as e:
            logger.print(f"Error logging SQUID_INK data: {e}")

    def calculate_recent_volatility(self, symbol):
        """
        Calculates normalized volatility based on recent price history.
        Returns a value where higher means more volatile.
        """
        if symbol == "KELP" and len(self.kelp_prices) >= 10:
            # Calculate rolling standard deviation of recent prices
            recent_prices = self.kelp_prices[-10:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            logger.print(f"Calculated volatility for {symbol}: {volatility:.4f}")
            return volatility
        elif symbol == "RAINFOREST_RESIN":
            # For stable assets, use a low default volatility
            return 0.001
        
        # Default fallback
        return 0.01

    def calculate_ema(self, prices, period):
        """
        Calculate the Exponential Moving Average (EMA) efficiently.
        Only calculates the most recent EMA value.
        """
        if not prices or len(prices) < 2:  # Need at least 2 prices
            return None
            
        alpha = 2 / (period + 1)
        
        # If we don't have a last EMA, initialize with first price
        if self.last_ema is None:
            self.last_ema = prices[0]
            
        # Calculate only the most recent EMA
        current_price = prices[-1]
        new_ema = alpha * current_price + (1 - alpha) * self.last_ema
        self.last_ema = new_ema
        
        logger.print(f"Calculated EMA: {new_ema:.2f} (period={period})")
        return new_ema

    def assess_liquidity(self, order_depth):
        """
        Evaluates current market liquidity based on order book depth.
        Returns a score between 0 (illiquid) and 1 (highly liquid).
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        
        # 1. Calculate total volume available within reasonable price range
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Look at volume within 1% of mid price
        price_range = mid_price * 0.01
        relevant_bids = {p: v for p, v in order_depth.buy_orders.items() 
                        if p >= mid_price - price_range}
        relevant_asks = {p: v for p, v in order_depth.sell_orders.items() 
                        if p <= mid_price + price_range}
        
        total_volume = sum(relevant_bids.values()) + sum(abs(v) for v in relevant_asks.values())
        
        # 2. Calculate order book imbalance
        buy_volume = sum(order_depth.buy_orders.values())
        sell_volume = sum(abs(v) for v in order_depth.sell_orders.values())
        total_market_volume = buy_volume + sell_volume
        
        # Lower score for imbalanced books
        balance_score = 1 - abs(buy_volume - sell_volume) / total_market_volume if total_market_volume > 0 else 0
        
        # 3. Consider spread as liquidity indicator
        spread_score = min(1.0, 1 / (best_ask - best_bid)) if best_ask > best_bid else 0
        
        # Calculate final liquidity score - weighted average of components
        liquidity_score = 0.5 * min(1.0, total_volume / 200) + 0.3 * balance_score + 0.2 * spread_score
        
        logger.print(f"Liquidity assessment - Total Volume: {total_volume}, Balance Score: {balance_score:.2f}, Spread Score: {spread_score:.2f}, Final Score: {liquidity_score:.2f}")
        return liquidity_score

    def calculate_dynamic_spread(self, symbol, order_depth, position, position_limit):
        """
        Calculates optimal spread width based on current market conditions.
        Narrows spreads during volatility to ensure execution.
        Widens spreads during stable conditions to maximize profit.
        """
        # Default base spread values
        base_spread = 2.0 if symbol == "RAINFOREST_RESIN" else 3.5
        
        # 1. Volatility component
        volatility = self.calculate_recent_volatility(symbol)
        
        # 2. Liquidity component - analyze order book depth
        liquidity_score = self.assess_liquidity(order_depth)
        
        # 3. Position component - adjust based on inventory
        position_ratio = abs(position) / position_limit
        inventory_factor = 1 + position_ratio * 0.5  # Widen as inventory grows
        
        # Compute the combined spread adjustment factor
        # Higher volatility → narrower spreads to ensure execution
        # Lower liquidity → wider spreads for protection
        volatility_factor = 1 - min(0.6, volatility * 0.2)  # Cap at 60% reduction
        liquidity_factor = 1 + max(0, (1 - liquidity_score)) * 0.5  # Up to 50% wider
        
        # Calculate final spread
        adjusted_spread = base_spread * volatility_factor * liquidity_factor * inventory_factor
        
        # Ensure minimum spread
        final_spread = max(1.0, adjusted_spread)
        
        logger.print(f"Spread calculation for {symbol} - Base: {base_spread:.2f}, Volatility Factor: {volatility_factor:.2f}, "
                    f"Liquidity Factor: {liquidity_factor:.2f}, Inventory Factor: {inventory_factor:.2f}, Final: {final_spread:.2f}")
        return final_spread

    def resin_orders(self, order_depth: OrderDepth, fair_value: float, width: int, position: int, position_limit: int) -> List[Order]:
        """
        For Rainforest Resin, which is stable, we use a simple market-making strategy.
        If the best ask is below our fair value, we buy. If the best bid is above, we sell.
        We then clear our position and post additional liquidity orders.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Identify prices for liquidity orders
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity
                    logger.print(f"RAINFOREST_RESIN: Buying {quantity} at {best_ask} (below fair value {fair_value})")

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_order_volume += quantity
                    logger.print(f"RAINFOREST_RESIN: Selling {quantity} at {best_bid} (above fair value {fair_value})")

        # Clear excess position if needed
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN",
            buy_order_volume, sell_order_volume, fair_value, width
        )

        # Place additional orders to make liquidity
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        """
        For Kelp, whose value fluctuates, we update a moving record of mid prices and a simple VWAP.
        We then use these to compute a dynamic fair value. Orders are issued if the market offers
        prices that deviate from our fair value by more than kelp_take_width.
        """
        dynamic_width = self.calculate_dynamic_spread("KELP", order_depth, position, position_limit)
        # Take width can be a fraction of the dynamic width
        dynamic_take_width = dynamic_width * 1  # Adjust this ratio as needed
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            # Use a liquidity filter (only orders with a minimum size) for the mid price calculation.
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >=18 ]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price])>= 18]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            mmmid_price = (mm_ask + mm_bid) / 2

            # Append the current mid price to our history
            self.kelp_prices.append(mmmid_price)
            
            # Compute a simple VWAP from the best bid/ask and their volumes
            volume = (-order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid])
            vwap = (best_bid * (-order_depth.sell_orders[best_ask]) + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})

            # Maintain history length
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)

            # Calculate VWAP-based fair value
            vwap_fair = sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / sum(x["vol"] for x in self.kelp_vwap)
            
            # Calculate EMA
            current_ema = self.calculate_ema(self.kelp_prices, timespan)
            if current_ema is None:
                current_ema = mmmid_price

            # Blend different price signals with EMA for a more robust fair value
            fair_value = (0.4 * mmmid_price +  # Current market mid price
                         0.15 * vwap_fair +     # Volume-weighted average price
                         0.4 * vwap +          # Current VWAP
                         0.05 * current_ema)    # Exponential moving average

            # Take liquidity if the market is offering a price away from our fair value
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
                        logger.print(f"KELP: Taking liquidity - buying {quantity} at {best_ask} (fair value: {fair_value:.2f})")

            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -quantity))
                        sell_order_volume += quantity
                        logger.print(f"KELP: Taking liquidity - selling {quantity} at {best_bid} (fair value: {fair_value:.2f})")

            # Clear any over- or under-sized positions using the helper method
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP",
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            
            # Prepare additional orders for providing liquidity.
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", int(bbbf) + 1, buy_quantity))
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", int(baaf) - 1, -sell_quantity))

        return orders
    
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
                             product: str, buy_order_volume: int, sell_order_volume: int,
                             fair_value: float, width: int):
        """
        Attempts to reduce the net position by matching at a price close to fair_value.
        It rounds the fair_value appropriately and issues orders to bring the net exposure
        closer to zero without exceeding the position limits.
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
                logger.print(f"{product}: Clearing {sent_quantity} long position at {fair_for_ask}")

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
                logger.print(f"{product}: Clearing {sent_quantity} short position at {fair_for_bid}")

        return buy_order_volume, sell_order_volume
    def calculate_risk_adjusted_size(self, symbol, base_size, position, position_limit):
       
        # Start with the base size from market microstructure
        
        # 1. Position adjustment - reduce size as position grows
        position_ratio = abs(position) / position_limit
        position_factor = 1 - (position_ratio * 0.7)  # Reduce by up to 70% at limit
        
        # 2. Volatility adjustment - reduce size in high volatility
        if symbol == "SQUID_INK" and len(self.squid_ink_prices) >= 10:
            recent_prices = self.squid_ink_prices[-10:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            vol_factor = max(0.5, 1 - (volatility * 5))  # Scale down in high volatility
        else:
            vol_factor = 0.9  # Default conservative factor
        
        # 3. Price confidence - trade more when confident in fair value
        if hasattr(self, 'fair_value_confidence'):
            confidence_factor = self.fair_value_confidence
        else:
            confidence_factor = 0.85  # Moderate confidence default
        
        # Calculate final size
        adjusted_size = base_size * position_factor * vol_factor * confidence_factor
        
        # Ensure reasonable size limits
        final_size = max(10, min(50, round(adjusted_size)))
        
        return final_size
    def squid_ink_orders(self, order_depth: OrderDepth, timespan: int, width: float, squid_ink_take_width: float, position: int, position_limit: int) -> List[Order]:
        """
        For SQUID_INK, whose value fluctuates, we update a moving record of mid prices and a simple VWAP.
        We then use these to compute a dynamic fair value. Orders are issued if the market offers
        prices that deviate from our fair value by more than squid_ink_take_width.
        """
        dynamic_width = self.calculate_dynamic_spread("SQUID_INK", order_depth, position, position_limit)
        # Take width can be a fraction of the dynamic width
        dynamic_take_width = dynamic_width * 1  # Adjust this ratio as needed
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        current_ema = None
        short_ema = None
        long_ema = None
        trend_signal = None

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            # Use a liquidity filter (only orders with a minimum size) for the mid price calculation.
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 30 ]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price])>= 30]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            #fairly liquid stock, see how might we adjust our fairvalue
            mmmid_price = (mm_ask + mm_bid) / 2

            # Append the current mid price to our history
            self.squid_ink_prices.append(mmmid_price)
            
            # Compute a simple VWAP from the best bid/ask and their volumes
            volume = (-order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid])
            vwap = (best_bid * (-order_depth.sell_orders[best_ask]) + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.squid_ink_vwap.append({"vol": volume, "vwap": vwap})

            # Maintain history length
            if len(self.squid_ink_vwap) > timespan:
                self.squid_ink_vwap.pop(0)
            if len(self.squid_ink_prices) > timespan:
                self.squid_ink_prices.pop(0)

            # Calculate VWAP-based fair value
            vwap_fair = sum(x["vwap"] * x["vol"] for x in self.squid_ink_vwap) / sum(x["vol"] for x in self.squid_ink_vwap)
            
            # Calculate EMA
            current_ema = self.calculate_squid_ink_ema(self.squid_ink_prices, timespan)
            short_ema, long_ema, trend_signal = self.calculate_squid_ink_dual_emas(self.squid_ink_prices)

            if current_ema is None:
                current_ema =  mmmid_price

            # Blend different price signals with EMA for a more robust fair value
            fair_value =  mmmid_price  # Current market mid price
                             # Exponential moving average

            # Take liquidity if the market is offering a price away from our fair value
            #potential paramterisation
            avg_recent_pl = sum(self.recent_squid_ink_pl[-5:]) / 5 if len(self.recent_squid_ink_pl) >= 5 else 0

            # If we're consistently profitable, increase size gradually
            if avg_recent_pl > 0:
                base_size = 40  # More aggressive
            else:
                base_size = 30  # More conservative
            optimal_size = self.calculate_risk_adjusted_size("SQUID_INK", base_size, position, position_limit)
        
            if best_ask <= fair_value - dynamic_take_width:
                #we are not placing orders for more than 27, see maybe if we can adjust this?(adjusted)
                
                ask_amount = -order_depth.sell_orders[best_ask]
                 
                quantity = min(ask_amount, position_limit - position, optimal_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    buy_order_volume += quantity
            if best_bid >= fair_value + dynamic_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                
                quantity = min(bid_amount, position_limit + position, optimal_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    sell_order_volume += quantity

            # Clear any over- or under-sized positions using the helper method

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "SQUID_INK",
                buy_order_volume, sell_order_volume, fair_value, dynamic_width
            )
            
            # Prepare additional orders for providing liquidity.
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + dynamic_width]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - dynamic_width]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("SQUID_INK", int(bbbf) + 1, buy_quantity))
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("SQUID_INK", int(baaf) - 1, -sell_quantity))

        return orders, current_ema, short_ema, long_ema, trend_signal, dynamic_width, buy_order_volume, sell_order_volume

    def calculate_squid_ink_ema(self, prices, period):
        """
        Calculate the Exponential Moving Average (EMA) efficiently for SQUID_INK.
        Only calculates the most recent EMA value.
        
        :param self: The trader instance
        :param prices: List of price values (floats or ints)
        :param period: The period for the EMA
        :return: The most recent EMA value, or None if not enough data
        """
        if not prices or len(prices) < 2:  # Need at least 2 prices
            return None
            
        alpha = 2 / (period + 1)
        
        # If we don't have a last EMA for SQUID_INK, initialize with first price
        if self.last_squid_ink_ema is None:
            self.last_squid_ink_ema = prices[0]
            
        # Calculate only the most recent EMA
        current_price = prices[-1]
        new_ema = alpha * current_price + (1 - alpha) * self.last_squid_ink_ema
        self.last_squid_ink_ema = new_ema
        
        return new_ema
    
    def calculate_squid_ink_dual_emas(self, prices, short_period=5, long_period=20):
        
        if not prices:
            return None, None, 0
        
        # Use available data even if less than ideal
        actual_short_period = min(short_period, len(prices))
        actual_long_period = min(long_period, len(prices))
        
        # Calculate alpha values with available data
        alpha_short = 2 / (actual_short_period + 1)
        alpha_long = 2 / (actual_long_period + 1)
        
        # Initialize EMAs if needed
        if self.squid_ink_last_short_ema is None:
            self.squid_ink_last_short_ema = prices[0]
        if self.squid_ink_last_long_ema is None:
            self.squid_ink_last_long_ema = prices[0]
        
        # Calculate current EMAs
        current_price = prices[-1]
        short_ema = alpha_short * current_price + (1 - alpha_short) * self.squid_ink_last_short_ema
        long_ema = alpha_long * current_price + (1 - alpha_long) * self.squid_ink_last_long_ema
        
        # Update stored EMAs
        self.squid_ink_last_short_ema = short_ema
        self.squid_ink_last_long_ema = long_ema
        
        # Store new EMA for momentum calculation
        self.squid_ink_last_ema_values.append(short_ema)
        if len(self.squid_ink_last_ema_values) > 5:
            self.squid_ink_last_ema_values.pop(0)
        
        # Determine trend signal
        if short_ema > long_ema:
            trend_signal = 1  # Uptrend
        elif short_ema < long_ema:
            trend_signal = -1  # Downtrend
        else:
            trend_signal = 0  # Neutral
        
        return short_ema, long_ema, trend_signal

    def calculate_momentum(self):
        """
        Calculate the momentum based on the rate of change of the EMA.
        
        :return: Momentum value and strength (0-1)
        """
        if len(self.squid_ink_last_ema_values) < 3:
            return 0, 0
        
        # Calculate rate of change over the last few periods
        ema_changes = [self.squid_ink_last_ema_values[i] - self.squid_ink_last_ema_values[i-1] 
                    for i in range(1, len(self.squid_ink_last_ema_values))]
        
        # Calculate average change
        avg_change = sum(ema_changes) / len(ema_changes)
        
        # Normalize against recent EMA value for comparable scale
        if self.squid_ink_last_ema_values[-1] != 0:
            normalized_momentum = avg_change / self.squid_ink_last_ema_values[-1] * 100
        else:
            normalized_momentum = 0
        
        # Determine momentum strength (0-1 scale)
        momentum_strength = min(1.0, abs(normalized_momentum) / 0.5)  # 0.5% change is considered full strength
        
        return normalized_momentum, momentum_strength

    def run(self, state: TradingState):
        """
        Main entry point of the algorithm.
        It generates orders for both RAINFOREST_RESIN and KELP based on their respective strategies.
        """
        result = {}
        logger.print(f"Processing timestamp: {state.timestamp}")
        logger.print(f"Current positions: {state.position}")

        # Parameters for RAINFOREST_RESIN (assumed stable)
        resin_fair_value = 10000  # A constant fair value; alternatively, you might use mid-price if desired.
        resin_width = 2
        resin_position_limit = 50

        # Parameters for KELP (volatile)
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10
        kelp_width = 3.5
        squid_ink_take_width = 1
        squid_ink_position_limit = 50
        squid_ink_timespan = 10
        squid_ink_width = 3.5

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_fair_value, resin_width, resin_position, resin_position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], kelp_timespan, kelp_width, kelp_take_width, kelp_position, kelp_position_limit
            )
            result["KELP"] = kelp_orders
            
        if "SQUID_INK" in state.order_depths:
            squid_ink_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            squid_ink_data = self.squid_ink_orders(
                state.order_depths["SQUID_INK"], squid_ink_timespan, squid_ink_width, squid_ink_take_width, 
                squid_ink_position, squid_ink_position_limit
            )
            
            # Unpack return values from squid_ink_orders
            squid_ink_orders = squid_ink_data[0]
            current_ema = squid_ink_data[1]
            short_ema = squid_ink_data[2]
            long_ema = squid_ink_data[3]
            trend_signal = squid_ink_data[4]
            dynamic_width = squid_ink_data[5]
            buy_volume = squid_ink_data[6]
            sell_volume = squid_ink_data[7]
            
            result["SQUID_INK"] = squid_ink_orders
            
            # Log SQUID_INK data for ML analysis
            self.log_squid_ink_data(
                state, 
                squid_ink_orders, 
                state.timestamp, 
                state.order_depths["SQUID_INK"],
                current_ema,
                short_ema,
                long_ema,
                trend_signal,
                dynamic_width,
                buy_volume,
                sell_volume
            )

        # After executing trades, update P&L tracking
        if "SQUID_INK" in state.own_trades and state.timestamp > 0:
            prev_position = 0
            current_position = state.position.get("SQUID_INK", 0)
            mid_price = 0
            
            # Get current mid price if available
            if "SQUID_INK" in state.order_depths:
                order_depth = state.order_depths["SQUID_INK"]
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
            
            # Estimate P&L from position change
            if state.own_trades.get("SQUID_INK"):
                trade_pl = 0
                for trade in state.own_trades["SQUID_INK"]:
                    if trade.timestamp == state.timestamp:
                        # Positive P&L if we buy low or sell high relative to mid
                        if trade.quantity > 0:  # Buy
                            trade_pl += (mid_price - trade.price) * trade.quantity
                        else:  # Sell
                            trade_pl += (trade.price - mid_price) * (-trade.quantity)
                
                # Add to recent P&L history
                self.recent_squid_ink_pl.append(trade_pl)
                # Keep history reasonably sized
                if len(self.recent_squid_ink_pl) > 20:
                    self.recent_squid_ink_pl.pop(0)
        traderData = jsonpickle.encode({
                "kelp_prices": self.kelp_prices,
                "kelp_vwap": self.kelp_vwap,
                "squid_ink_prices": self.squid_ink_prices,
                "squid_ink_vwap": self.squid_ink_vwap
            })
     
        conversions = 1
        logger.flush(state,result,conversions,traderData)
        return result, conversions, traderData