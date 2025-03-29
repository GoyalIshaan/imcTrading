from datamodel import OrderDepth, TradingState, Order, Listing, Trade, Observation, Symbol
from typing import List, Dict, Any
import json
from datamodel import ProsperityEncoder

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
        # Pre-allocate a compact price history list
        self.price_history = [0.0] * 10
        self.history_index = 0

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        max_orders = 20  # Reduced to prevent excessive order generation
        order_volume = 5
        spread = 1

        # Efficient price history parsing
        if state.traderData:
            try:
                # Use faster list comprehension
                self.price_history = [float(p) for p in state.traderData.split(",") if p][:10]
            except:
                # Reset to default if parsing fails
                self.price_history = [0.0] * 10

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            
            # Simplified fair value estimation
            if product == "RAINFOREST_RESIN":
                fair_value = 10000
            elif product == "KELP":
                fair_value = state.observations.plainValueObservations.get(product, 2000)
            else:
                fair_value = 10

            # Efficient mid-price calculation for Kelp
            if product == "KELP":
                try:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Circular buffer for price history
                    self.price_history[self.history_index] = mid_price
                    self.history_index = (self.history_index + 1) % 10
                except:
                    mid_price = fair_value

            # Constrained order generation
            try:
                # Buy orders
                buy_prices = [fair_value - spread, fair_value - spread - 1]
                for price in buy_prices[:2]:
                    if len(orders) < max_orders:
                        orders.append(Order(product, price, order_volume))

                # Sell orders
                sell_prices = [fair_value + spread, fair_value + spread + 1]
                for price in sell_prices[:2]:
                    if len(orders) < max_orders:
                        orders.append(Order(product, price, -order_volume))

                result[product] = orders
            except Exception as e:
                # Ensure we don't crash if order generation fails
                result[product] = []

        # Compact price history encoding
        traderData = ",".join(map(str, self.price_history))
        
        return result, conversions, traderData