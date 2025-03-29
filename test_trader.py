# test_trader.py
from trader1 import Trader
from datamodel import OrderDepth, TradingState, Observation

# Create a simple test state
order_depth = OrderDepth()
order_depth.buy_orders = {2020: 20}
order_depth.sell_orders = {2024: -20}

state = TradingState(
    traderData="",
    timestamp=1000,
    listings={},
    order_depths={"KELP": order_depth},
    own_trades={},
    market_trades={},
    position={},
    observations=Observation({}, {})
)

# Initialize your trader
trader = Trader()

# Test if run method exists and works
try:
    result, conversions, trader_data = trader.run(state)
    print("Trader.run() method called successfully!")
    print(f"Result: {result}")
except AttributeError as e:
    print(f"Error: {e}")