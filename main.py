from datamodel import TradingState
from trader1 import Trader
import json
import os

# Step 1: Load the test data (round_data.json)
with open("round_data.json", "r") as f:
    simulation_data = json.load(f)

# Step 2: Initialize your trader instance
trader = Trader()

# Step 3: Step through each timestamp (tick)
for step in simulation_data:
    state = TradingState(
        traderData=step["traderData"],
        timestamp=step["timestamp"],
        listings=step["listings"],
        order_depths=step["order_depths"],
        own_trades=step["own_trades"],
        market_trades=step["market_trades"],
        position=step["position"],
        observations=step["observations"]
    )

    trader.run(state)  # This automatically calls logger.flush()

# Step 4: Output goes to stdout → save it if you want:
# python3 main.py > logs/log.json
