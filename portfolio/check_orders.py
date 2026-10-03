import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(os.environ["APCA_API_KEY_ID"],
                       os.environ["APCA_API_SECRET_KEY"], paper=True)

clock = client.get_clock()
print("Market open:", clock.is_open, "| next open:", clock.next_open)

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=20))
for o in orders:
    print(o.symbol, o.status, "notional:", o.notional, "filled_qty:", o.filled_qty,
          "avg_price:", o.filled_avg_price)

print("Positions:", [(p.symbol, p.qty, p.market_value) for p in client.get_all_positions()])
