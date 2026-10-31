import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from sizing import size_orders

TICKERS = ["SPY", "QQQ", "GLD"]
TEST_EQUITY = 15.0   # tiny sleeve; each ticker targets $5

client = TradingClient(os.environ["APCA_API_KEY_ID"],
                       os.environ["APCA_API_SECRET_KEY"], paper=True)

# Safety: this script must only ever talk to the paper API.
assert "paper" in client._base_url, "Not pointed at paper API; aborting."

for sym in TICKERS:
    asset = client.get_asset(sym)
    print(f"{sym}: tradable={asset.tradable} fractionable={asset.fractionable}")
    assert asset.tradable and asset.fractionable, f"{sym} cannot take fractional orders"

weights = {s: 1 / len(TICKERS) for s in TICKERS}
orders, leftover = size_orders(weights, TEST_EQUITY, mode="fractional")
print("Sized orders:", orders, "leftover:", leftover)

for o in orders:
    req = MarketOrderRequest(symbol=o["symbol"], notional=o["notional"],
                             side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    resp = client.submit_order(req)
    print(f"Submitted {o['symbol']} ${o['notional']}: id={resp.id} status={resp.status}")

print("Open orders:", [(x.symbol, str(x.status)) for x in client.get_orders()])
print("Positions:", [(p.symbol, p.qty, p.market_value) for p in client.get_all_positions()])
