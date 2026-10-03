import os
from alpaca.trading.client import TradingClient

client = TradingClient(os.environ["APCA_API_KEY_ID"],
                       os.environ["APCA_API_SECRET_KEY"], paper=True)
acct = client.get_account()
print(acct.status, acct.equity, acct.cash, acct.buying_power)
print([(p.symbol, p.qty) for p in client.get_all_positions()])
