import os
import requests

api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

url = "https://data.alpaca.markets/v2/stocks/AAPL/bars"

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

params = {
    "timeframe": "1Day",
    "start": "2026-09-01",
    "end": "2026-10-10",
    "feed": "sip",
}

r = requests.get(url, headers=headers, params=params, timeout=20)

print("HTTP:", r.status_code)
print(r.text[:3000])
