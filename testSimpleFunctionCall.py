import requests


class BitCoinPriceAgent:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetchBitcoinPrice(self):
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                data = response.json()
                bitcoin_price = data['bpi']['USD']['rate']
                print(f"Current bitcoin price USD: ${bitcoin_price}")
            else:
                print("failed to fetch price from API")
        except Exception as e:
            print(f'error occured {e}')
            
api_url = "https://api.coindesk.com/v1/bpi/currentprice/BTC.json"
agent = BitCoinPriceAgent(api_url)
agent.fetchBitcoinPrice()
