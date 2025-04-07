# Get twitter last tweets endpoint
import requests

url = "https://api.twitterapi.io/twitter/user/last_tweets"

querystring = {"userId":"44196397"}

headers = {"X-API-Key": "2e26af864ace4a0688a62e56fa6bfd54"}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)

# Get user info using username endpoint
import requests

url = "https://api.twitterapi.io/twitter/user/info"

querystring = {"userName":"elonmusk"}

headers = {"X-API-Key": "2e26af864ace4a0688a62e56fa6bfd54"}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)