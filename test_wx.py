import os
import requests
from dotenv import load_dotenv

load_dotenv()
apikey = os.environ["WATSONX_APIKEY"]
url = os.environ["WATSONX_URL"].rstrip('/')

headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": apikey}
resp = requests.post("https://iam.cloud.ibm.com/identity/token", headers=headers, data=data)
token = resp.json()["access_token"]

endpoint = f"{url}/ml/v1/foundation_model_specs?version=2023-05-29"
headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
res = requests.get(endpoint, headers=headers)
models = [m["model_id"] for m in res.json().get("resources", [])]
print(models)
