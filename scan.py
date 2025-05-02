import json
import time
import requests

RPC_URL = "https://mainnet.helius-rpc.com/?api-key=9fa576c9-8798-4b25-a65b-c62dc87be4be"

def rpc_request(payload):
    headers = {"Content-Type": "application/json"}
    retries = 3
    for i in range(retries):
        try:
            response = requests.post(RPC_URL, headers=headers, data=json.dumps(payload), timeout=30)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[!] RPC request failed (attempt {i+1}/{retries}): {e}")
            time.sleep(5)
    raise Exception("RPC request failed after max retries")

def get_token_supply(mint_address):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [mint_address]
    }
    result = rpc_request(payload)
    amount = int(result["result"]["value"]["amount"])
    decimals = int(result["result"]["value"]["decimals"])
    return amount, decimals

def get_token_largest_accounts(mint_address):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [mint_address]
    }
    result = rpc_request(payload)
    return [{"address": acc["address"], "amount": int(acc["amount"])} for acc in result["result"]["value"]]

def get_token_holders(mint_address):
    """Try to get full holder list from Helius"""
    url = f"https://mainnet.helius-rpc.com/api/v0/token/{mint_address}/holders?network=mainnet"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "holders" in data:
                return data["holders"]
            elif isinstance(data, list):
                return data
        print("[!] Helius does not support this token — falling back to top accounts")
        return []
    except Exception:
        print(f"[!] Helius request failed — falling back to top accounts")
        return []

def scan_token(mint_address):
    supply_raw, decimals = get_token_supply(mint_address)
    supply = supply_raw / (10 ** decimals)
    top_accounts = get_token_largest_accounts(mint_address)
    holders_data = get_token_holders(mint_address)

    holders = []
    if holders_data:
        for acc in holders_data:
            wallet = acc.get("owner") or acc.get("pubkey") or acc.get("address")
            raw_balance = int(acc.get("token_balance", 0)) or int(acc.get("amount", 0))
            balance = raw_balance / (10 ** decimals)
            if wallet:
                holders.append({"wallet": wallet, "balance": balance})

        top_sum = sum(a["amount"] for a in top_accounts) / (10 ** decimals)
        print(f"[+] Fetched full holder list from Helius: {len(holders)} wallets")
        print(f"[+] Top {len(top_accounts)} holders control {round(top_sum / supply * 100, 2)}% of supply")
        print(f"[+] Remaining {len(holders) - len(top_accounts)} holders share the other {round(100 - (top_sum / supply * 100), 2)}%")
    else:
        print("[!] Using fallback: top accounts only")
        holders = [{
            "wallet": acc["address"],
            "balance": acc["amount"] / (10 ** decimals)
        } for acc in top_accounts]

    return {"supply": supply, "holders": holders}
