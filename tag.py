import json
import os

TAGS_PATH = "tags.json"

FALLBACK_TAGS = {
    "4ex": "PumpFun",
    "7sX": "PumpFun",
    "5uS": "Early Buyer",
    "Ccz": "Team",
    "Tokenkeg": "Liquidity Provider",
    "So111": "Liquidity Provider",
    "Sysvar": "Sysvar",
    "Whirlp": "DEX"
}

def load_tags():
    if not os.path.exists(TAGS_PATH):
        print("[!] No tags.json found. Skipping static tag loading.")
        return {}
    try:
        with open(TAGS_PATH, "r") as f:
            tags = json.load(f)
            print(f"[+] Loaded {sum(len(v) for v in tags.values())} tagged wallets from tags.json")
            return tags
    except Exception as e:
        print(f"[!] Failed to load tags.json: {e}")
        return {}

def get_fallback_tag(wallet):
    for prefix, tag in FALLBACK_TAGS.items():
        if wallet.startswith(prefix):
            return tag
    return None

def tag_wallets(holders):
    static_tags = load_tags()
    for holder in holders:
        wallet = holder["wallet"]
        tag = None

        for tag_name, address_list in static_tags.items():
            if wallet in address_list:
                tag = tag_name
                break

        if not tag:
            tag = get_fallback_tag(wallet)

        holder["tag"] = tag if tag else "Unlabeled"

    return holders
