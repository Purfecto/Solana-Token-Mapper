# tag.py

from collections import defaultdict

def tag_wallets(wallets, total_supply=None):
    tag_summary = defaultdict(int)

    # Ensure consistent structure
    wallets = [w for w in wallets if "wallet" in w and "balance" in w]
    wallets.sort(key=lambda x: x["balance"], reverse=True)

    total_supply = total_supply or sum(w["balance"] for w in wallets)
    top_10_cutoff = set(w["wallet"] for w in wallets[:10])

    for w in wallets:
        pct = (w["balance"] / total_supply) * 100

        if pct > 5:
            tag = "Whale"
        elif pct > 1:
            tag = "Large Holder"
        elif w["wallet"] in top_10_cutoff:
            tag = "Top 10 Holder"
        else:
            tag = "Retail"

        w["tag"] = tag
        tag_summary[tag] += 1

    print("\n🔍 Tagging Summary\n" + "=" * 24)
    for t, count in tag_summary.items():
        print(f"{t} Tags: {count}")

    return wallets
