import os
import json
from datetime import datetime
from collections import Counter



def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def create_output_dir(path):
    os.makedirs(path, exist_ok=True)


def get_or_infer_tag(wallet, existing_tag):
    if existing_tag:
        return existing_tag
    inferred = is_program_account(wallet)
    return inferred if inferred else "Unlabeled"


def generate_summary(scan_data, mint_address, old_file=None):
    holders = scan_data["holders"]
    total_supply = scan_data["supply"]

    clusters = {}
    for holder in holders:
        cluster_id = holder.get("cluster")
        if cluster_id is not None:
            clusters.setdefault(cluster_id, []).append(holder)

    top_10 = sorted(holders, key=lambda x: x["balance"], reverse=True)[:10]
    concentration_top_10 = sum(h["balance"] for h in top_10) / total_supply * 100

    risk_flags = []
    if concentration_top_10 > 80:
        risk_flags.append("High concentration: Top 10 hold >80% supply")

    cluster_summary = {}
    for cluster_id, members in clusters.items():
        for m in members:
            m["tag"] = get_or_infer_tag(m["wallet"], m.get("tag"))

        tag_counter = Counter(m["tag"] for m in members)
        majority_tag = tag_counter.most_common(1)[0][0]
        label = f"{majority_tag} Bundle"

        cluster_summary[str(cluster_id)] = {
            "label": label,
            "members": [
                {
                    "wallet": m["wallet"],
                    "balance": m["balance"],
                    "tag": m["tag"]
                }
                for m in members
            ]
        }

    return {
        "mint": mint_address,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_supply": total_supply,
        "holder_count": len(holders),
        "top_10_holders": [
            {
                "wallet": h["wallet"],
                "balance": h["balance"],
                "cluster": h.get("cluster"),
                "tag": get_or_infer_tag(h["wallet"], h.get("tag"))
            } for h in top_10
        ],
        "top_10_concentration_percent": round(concentration_top_10, 2),
        "large_holders_count": len([h for h in holders if h["balance"] > 100_000_000]),
        "risk_flags": risk_flags,
        "clusters": cluster_summary
    }


def print_summary(summary):
    print("\n📊 TOKEN DISTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"Mint Address: {summary['mint']}")
    print(f"Scan Timestamp: {summary['timestamp']}")
    print(f"Total Supply: {summary['total_supply']:,}")
    print(f"Unique Holders: {summary['holder_count']}")
    print(f"Top 10 Holders: {summary['top_10_concentration_percent']}% of supply")
    print(f"Large Holders (>100M tokens): {summary['large_holders_count']}")

    if summary.get("clusters"):
        print("\n🔗 WALLET CLUSTERS")
        for cluster_id, cluster_data in summary["clusters"].items():
            print(f"Cluster {cluster_id} ({cluster_data['label']}):")
            for member in cluster_data["members"]:
                print(f"  - {member['wallet']} | {member['balance']:,} [{member['tag']}]")

    if summary["risk_flags"]:
        print("\n⚠️ RISK FLAGS")
        print("- " + "\n- ".join(summary["risk_flags"]))

    print("\n📈 TOP 10 HOLDERS")
    for i, holder in enumerate(summary["top_10_holders"], 1):
        percent = (holder["balance"] / summary["total_supply"]) * 100
        print(f"{i}. Cluster {holder.get('cluster', '-')}: {holder['wallet']} | {holder['balance']:,} ({percent:.2f}%) [{holder['tag']}]")
