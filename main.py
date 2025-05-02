import argparse
import os
import json
from datetime import datetime
from scan import scan_token
from tag import tag_wallets
from utils import (
    save_json, load_json, create_output_dir,
    generate_summary, print_summary
)
from cluster import cluster_wallets

def parse_args():
    parser = argparse.ArgumentParser(description="Solana Token Mapper: CLI Tool for Token Distribution Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan token holders and supply")
    scan_parser.add_argument("--mint", required=True, help="Token mint address")
    scan_parser.add_argument("--tag", action="store_true", help="Auto-tag known wallets after scan")
    scan_parser.add_argument("--cluster", action="store_true", help="Run wallet clustering after scan")

    tag_parser = subparsers.add_parser("tag", help="Apply wallet labels to holders")
    tag_parser.add_argument("--input", required=True, help="Input JSON file (holders.json)")
    tag_parser.add_argument("--output", required=True, help="Output JSON file (holders_tagged.json)")

    viz_parser = subparsers.add_parser("viz", help="Generate distribution charts")
    viz_parser.add_argument("--input", required=True, help="Input JSON file (holders_tagged.json)")
    viz_parser.add_argument("--charts_dir", default="charts", help="Directory to save charts")

    compare_parser = subparsers.add_parser("compare", help="Compare two scans")
    compare_parser.add_argument("--old", required=True, help="Old summary.json")
    compare_parser.add_argument("--new", required=True, help="New summary.json")
    compare_parser.add_argument("--output", default="diff_report.json", help="Output diff report")

    lookup_parser = subparsers.add_parser("lookup", help="Lookup wallet cluster, tag, and % of supply")
    lookup_parser.add_argument("--wallet", required=True, help="Wallet address to lookup")
    lookup_parser.add_argument("--summary", required=True, help="Path to summary.json file")

    return parser.parse_args()

def main():
    args = parse_args()
    command = args.command

    if command == "scan":
        mint = args.mint
        print(f"[+] Scanning token: {mint}")
        try:
            scan_data = scan_token(mint)
        except Exception as e:
            print(f"[!] Error scanning token: {e}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f"distributions/{mint}/{timestamp}"
        create_output_dir(output_dir)

        holders_file = f"{output_dir}/holders.json"
        save_json(scan_data["holders"], holders_file)

        if args.tag:
            print("[+] Applying wallet tags...")
            scan_data["holders"] = tag_wallets(scan_data["holders"], scan_data["supply"])

        if args.cluster:
            print("[+] Running wallet clustering...")
            holders, clusters = cluster_wallets(scan_data["holders"])
            scan_data["holders"] = holders
            scan_data["clusters"] = clusters

        summary = generate_summary(scan_data, mint)
        summary_file = f"{output_dir}/summary.json"
        save_json(summary, summary_file)

        print_summary(summary)
        print(f"\n[+] Scan saved to {output_dir}/")

    elif command == "tag":
        input_file = args.input
        output_file = args.output
        print(f"[+] Tagging wallets in {input_file}")
        holders = load_json(input_file)
        supply = sum(h["balance"] for h in holders)
        tagged = tag_wallets(holders, supply)
        save_json(tagged, output_file)
        print(f"[+] Tagged holders saved to {output_file}")

    elif command == "viz":
        input_file = args.input
        charts_dir = args.charts_dir
        print(f"[+] Generating charts from {input_file}")
        holders = load_json(input_file)
        from visualize import plot_distribution
        plot_distribution(holders, charts_dir)
        print(f"[+] Charts saved to {charts_dir}/")

    elif command == "compare":
        old_file = args.old
        new_file = args.new
        output_file = args.output
        print(f"[+] Comparing {old_file} vs {new_file}")
        from compare import compare_scans
        diff = compare_scans(old_file, new_file)
        save_json(diff, output_file)
        print(f"[+] Diff report saved to {output_file}")

    elif command == "lookup":
        wallet = args.wallet
        summary = load_json(args.summary)
        total_supply = summary["total_supply"]
        found = False

        for cluster_id, cluster in summary["clusters"].items():
            for member in cluster["members"]:
                if member["wallet"] == wallet:
                    percent = (member["balance"] / total_supply) * 100
                    print("\n\U0001F50D WALLET LOOKUP")
                    print("=" * 40)
                    print(f"Wallet: {wallet}")
                    print(f"Cluster: {cluster_id} ({cluster['label']})")
                    print(f"Tag: {member.get('tag', 'Unlabeled')}")
                    print(f"Balance: {member['balance']:,} ({percent:.2f}% of supply)")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"[!] Wallet {wallet} not found in summary.")

if __name__ == "__main__":
    main()
