from collections import defaultdict

def cluster_wallets(holders):
    clusters = []
    wallet_to_cluster = {}

    # Very naive clustering: Group wallets with identical balances
    for holder in holders:
        wallet = holder["wallet"]
        balance = holder["balance"]
        matched_cluster = None

        # Try to group wallets with same rounded balance
        for idx, cluster in enumerate(clusters):
            rep_balance = cluster[0]["balance"]
            if abs(rep_balance - balance) < 1_000_000:  # within ~1M
                matched_cluster = idx
                break

        if matched_cluster is None:
            matched_cluster = len(clusters)
            clusters.append([])

        clusters[matched_cluster].append(holder)
        wallet_to_cluster[wallet] = matched_cluster

    # Assign cluster ID to each holder
    for wallet, cluster_id in wallet_to_cluster.items():
        for h in holders:
            if h["wallet"] == wallet:
                h["cluster"] = cluster_id
                break

    return holders, wallet_to_cluster
