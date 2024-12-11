import os
import json
import argparse
import requests
import configparser
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
import time

# Configuration Management
def load_config(config_path: str = "config.ini") -> Dict:
    """Load configuration from file or create default"""
    default_config = {
        "API": {
            "quicknode_url": "https://smart-purple-moon.solana-mainnet.quiknode.pro/YOUR_QUICKNODE_URL_HERE/",
            "rate_limit_per_second": "5"
        },
        "ANALYSIS": {
            "top_holders_limit": "10",
            "transactions_per_holder": "100"
        }
    }
    
    config = configparser.ConfigParser()
    
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        # Create default config
        for section, values in default_config.items():
            config[section] = values
        
        with open(config_path, 'w') as f:
            config.write(f)
        print(f"Created default config file at {config_path}")
    
    # Ensure URL is properly set
    if config["API"]["quicknode_url"] == "YOUR_QUICKNODE_URL":
        config["API"]["quicknode_url"] = default_config["API"]["quicknode_url"]
        with open(config_path, 'w') as f:
            config.write(f)
    
    return {
        "quicknode_url": config["API"]["quicknode_url"],
        "rate_limit": int(config["API"]["rate_limit_per_second"]),
        "top_holders_limit": int(config["ANALYSIS"]["top_holders_limit"]),
        "transactions_per_holder": int(config["ANALYSIS"]["transactions_per_holder"])
    }

# Rate Limiting
class RateLimiter:
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self.last_call = 0

    def wait(self):
        """Wait if needed to respect rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call
        
        if time_since_last_call < (1.0 / self.calls_per_second):
            time.sleep((1.0 / self.calls_per_second) - time_since_last_call)
        
        self.last_call = time.time()

@dataclass
class TokenAccount:
    """Represents a Solana token account with its essential information"""
    address: str
    owner: str
    amount: int
    delegate: Optional[str] = None

@dataclass
class TokenFlow:
    """Tracks how tokens move between wallets"""
    from_address: Optional[str]
    to_address: str
    amount: int
    timestamp: str
    method: str
    transaction_signature: str

class SolanaTokenAnalyzer:
    def __init__(self, rpc_url: str, rate_limiter: RateLimiter):
        self.rpc_url = rpc_url
        self.headers = {"Content-Type": "application/json"}
        self.rate_limiter = rate_limiter

    def get_token_info(self, mint_address: str) -> Dict:
        """Gets token creation info and current distribution"""
        print(f"Fetching token information...")
        
        try:
            # Get creation data
            launch_info = self._get_token_creation(mint_address)
            if not launch_info:
                raise Exception("Could not retrieve token creation information")
            
            print(f"Launch info retrieved. Created on: {launch_info.get('date', 'Unknown')}")
            
            # Get current holders
            holders = self._get_token_holders(mint_address)
            if holders:
                print(f"Found {holders.get('unique_holders', 0):,} current holders")
            else:
                raise Exception("Could not retrieve holder information")
            
            return {
                "launch_info": launch_info,
                "current_holders": holders
            }
        except Exception as e:
            print(f"Error getting token info: {str(e)}")
            return {"launch_info": {}, "current_holders": {}}
    
    def _get_token_creation(self, mint_address: str) -> Dict:
        """Finds token creation transaction"""
        try:
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    mint_address,
                    {
                        "limit": 1000,
                        "commitment": "finalized"
                    }
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            data = response.json()
            
            if "result" in data and data["result"]:
                # Sort by blockTime to get earliest transaction
                transactions = sorted(data["result"], key=lambda x: x.get("blockTime", 0))
                
                self.rate_limiter.wait()
                first_tx = self._get_transaction_details(transactions[0]["signature"])
                
                if first_tx:
                    return {
                        "date": datetime.fromtimestamp(first_tx["blockTime"]).isoformat(),
                        "creator": first_tx["transaction"]["message"]["accountKeys"][0],
                        "signature": transactions[0]["signature"]
                    }
            
            return {}
            
        except Exception as e:
            print(f"Error getting token creation info: {str(e)}")
            return {}

    def _get_token_holders(self, mint_address: str) -> List[Dict]:
        """Gets current token holder distribution"""
        try:
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getProgramAccounts",
                "params": [
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                    {
                        "encoding": "jsonParsed",
                        "filters": [
                            {"dataSize": 165},
                            {
                                "memcmp": {
                                    "offset": 0,
                                    "bytes": mint_address
                                }
                            }
                        ]
                    }
                ]
            }

            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            data = response.json()

            if "result" not in data:
                return []

            holders = []
            total_supply = 0
            
            print("Processing holder data...")
            for account in tqdm(data["result"], desc="Processing accounts"):
                info = account["account"]["data"]["parsed"]["info"]
                amount = int(info["tokenAmount"]["amount"])
                total_supply += amount
                
                holders.append({
                    "address": info["owner"],
                    "balance": amount
                })

            # Sort by balance and calculate percentages
            holders.sort(key=lambda x: x["balance"], reverse=True)
            for holder in holders:
                holder["percentage"] = round((holder["balance"] / total_supply) * 100, 4)

            return {
                "total_supply": total_supply,
                "unique_holders": len(holders),
                "holders": holders
            }

        except Exception as e:
            print(f"Error getting token holders: {str(e)}")
            return []

    def _get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Gets detailed transaction information"""
        try:
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            return response.json().get("result")
        
        except Exception as e:
            print(f"Error getting transaction details: {str(e)}")
            return None
class TokenFlowTracker:
    def __init__(self, quicknode_url: str, rate_limiter: RateLimiter, max_transactions: int = 100):
        self.rpc_url = quicknode_url
        self.headers = {"Content-Type": "application/json"}
        self.rate_limiter = rate_limiter
        self.max_transactions = max_transactions

    def analyze_token_flow(self, token_mint: str, top_holders: List[str]) -> List[TokenFlow]:
        """
        Analyzes token flow from creation to current top holders.
        Returns list of token movements.
        """
        flows = []
        total_holders = len(top_holders)
        
        print(f"\nAnalyzing token flows for {total_holders} holders...")
        
        with tqdm(total=total_holders, desc="Processing holders") as pbar:
            for idx, holder in enumerate(top_holders, 1):
                print(f"\nProcessing holder {idx}/{total_holders}: {holder}")
                holder_flows = self._trace_token_path(token_mint, holder)
                if holder_flows:
                    print(f"Found {len(holder_flows)} transactions for this holder")
                else:
                    print("No transactions found for this holder")
                flows.extend(holder_flows)
                pbar.update(1)
        
        print(f"\nAnalysis complete. Found {len(flows)} total token flow transactions")
        return flows

    def _get_holder_transactions(self, holder_address: str, token_mint: str) -> List[Dict]:
        """Gets token-specific transaction history for a holder"""
        try:
            print(f"Fetching token transactions for holder: {holder_address}")
            
            # First get the token account for this holder
            token_account = self._get_token_account(token_mint, holder_address)
            if not token_account:
                print(f"No token account found for holder")
                return []
                
            print(f"Found token account: {token_account}")
            
            self.rate_limiter.wait()
            # Now get transactions for this token account
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    token_account,
                    {"limit": self.max_transactions}
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            data = response.json()
            
            if "result" not in data:
                return []
            
            transactions = []
            with tqdm(total=len(data["result"]), desc="Fetching transactions") as pbar:
                for tx in data["result"]:
                    if "signature" in tx:
                        details = self._get_transaction_details(tx["signature"])
                        if details:
                            transactions.append(details)
                        pbar.update(1)
            
            print(f"Found {len(transactions)} token-specific transactions")
            return transactions
            
        except Exception as e:
            print(f"Error getting holder transactions: {str(e)}")
            return []

    def _get_token_account(self, token_mint: str, owner_address: str) -> Optional[str]:
        """Gets the token account address for a specific token and owner"""
        try:
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    owner_address,
                    {
                        "mint": token_mint
                    },
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            data = response.json()
            
            if "result" in data and data["result"]["value"]:
                return data["result"]["value"][0]["pubkey"]
            
            return None
            
        except Exception as e:
            print(f"Error getting token account: {str(e)}")
            return None

    def _get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Gets detailed transaction information"""
        try:
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            return response.json().get("result")
        
        except Exception as e:
            print(f"Error getting transaction details: {str(e)}")
            return None
        
    def _analyze_flow(self, transaction: Dict, token_mint: str, holder_address: str) -> Optional[TokenFlow]:
        """Analyzes a transaction for token flow information"""
        try:
            if not transaction or "transaction" not in transaction:
                return None

            # Get the token account for this holder
            token_account = self._get_token_account(token_mint, holder_address)
            if not token_account:
                return None

            instructions = transaction["transaction"]["message"]["instructions"]
            for instruction in instructions:
                # Check for parsed instruction data
                if not isinstance(instruction, dict) or "parsed" not in instruction:
                    continue

                parsed = instruction["parsed"]
                if isinstance(parsed, dict):
                    # Check for various token transfer types
                    if parsed.get("type") in ["transfer", "transferChecked", "mintTo", "transferChecked"]:
                        info = parsed.get("info", {})
                        
                        # Check token mint matches if available
                        if "mint" in info and info["mint"] != token_mint:
                            continue

                        # For mint operations
                        if parsed["type"] == "mintTo" and info.get("account") == token_account:
                            return TokenFlow(
                                from_address="Mint",
                                to_address=holder_address,
                                amount=int(info["amount"]),
                                timestamp=datetime.fromtimestamp(transaction["blockTime"]).isoformat(),
                                method="mint",
                                transaction_signature=transaction["transaction"]["signatures"][0]
                            )

                        # For transfers
                        destination = info.get("destination") or info.get("account")
                        source = info.get("source") or info.get("authority")
                        
                        # Check if this holder is involved in the transfer
                        if destination == token_account:
                            try:
                                amount = int(info.get("tokenAmount", {}).get("amount") or info.get("amount", 0))
                                print(f"Found incoming transfer of {amount:,} tokens to {holder_address}")
                                return TokenFlow(
                                    from_address=source,
                                    to_address=holder_address,
                                    amount=amount,
                                    timestamp=datetime.fromtimestamp(transaction["blockTime"]).isoformat(),
                                    method="transfer",
                                    transaction_signature=transaction["transaction"]["signatures"][0]
                                )
                            except (KeyError, ValueError) as e:
                                print(f"Error parsing amount: {e}")
                                continue
            
            return None
            
        except Exception as e:
            print(f"Error analyzing flow: {str(e)}")
            return None

    def _trace_token_path(self, token_mint: str, holder_address: str) -> List[TokenFlow]:
        """Traces the complete path of how tokens reached a holder"""
        flows = []
        transactions = self._get_holder_transactions(holder_address, token_mint)
        
        if transactions:
            print(f"Analyzing {len(transactions)} transactions for token flows...")
            with tqdm(total=len(transactions), desc="Analyzing transactions") as pbar:
                for tx in transactions:
                    if flow := self._analyze_flow(tx, token_mint, holder_address):
                        print(f"Found token transfer: {flow.amount:,} tokens from {flow.from_address} to {flow.to_address}")
                        flows.append(flow)
                    pbar.update(1)
        
        if not flows:
            print("No token transfers found in transactions")
        else:
            print(f"Found {len(flows)} token transfers")
                
        return flows

    def save_analysis(self, token_mint: str, flows: List[TokenFlow], info: Dict):
        """Saves flow analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        analysis_dir = os.path.join('token_analysis', token_mint, f'analysis_{timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"\nSaving analysis to: {analysis_dir}")
        
        # Save terminal output log
        with open(os.path.join(analysis_dir, 'analysis_log.txt'), 'w') as f:
            f.write(f"Token Analysis Log for {token_mint}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Token Information:\n")
            f.write(f"- Launch Date: {info['launch_info'].get('date', 'Unknown')}\n")
            f.write(f"- Creator: {info['launch_info'].get('creator', 'Unknown')}\n")
            f.write(f"- Total Supply: {info['current_holders'].get('total_supply', 0):,}\n")
            f.write(f"- Unique Holders: {info['current_holders'].get('unique_holders', 0):,}\n\n")
            
            f.write("Analysis Results:\n")
            f.write(f"- Analyzed Holders: {len(info['current_holders']['holders'][:10])}\n")
            f.write(f"- Total Flows Found: {len(flows)}\n")

        # Save raw holder data
        raw_data_dir = os.path.join(analysis_dir, 'raw_data')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        with open(os.path.join(raw_data_dir, 'holders_data.json'), 'w') as f:
            json.dump(info['current_holders']['holders'][:10], f, indent=2)

        if not flows:
            self._save_no_transactions_report(analysis_dir, token_mint, info)
            return

        # Save flow data to CSV
        self._save_flow_data(analysis_dir, flows)

        # Generate analysis report
        self._generate_report(analysis_dir, flows, info)

    def _save_no_transactions_report(self, analysis_dir: str, token_mint: str, info: Dict):
        """Saves a report when no transactions are found"""
        with open(os.path.join(analysis_dir, 'no_transactions.txt'), 'w') as f:
            f.write(f"No token flow transactions found for {token_mint}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Token Information:\n")
            f.write(f"Launch Date: {info['launch_info'].get('date', 'Unknown')}\n")
            f.write(f"Creator: {info['launch_info'].get('creator', 'Unknown')}\n")
            f.write(f"Total Supply: {info['current_holders'].get('total_supply', 0):,}\n")
            f.write(f"Unique Holders: {info['current_holders'].get('unique_holders', 0):,}\n\n")
            f.write("Analysis Details:\n")
            f.write("No token transfer transactions were found for the analyzed holders.\n")
        print("No transactions found. Created detailed no_transactions.txt")

    def _save_flow_data(self, analysis_dir: str, flows: List[TokenFlow]):
        """Saves flow data to CSV"""
        flow_data = pd.DataFrame([{
            'From': flow.from_address or 'Creation',
            'To': flow.to_address,
            'Amount': flow.amount,
            'Time': flow.timestamp,
            'Method': flow.method,
            'Transaction': flow.transaction_signature
        } for flow in flows])
        
        csv_path = os.path.join(analysis_dir, 'token_flows.csv')
        flow_data.to_csv(csv_path, index=False)
        print(f"Saved flow data to: {csv_path}")

    def _generate_report(self, output_dir: str, flows: List[TokenFlow], info: Dict):
        """Creates human-readable analysis report"""
        report_path = os.path.join(output_dir, 'flow_analysis.txt')
        print(f"Generating analysis report: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("Token Flow Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Token Information:\n")
            f.write(f"Launch Date: {info['launch_info'].get('date', 'Unknown')}\n")
            f.write(f"Creator: {info['launch_info'].get('creator', 'Unknown')}\n")
            f.write(f"Total Supply: {info['current_holders'].get('total_supply', 0):,}\n")
            f.write(f"Current Holders: {info['current_holders'].get('unique_holders', 0):,}\n\n")
            
            f.write("Token Flow Summary:\n")
            
            # Group flows by recipient
            by_recipient = {}
            for flow in flows:
                if flow.to_address not in by_recipient:
                    by_recipient[flow.to_address] = []
                by_recipient[flow.to_address].append(flow)
            
            for address, holder_flows in by_recipient.items():
                f.write(f"\nHolder: {address}\n")
                f.write("-" * 40 + "\n")
                total_received = sum(flow.amount for flow in holder_flows)
                f.write(f"Total Received: {total_received:,}\n")
                
                f.write("\nAcquisition Timeline:\n")
                for flow in sorted(holder_flows, key=lambda x: x.timestamp):
                    f.write(f"- {flow.timestamp}: {flow.amount:,} via {flow.method}\n")
                    if flow.from_address:
                        f.write(f"  From: {flow.from_address}\n")
                    f.write(f"  Tx: {flow.transaction_signature}\n")
        
        print(f"Analysis report generated successfully")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze Solana token flows and holder distribution.')
    parser.add_argument('--token', '-t', type=str, help='Token mint address to analyze')
    parser.add_argument('--config', '-c', type=str, default='config.ini', help='Path to configuration file')
    parser.add_argument('--holders', '-n', type=int, help='Number of top holders to analyze')
    parser.add_argument('--transactions', '-tx', type=int, help='Number of transactions to analyze per holder')
    return parser.parse_args()

def analyze_token(token_mint: Optional[str] = None, config_path: str = "config.ini"):
    """Main analysis function"""
    print("\nInitializing Token Flow Analysis...")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config(config_path)
        rate_limiter = RateLimiter(config["rate_limit"])
        
        # Get token mint from arguments or use default
        if not token_mint:
            token_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # Default to BONK token
        
        print(f"Analyzing token: {token_mint}")
        
        # Initialize analyzers with rate limiter
        analyzer = SolanaTokenAnalyzer(config["quicknode_url"], rate_limiter)
        
        print("\nFetching token information and holder data...")
        info = analyzer.get_token_info(token_mint)
        
        if not info["current_holders"]:
            print("Error: Could not get token information")
            return
        
        print(f"\nFound {info['current_holders'].get('unique_holders', 0):,} unique holders")
        print(f"Total supply: {info['current_holders'].get('total_supply', 0):,}")
        
        # Get top holders
        top_holders = [h["address"] for h in info["current_holders"]["holders"][:config["top_holders_limit"]]]
        print(f"\nAnalyzing top {len(top_holders)} holders...")
        
        # Track token flow
        tracker = TokenFlowTracker(
            config["quicknode_url"],
            rate_limiter,
            config["transactions_per_holder"]
        )
        flows = tracker.analyze_token_flow(token_mint, top_holders)
        
        # Save results
        tracker.save_analysis(token_mint, flows, info)
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return

if __name__ == "__main__":
    args = parse_arguments()
    analyze_token(args.token, args.config)       