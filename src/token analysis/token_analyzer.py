import os
import json
import logging
import argparse
import configparser
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import requests
from tqdm import tqdm
import time

# Configuration Management
class Config:
    DEFAULT_CONFIG = {
        "API": {
            "quicknode_url": "https://smart-purple-moon.solana-mainnet.quiknode.pro/ADD_YOUR_QUICKNODE_RPC/",
            "rate_limit_per_second": "5"
        },
        "ANALYSIS": {
            "min_holders": "10",
            "max_holders": "1000",
            "transactions_per_holder": "100",
            "log_level": "INFO"
        }
    }

    @staticmethod
    def load(config_path: str = "config.ini") -> Dict:
        """Load configuration with validation"""
        config = configparser.ConfigParser()
        
        if os.path.exists(config_path):
            config.read(config_path)
        else:
            # Create default config
            for section, values in Config.DEFAULT_CONFIG.items():
                config[section] = values
            
            # Write to file
            with open(config_path, 'w') as f:
                config.write(f)
            print(f"Created default config file at {config_path}")

        return {
            "quicknode_url": config["API"]["quicknode_url"],
            "rate_limit": int(config["API"]["rate_limit_per_second"]),
            "min_holders": int(config["ANALYSIS"]["min_holders"]),
            "max_holders": int(config["ANALYSIS"]["max_holders"]),
            "transactions_per_holder": int(config["ANALYSIS"]["transactions_per_holder"]),
            "log_level": config["ANALYSIS"]["log_level"]
        }

# Transaction Definitions
TRANSACTION_TYPES = {
    # Token Program Instructions
    "mintTo": "MINT",
    "transfer": "TRANSFER",
    "transferChecked": "TRANSFER",
    "burn": "BURN",
    
    # AMM Programs
    "ORCA": {
        "program_id": "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",
        "type": "SWAP"
    },
    "JUPITER": {
        "program_id": "JUP6LkbZbjS1jKKwapdHGA9Bo7fSyfFS8JdzYFLkwp6",
        "type": "SWAP"
    },
    "RAYDIUM": {
        "program_id": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        "type": "SWAP"
    }
}

SOURCE_TYPES = {
    "MINT": "Token Mint",
    "AMM": "DEX Swap",
    "WALLET": "Direct Transfer",
    "PROGRAM": "Program Interaction"
}

@dataclass
class EnhancedTokenFlow:
    """Enhanced token flow tracking with additional metadata"""
    from_address: Optional[str]
    to_address: str
    amount: int
    timestamp: str
    method: str
    transaction_signature: str
    source_type: str
    program_id: Optional[str] = None
    instruction_index: Optional[int] = None
    additional_data: Optional[Dict] = None

class RateLimiter:
    """Rate limiting for API calls"""
    def __init__(self, calls_per_second: int):
        self.calls_per_second = calls_per_second
        self.last_call = 0

    def wait(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call
        if time_since_last_call < (1.0 / self.calls_per_second):
            time.sleep((1.0 / self.calls_per_second) - time_since_last_call)
        self.last_call = time.time()

class TokenAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.rate_limiter = RateLimiter(config["rate_limit"])
        self.setup_logging(config["log_level"])
        self.known_amm_programs = {v['program_id']: k for k, v in TRANSACTION_TYPES.items() 
                                 if isinstance(v, dict) and 'program_id' in v}

    def setup_logging(self, log_level: str):
        """Configure comprehensive logging system"""
        self.logger = logging.getLogger('TokenAnalyzer')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)

    def setup_analysis_logging(self, analysis_dir: str):
        """Set up logging for specific analysis run"""
        file_handler = logging.FileHandler(os.path.join(analysis_dir, 'analysis.log'))
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def analyze_token(self, token_mint: str, num_holders: Optional[int] = None) -> None:
        """Main analysis entry point"""
        try:
            analysis_start = datetime.now()
            self.logger.info(f"Starting analysis for token: {token_mint}")
            
            # Create analysis directory structure
            timestamp = analysis_start.strftime('%Y%m%d_%H%M')
            base_dir = os.path.join('mapper', 'token_analysis')
            analysis_dir = os.path.join(base_dir, token_mint, f'analysis_{timestamp}')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Set up file logging for this analysis
            self.setup_analysis_logging(analysis_dir)
            
            # Get token info and holders
            token_info = self.get_token_info(token_mint)
            if not token_info:
                self.logger.error("Failed to get token information")
                return
            
            # Determine number of holders to analyze
            num_holders = self._validate_holder_count(num_holders)
            holders = self.get_holders(token_mint, num_holders)
            
            if not holders:
                self.logger.error("No holders found to analyze")
                return
            
            # Analyze flows for each holder
            flows = []
            for holder in tqdm(holders, desc="Analyzing holders"):
                holder_flows = self.analyze_holder_flows(token_mint, holder)
                flows.extend(holder_flows)
            
            # Save results
            self.save_analysis(analysis_dir, token_mint, flows, token_info, analysis_start)
            self.logger.info("Analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in token analysis: {str(e)}", exc_info=True)
            raise

    def get_token_info(self, token_mint: str) -> Optional[Dict]:
        """Get token metadata and current distribution"""
        self.logger.info(f"Fetching token information for {token_mint}")
        
        try:
            self.rate_limiter.wait()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    token_mint,
                    {"encoding": "jsonParsed"}
                ]
            }
            
            response = requests.post(self.config["quicknode_url"], 
                               json=payload,
                               headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get token info: {response.status_code}")
                return None
                
            data = response.json()
            if "result" not in data or not data["result"]["value"]:
                self.logger.error("Invalid token mint address")
                return None
            
            token_info = {
                "token_mint": token_mint,
                "decimals": data["result"]["value"]["data"]["parsed"]["info"]["decimals"],
                "token_program": data["result"]["value"]["owner"]
            }
            
            self.logger.info(f"Token info retrieved successfully")
            return token_info
            
        except Exception as e:
            self.logger.error(f"Error getting token info: {str(e)}")
            return None

    def get_holders(self, token_mint: str, num_holders: Optional[int] = None) -> List[str]:
        """Get list of token holders"""
        self.logger.info("Fetching token holders...")
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
                                    "bytes": token_mint
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(self.config["quicknode_url"], 
                               json=payload,
                               headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get holders: {response.status_code}")
                return []
            
            data = response.json()
            if "result" not in data:
                return []
            
            # Extract holder addresses with balances
            holders = []
            for account in data["result"]:
                try:
                    info = account["account"]["data"]["parsed"]["info"]
                    amount = int(info["tokenAmount"]["amount"])
                    if amount > 0:
                        holders.append(info["owner"])
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error parsing holder data: {str(e)}")
                    continue
            
            # Sort holders by balance and limit
            max_holders = num_holders or self.config["max_holders"]
            holders = holders[:max_holders]
            
            self.logger.info(f"Found {len(holders)} holders to analyze")
            return holders
            
        except Exception as e:
            self.logger.error(f"Error getting holders: {str(e)}")
            return []

    def analyze_holder_flows(self, token_mint: str, holder_address: str) -> List[EnhancedTokenFlow]:
        """Analyze token flows for a specific holder"""
        self.logger.info(f"Analyzing flows for holder: {holder_address}")
        
        try:
            # Get holder's token account
            token_account = self._get_token_account(token_mint, holder_address)
            if not token_account:
                self.logger.warning(f"No token account found for holder: {holder_address}")
                return []
            
            # Get transactions
            transactions = self._get_holder_transactions(token_account)
            
            # Analyze each transaction
            flows = []
            for tx in tqdm(transactions, desc="Analyzing transactions", leave=False):
                tx_flows = self._analyze_transaction(tx, token_mint, holder_address, token_account)
                flows.extend(tx_flows)
            
            self.logger.info(f"Found {len(flows)} flows for holder {holder_address}")
            return flows
            
        except Exception as e:
            self.logger.error(f"Error analyzing holder {holder_address}: {str(e)}")
            return []

    def _get_token_account(self, token_mint: str, owner_address: str) -> Optional[str]:
        """Get token account for holder"""
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
            
            response = requests.post(self.config["quicknode_url"], 
                               json=payload,
                               headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            if "result" not in data or not data["result"]["value"]:
                return None
                
            return data["result"]["value"][0]["pubkey"]
            
        except Exception as e:
            self.logger.error(f"Error getting token account: {str(e)}")
            return None

    def _get_holder_transactions(self, token_account: str) -> List[Dict]:
        """Get transactions for token account"""
        try:
            self.rate_limiter.wait()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    token_account,
                    {"limit": self.config["transactions_per_holder"]}
                ]
            }
            
            response = requests.post(self.config["quicknode_url"], 
                               json=payload,
                               headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            if "result" not in data:
                return []
                
            transactions = []
            for sig_info in data["result"]:
                tx = self._get_transaction_details(sig_info["signature"])
                if tx:
                    transactions.append(tx)
                    
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error getting holder transactions: {str(e)}")
            return []

    def _get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Get detailed transaction information"""
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
            
            response = requests.post(self.config["quicknode_url"], 
                               json=payload,
                               headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            return data.get("result")
            
        except Exception as e:
            self.logger.error(f"Error getting transaction details: {str(e)}")
            return None

    def _analyze_transaction(self, transaction: Dict, token_mint: str, 
                           holder_address: str, token_account: str) -> List[EnhancedTokenFlow]:
        """Analyze transaction for token flows"""
        flows = []
        try:
            if not transaction or "transaction" not in transaction:
                return flows

            instructions = transaction["transaction"]["message"]["instructions"]
            for idx, instruction in enumerate(instructions):
                if "parsed" not in instruction:
                    continue

                parsed = instruction.get("parsed", {})
                if isinstance(parsed, dict):
                    program_id = instruction.get("programId")
                    
                    # Check for AMM interaction
                    if program_id in self.known_amm_programs:
                        if flow := self._analyze_amm_instruction(instruction, transaction,
                                                              token_mint, holder_address,
                                                              token_account):
                            flows.append(flow)
                    
                    # Check for token program instruction
                    elif program_id == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
                        if flow := self._analyze_token_instruction(instruction, transaction,
                                                                token_mint, holder_address,
                                                                token_account):
                            flows.append(flow)

        except Exception as e:
            self.logger.error(f"Error analyzing transaction: {str(e)}")
            
        return flows

    def _analyze_amm_instruction(self, instruction: Dict, transaction: Dict,
                               token_mint: str, holder_address: str,
                               token_account: str) -> Optional[EnhancedTokenFlow]:
        """Analyze AMM instruction for token movements"""
        try:
            program_id = instruction.get("programId")
            amm_name = self.known_amm_programs.get(program_id)
            
            # Basic AMM detection
            return EnhancedTokenFlow(
                from_address=f"{amm_name}_AMM",
                to_address=holder_address,
                amount=0,  # Amount would need to be calculated from token account changes
                timestamp=datetime.fromtimestamp(transaction["blockTime"]).isoformat(),
                method="swap",
                transaction_signature=transaction["transaction"]["signatures"][0],
                source_type=SOURCE_TYPES["AMM"],
                program_id=program_id,
                instruction_index=instruction.get("index", 0),
                additional_data={"amm_name": amm_name}
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing AMM instruction: {str(e)}")
            return None

    def _analyze_token_instruction(self, instruction: Dict, transaction: Dict,
                                 token_mint: str, holder_address: str,
                                 token_account: str) -> Optional[EnhancedTokenFlow]:
        """Analyze token program instruction for token movements"""
        try:
            info = instruction["parsed"].get("info", {})
            instruction_type = instruction["parsed"].get("type")
            
            if instruction_type not in TRANSACTION_TYPES:
                return None
                
            # Handle different instruction types
            if instruction_type == "mintTo" and info.get("account") == token_account:
                return EnhancedTokenFlow(
                    from_address="Mint",
                    to_address=holder_address,
                    amount=int(info["amount"]),
                    timestamp=datetime.fromtimestamp(transaction["blockTime"]).isoformat(),
                    method="mint",
                    transaction_signature=transaction["transaction"]["signatures"][0],
                    source_type=SOURCE_TYPES["MINT"],
                    additional_data={"mint_authority": info.get("mintAuthority")}
                )
            elif instruction_type in ["transfer", "transferChecked"]:
                destination = info.get("destination") or info.get("account")
                source = info.get("source") or info.get("authority")
                
                if destination == token_account:
                    amount = int(info.get("tokenAmount", {}).get("amount") or info.get("amount", 0))
                    return EnhancedTokenFlow(
                        from_address=source,
                        to_address=holder_address,
                        amount=amount,
                        timestamp=datetime.fromtimestamp(transaction["blockTime"]).isoformat(),
                        method=instruction_type,
                        transaction_signature=transaction["transaction"]["signatures"][0],
                        source_type=SOURCE_TYPES["WALLET"]
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing token instruction: {str(e)}")
            return None

    def _validate_holder_count(self, requested_count: Optional[int]) -> int:
        """Validate and adjust holder count based on configuration"""
        if requested_count is None:
            return self.config["min_holders"]
        
        return max(min(requested_count, self.config["max_holders"]), 
                  self.config["min_holders"])

    def save_analysis(self, analysis_dir: str, token_mint: str, flows: List[EnhancedTokenFlow], 
                     token_info: Dict, analysis_start: datetime):
        """Save comprehensive analysis results"""
        try:
            # Create directory structure
            raw_dir = os.path.join(analysis_dir, 'raw_data')
            report_dir = os.path.join(analysis_dir, 'reports')
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            
            # Save flows
            self._save_flow_data(raw_dir, flows)
            
            # Save token info
            with open(os.path.join(raw_dir, 'token_info.json'), 'w') as f:
                json.dump(token_info, f, indent=2)
            
            # Generate reports
            self._generate_summary_report(report_dir, flows, token_info, analysis_start)
            self._generate_detailed_report(report_dir, flows, token_info)
            self._generate_holder_reports(report_dir, flows)
            
            self.logger.info(f"Analysis saved to: {analysis_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")
            raise

    def _save_flow_data(self, raw_dir: str, flows: List[EnhancedTokenFlow]):
        flow_data = [self._flow_to_dict(flow) for flow in flows]
        
        # Save as JSON
        with open(os.path.join(raw_dir, 'flows.json'), 'w') as f:
            json.dump(flow_data, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(flow_data)
        df.to_csv(os.path.join(raw_dir, 'flows.csv'), index=False)
        
        # Save by source type
        by_source = {}
        for flow in flows:
            if flow.source_type not in by_source:
                by_source[flow.source_type] = []
            by_source[flow.source_type].append(self._flow_to_dict(flow))
        
        for source_type, source_flows in by_source.items():
            with open(os.path.join(raw_dir, f'flows_{source_type.lower()}.json'), 'w') as f:
                json.dump(source_flows, f, indent=2)

    @staticmethod
    def _flow_to_dict(flow: EnhancedTokenFlow) -> Dict:
        """Convert flow to dictionary format"""
        return {
            'from_address': flow.from_address,
            'to_address': flow.to_address,
            'amount': flow.amount,
            'timestamp': flow.timestamp,
            'method': flow.method,
            'source_type': flow.source_type,
            'transaction_signature': flow.transaction_signature,
            'program_id': flow.program_id,
            'instruction_index': flow.instruction_index,
            'additional_data': flow.additional_data
        }

    def _generate_summary_report(self, report_dir: str, flows: List[EnhancedTokenFlow], 
                               token_info: Dict, analysis_start: datetime):
        """Generate summary report"""
        report_path = os.path.join(report_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write("Token Flow Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Token Information
            f.write("Token Information:\n")
            f.write(f"Mint Address: {token_info['token_mint']}\n")
            f.write(f"Token Program: {token_info.get('token_program', 'Unknown')}\n")
            f.write(f"Decimals: {token_info.get('decimals', 0)}\n")
            f.write(f"Analysis Started: {analysis_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Flow Statistics
            source_types = {}
            total_volume = 0
            unique_addresses = set()
            
            for flow in flows:
                source_types[flow.source_type] = source_types.get(flow.source_type, 0) + 1
                total_volume += flow.amount
                if flow.from_address:
                    unique_addresses.add(flow.from_address)
                unique_addresses.add(flow.to_address)
            
            f.write("Flow Statistics:\n")
            f.write(f"Total Transactions: {len(flows):,}\n")
            f.write(f"Total Volume: {total_volume:,}\n")
            f.write(f"Unique Addresses: {len(unique_addresses):,}\n\n")
            
            f.write("Transaction Sources:\n")
            for source, count in source_types.items():
                f.write(f"- {source}: {count:,} ({count/len(flows)*100:.1f}%)\n")

    def _generate_detailed_report(self, report_dir: str, flows: List[EnhancedTokenFlow], 
                                token_info: Dict):
        """Generate detailed report"""
        report_path = os.path.join(report_dir, 'detailed_report.txt')
        with open(report_path, 'w') as f:
            f.write("Detailed Token Flow Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            by_recipient = {}
            for flow in flows:
                if flow.to_address not in by_recipient:
                    by_recipient[flow.to_address] = []
                by_recipient[flow.to_address].append(flow)
            
            for address, holder_flows in by_recipient.items():
                f.write(f"\nHolder Analysis: {address}\n")
                f.write("-" * 50 + "\n")
                
                total_received = sum(flow.amount for flow in holder_flows)
                sources = {}
                for flow in holder_flows:
                    sources[flow.source_type] = sources.get(flow.source_type, 0) + 1
                
                f.write(f"Total Tokens Received: {total_received:,}\n")
                f.write(f"Total Transactions: {len(holder_flows)}\n\n")
                
                f.write("Acquisition Sources:\n")
                for source, count in sources.items():
                    f.write(f"- {source}: {count} transactions\n")
                
                f.write("\nTransaction Timeline:\n")
                for flow in sorted(holder_flows, key=lambda x: x.timestamp):
                    f.write(f"\nTime: {flow.timestamp}\n")
                    f.write(f"Type: {flow.source_type}\n")
                    f.write(f"Amount: {flow.amount:,}\n")
                    if flow.from_address:
                        f.write(f"From: {flow.from_address}\n")
                    f.write(f"Tx: {flow.transaction_signature}\n")

    def _generate_holder_reports(self, report_dir: str, flows: List[EnhancedTokenFlow]):
        """Generate individual holder reports"""
        holder_dir = os.path.join(report_dir, 'holder_reports')
        os.makedirs(holder_dir, exist_ok=True)
        
        by_holder = {}
        for flow in flows:
            if flow.to_address not in by_holder:
                by_holder[flow.to_address] = []
            by_holder[flow.to_address].append(flow)
        
        for holder, holder_flows in by_holder.items():
            report_path = os.path.join(holder_dir, f'{holder}.txt')
            with open(report_path, 'w') as f:
                f.write(f"Holder Analysis: {holder}\n")
                f.write("=" * 50 + "\n\n")
                
                total_received = sum(flow.amount for flow in holder_flows)
                sources = {}
                for flow in holder_flows:
                    sources[flow.source_type] = sources.get(flow.source_type, 0) + 1
                
                f.write("Summary:\n")
                f.write(f"Total Tokens Received: {total_received:,}\n")
                f.write(f"Total Transactions: {len(holder_flows)}\n\n")
                
                f.write("Acquisition Sources:\n")
                for source, count in sources.items():
                    f.write(f"- {source}: {count} transactions "
                           f"({count/len(holder_flows)*100:.1f}%)\n")
                
                f.write("\nTransaction Timeline:\n")
                for flow in sorted(holder_flows, key=lambda x: x.timestamp):
                    f.write(f"\nTime: {flow.timestamp}\n")
                    f.write(f"Type: {flow.source_type}\n")
                    f.write(f"Amount: {flow.amount:,}\n")
                    if flow.from_address:
                        f.write(f"From: {flow.from_address}\n")
                    f.write(f"Tx: {flow.transaction_signature}\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze token flows on Solana')
    parser.add_argument('--token', '-t', required=True, help='Token mint address')
    parser.add_argument('--holders', '-n', type=int, help='Number of holders to analyze')
    parser.add_argument('--config', '-c', default='config.ini', help='Config file path')
    
    args = parser.parse_args()
    
    config = Config.load(args.config)
    analyzer = TokenAnalyzer(config)
    analyzer.analyze_token(args.token, args.holders)

if __name__ == "__main__":
    main()