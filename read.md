# Solana Token Mapper

Solana Token Mapper is a lightweight command-line tool for analyzing token distribution on the Solana blockchain. Whether you're a developer tracking wallet concentration, a researcher monitoring project health, or a trader assessing token risk, this tool helps uncover wallet clusters, tag known addresses, and summarize distribution data.

## Features

- Scans any SPL token mint and exports holder data
- Auto-tags wallets using local tags and Solana RPC insights
- Detects wallet clusters by behavioral similarity
- Generates readable summaries with key metrics
- Lookup mode for wallet-level insight (cluster, tag, % supply)
- Optional chart generation and historical scan comparison

## Install

Clone this repo and install dependencies:

```bash
git clone https://github.com/Purfecto/Solana-Token-Mapper.git
cd Solana-Token-Mapper
pip install -r requirements.txt

Usage
Scan a token (with tagging and clustering):
bash
Copy
Edit
python main.py scan --mint <TOKEN_MINT> --tag --cluster
Example:

bash
Copy
Edit
python main.py scan --mint 7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr --tag --cluster
This saves a full snapshot of the token's holder structure in distributions/<mint>/<timestamp>/.

Lookup a wallet:
bash
Copy
Edit
python main.py lookup --wallet <WALLET_ADDRESS> --summary <PATH_TO_SUMMARY>
Example:

bash
Copy
Edit
python main.py lookup --wallet 8XAG2NezXoeKhKkyUpb91iRYGKr1sL2Cc4X9CHD34zrV --summary distributions/7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr/20250502_0120/summary.json