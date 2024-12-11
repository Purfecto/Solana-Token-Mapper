# Solana Token Mapper

Detailed token holder analysis tool for Solana blockchain.

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Configuration

Before running the analyzer, you need to:

1. Add your QuickNode URL to \`config/config.ini\`:
   ```ini
   [API]
   quicknode_url = # Add your QuickNode URL here

## Usage

\`\`\`bash
python src/token_analysis/token_analyzer.py --token <token_mint_address> --holders <number_of_holders>
\`\`\`
=======
 Solana Token Flow Analyzer

A Python tool for analyzing token holder distributions and tracking how top holders acquired their tokens on Solana.

## Features

- Track token creation and distribution history
- Analyze top holder acquisition patterns
- Generate detailed flow analysis reports
- Export data in CSV and JSON formats
- Rate-limited API interactions
- Progress tracking for long operations

## Setup

1. Get a QuickNode API Key:
   - Sign up at [QuickNode](https://quicknode.com)
   - Create a Solana endpoint
   - Copy your API URL

2. Configure your API:
   - Copy `config.ini.example` to `config.ini`
   - Replace `YOUR_QUICKNODE_URL_HERE` with your QuickNode URL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/solana-token-mapper.git
cd solana-token-mapper

Install required packages:

bashCopypip install -r requirements.txt

Configure your settings in config.ini

Usage
Basic usage:
bashCopypython token_mapper.py --token YOUR_TOKEN_MINT_ADDRESS
Advanced options:
bashCopypython token_mapper.py --token TOKEN_MINT_ADDRESS --holders 20 --transactions 200
Command Line Arguments

--token, -t: Token mint address to analyze
--config, -c: Path to configuration file (default: config.ini)
--holders, -n: Number of top holders to analyze
--transactions, -tx: Number of transactions to analyze per holder

Output Structure
The tool generates a structured analysis in the following format:
Copytoken_analysis/
└── TOKEN_ADDRESS/
    └── analysis_TIMESTAMP/
        ├── analysis_log.txt     # Complete analysis log
        ├── flow_analysis.txt    # Detailed flow report
        ├── token_flows.csv      # Transaction data
        └── raw_data/
            └── holders_data.json # Top holder information
Configuration
Edit config.ini to customize:

QuickNode RPC URL
Rate limiting
Default number of holders to analyze
Transactions per holder limit

Requirements

Python 3.7+
pandas
requests
tqdm

Example Output
Analysis includes:

Token creation info
Current holder distribution
Acquisition timelines
Transfer history
Summary statistics
>>>>>>> 8dc900ed708109dbfc8382e3e5f1abd95b4bff35
