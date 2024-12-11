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