# CS2/CS:GO Trade-Up Analyzer

A comprehensive Python tool for analyzing profitable CS2/CS:GO trade-up contracts. This tool automatically analyzes the market database to identify the most profitable trade-up opportunities and provides specific **buy recommendations** with optimal float values.

## Features

- **Database-Driven Analysis**: Automatically analyzes all available market data
- **Buy Recommendations**: Tells you exactly which items to buy with specific floats
- **Availability Checking**: Only considers items actually available on the market
- **Accurate Trade-Up Logic**: Implements canonical CS2/CS:GO trade-up rules and formulas
- **Float Optimization**: Automatically calculates optimal float recommendations for inputs
- **Steam Market Integration**: Real-time price data from community market
- **Multiple Interfaces**: Both CLI tool and web application
- **Comprehensive Analysis**: Expected value, ROI, and probability calculations
- **Fee Handling**: Proper Steam Community Market fee calculations (15% total)

## What This Tool Does

Unlike other trade-up calculators that require you to input your existing items, this analyzer:

1. **Scans the entire database** of available CS2 skins
2. **Identifies profitable trade-up opportunities** automatically
3. **Tells you exactly which items to buy** from the Steam Market
4. **Recommends optimal float values** for each purchase
5. **Shows expected outcomes** with probabilities and profit calculations
6. **Considers market availability** so you only see realistic opportunities

## Data Sources

The analyzer fetches real-time CS2 skin price data from:
- **Primary**: `https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json`
- **Fallback**: `https://raw.githubusercontent.com/TheLime1/cs2-price-database/main/data/skins_database.json`

Data includes:
- Market prices
- Item availability/stock
- Float value ranges
- Collection and rarity information

Data is automatically cached locally for 24 hours to minimize API calls and improve performance.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheLime1/cs2-trade-up.git
   cd cs2-trade-up
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python analyze_tradeups.py --help
   ```

## Quick Start

### CLI Usage

```bash
# Basic analysis for Mil-Spec trade-ups
python analyze_tradeups.py --rarity "Mil-Spec" --min_roi 0.05

# StatTrak analysis with higher ROI threshold
python analyze_tradeups.py --rarity "Restricted" --stattrak true --min_roi 0.10 --top 10

# Show top 50 results with no minimum ROI
python analyze_tradeups.py --rarity "Industrial" --top 50
```

### Web Interface

```bash
# Start the Flask web server
python app.py

# Open your browser to http://localhost:5000
```

## CLI Parameters

| Parameter                           | Required | Default                        | Description                                                        |
| ----------------------------------- | -------- | ------------------------------ | ------------------------------------------------------------------ |
| `--rarity`                          | Yes      | -                              | Input rarity: `Industrial`, `Mil-Spec`, `Restricted`, `Classified` |
| `--stattrak`                        | No       | `false`                        | Use StatTrak items (`true`/`false`)                                |
| `--min_roi`                         | No       | `0.0`                          | Minimum ROI threshold (decimal, e.g., 0.05 for 5%)                 |
| `--top`                             | No       | `25`                           | Number of top results to display                                   |
| `--assume_input_costs_include_fees` | No       | `true`                         | Whether input prices include Steam fees                            |
| `--allow_consumer_inputs`           | No       | `false`                        | Allow Consumer Grade items as inputs                               |
| `--cache_path`                      | No       | `./.cache/skins_database.json` | Database cache file path                                           |
| `--force_refresh`                   | No       | `false`                        | Force refresh of cached database                                   |

## API Endpoints

### Web Application
- **GET /** - Web interface
- **GET /ui** - Web interface
- **GET /health** - Health check endpoint

### REST API
- **GET /scan** - Analyze trade-ups with query parameters

#### Example API Call
```bash
curl "http://localhost:5000/scan?rarity=Mil-Spec&stattrak=false&min_roi=0.05&top=10"
```

#### Example API Response
```json
{
  "params": {
    "rarity": "Mil-Spec",
    "stattrak": false,
    "min_roi": 0.05,
    "top": 10
  },
  "generated_at": "2024-01-01T12:00:00Z",
  "results": [
    {
      "inputs": {
        "rarity": "Mil-Spec",
        "stattrak": false,
        "composition": {"Anubis": 8, "Italy": 2},
        "total_cost": 12.34
      },
      "outcomes": [
        {
          "market_name": "AK-47 | Steel Delta (Minimal Wear)",
          "collection": "Anubis",
          "p": 0.225,
          "price": 16.40,
          "net": 13.94,
          "contrib": 3.14
        }
      ],
      "ev": 1.82,
      "roi": 0.148
    }
  ]
}
```

## Trade-Up Rules & Assumptions

### Eligibility Constraints
- **Input Requirements**: Exactly 10 items of the same rarity grade
- **StatTrak Consistency**: All inputs must be either StatTrak™ OR normal (never mixed)
- **Prohibited Items**: Souvenir, Knives, Contraband items cannot be used as inputs
- **Rarity Ladder**: Consumer Grade → Industrial Grade → Mil-Spec → Restricted → Classified → Covert
- **Output Tier**: Always the next higher rarity from the input tier

### Probability Mathematics
For inputs from collections C₁, C₂, ..., Cₖ:
- Let nᶜ = number of inputs from collection C
- Let mᶜ = number of possible outputs in collection C at the target rarity
- Probability of specific output s in collection C: **P(s) = (nᶜ / 10) × (1 / mᶜ)**

### Float Value Calculations
- **Output Float Formula**: `output_float = min_out + (max_out - min_out) × average_input_float`
- **Exterior Mapping**:
  - Factory New: 0.00 - 0.07
  - Minimal Wear: 0.07 - 0.15
  - Field-Tested: 0.15 - 0.38
  - Well-Worn: 0.38 - 0.45
  - Battle-Scarred: 0.45 - 1.00

### Steam Market Fees
- **Total Fee Rate**: 15% (5% Steam + 10% Game)
- **Seller Receives**: `list_price × 0.85`
- **List Price for Target Net**: `net_amount ÷ 0.85`

### Expected Value Calculation
- **Expected Value**: `Σ[P(outcome) × net_sell_price] - total_input_cost`
- **ROI**: `expected_value ÷ total_input_cost`

## Example Analysis Output

```
Trade-up #1
Input: Mil-Spec (Normal)
Composition: Anubis×8, Italy×2
Total Cost: $12.34

┌─────────────────────────────────────┬────────────┬─────────────┬─────────┬─────────┬──────────────┐
│ Item                                │ Collection │ Probability │ Price   │ Net     │ Contribution │
├─────────────────────────────────────┼────────────┼─────────────┼─────────┼─────────┼──────────────┤
│ AK-47 | Steel Delta (Minimal Wear)  │ Anubis     │ 22.5%       │ $16.40  │ $13.94  │ $3.14        │
│ P250 | Cyber Shell (Minimal Wear)   │ Anubis     │ 22.5%       │ $8.20   │ $6.97   │ $1.57        │
│ M4A1-S | Printstream (Minimal Wear) │ Italy      │ 20.0%       │ $24.50  │ $20.83  │ $4.17        │
└─────────────────────────────────────┴────────────┴─────────────┴─────────┴─────────┴──────────────┘

Expected Value: $1.82
ROI: 14.8%
```

## Configuration Flags

### Fee Assumptions
- `--assume_input_costs_include_fees true`: Input prices are buyer-pays prices (what you spend)
- `--assume_input_costs_include_fees false`: Input prices are seller-net prices (add fees)

### Float Calculations
- `--float_aware false`: Use database prices as-is (exterior already encoded in item name)
- `--float_aware true`: Calculate output float and map to correct exterior SKU

### Input Restrictions
- `--allow_consumer_inputs false`: Exclude Consumer Grade items from trade-up inputs
- `--allow_consumer_inputs true`: Allow Consumer Grade items (if database contains them)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_analyzer.py::TestSteamFeeCalculator -v
python -m pytest tests/test_analyzer.py::TestProbabilityMath -v

# Run with dry-run example
python -m pytest tests/test_analyzer.py::TestFullWorkflowExample::test_dry_run_example -v -s
```

## Development

### Project Structure
```
cs2-trade-up/
├── analyze_tradeups.py    # Main CLI analyzer
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── tests/
│   └── test_analyzer.py   # Unit tests
├── .cache/               # Database cache directory
└── README.md            # This file
```

### Key Classes
- **`TradeUpAnalyzer`**: Main orchestration class
- **`DatabaseLoader`**: Handles data fetching and caching
- **`CollectionIndex`**: Builds collection mappings for analysis
- **`TradeUpCalculator`**: Computes probabilities and expected values
- **`SteamFeeCalculator`**: Handles Steam Market fee calculations
- **`FloatCalculator`**: Manages float value calculations and exterior mapping

## Troubleshooting

### Common Issues

1. **"No eligible inputs found"**
   - Check if the specified rarity has available items in the database
   - Verify StatTrak setting matches available items
   - Try enabling `--allow_consumer_inputs` if using Consumer Grade

2. **"Failed to download database"**
   - Check internet connection
   - Verify the database URLs are accessible
   - Try `--force_refresh` to bypass cache

3. **"Invalid input_floats"**
   - Ensure exactly 10 comma-separated float values
   - Float values should be between 0.0 and 1.0
   - Example: `0.03,0.04,0.02,0.01,0.05,0.02,0.03,0.04,0.02,0.01`

4. **Web interface not loading**
   - Ensure Flask dependencies are installed
   - Check that port 5000 is not in use
   - Try accessing `http://127.0.0.1:5000` instead of `localhost`

### Performance Tips

- Database is cached for 24 hours - use `--force_refresh` only when needed
- Limit `--top` parameter for faster results
- Use specific rarity filters to reduce computation time
- For programmatic use, prefer the CLI tool over the web interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Submit a pull request

## License

This project is open source. See the repository for license details.

## Disclaimer

This tool is for educational and analytical purposes only. CS2/CS:GO trading involves risk, and market prices can be volatile. Always verify calculations independently and trade responsibly. The authors are not responsible for any trading losses.