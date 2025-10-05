# CS2 Trade-ups Calculator

A Flask-based web application that calculates profitable CS2 trade-up opportunities using real-time market data. Find the best trade-up combinations with accurate probability calculations and expected value analysis.

![CS2 Trade-ups Calculator](https://img.shields.io/badge/CS2-Trade--ups-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 🎯 Features

- **Real-time Data**: Loads market prices from [CS2 Price Database](https://github.com/TheLime1/cs2-price-database)
- **Smart Caching**: 12-hour cache TTL with offline fallback capability
- **Accurate Calculations**: Proper trade-up probability and expected value (EV) modeling
- **Advanced Filtering**: Filter by max cost, collection, success rate, profit margin, and StatTrak
- **Multiple Strategies**: Mono-collection and bi-collection trade-up generation
- **Responsive UI**: Clean, mobile-friendly interface with Tailwind CSS
- **CSV Export**: Download results for further analysis
- **API Endpoints**: JSON API for integration with other tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TheLime1/cs2-trade-up.git
   cd cs2-trade-up
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

That's it! The application will automatically fetch and cache the latest market data on first run.

## 📖 How to Use

### Web Interface

1. **Set Your Filters**
   - **Max Input Price**: Maximum price per input item (e.g., $10.00)
   - **Collection**: Filter to specific collections or "Any"
   - **Min Success %**: Minimum trade-up success probability
   - **Min Profit %**: Minimum profit margin percentage
   - **StatTrak Filter**: Choose StatTrak, Non-StatTrak, or both
   - **Fee Rate**: Steam market fee (default 15%)

2. **Calculate Trade-ups**
   Click "Calculate" to find profitable opportunities

3. **Analyze Results**
   - View expected value, profit margin, and success rates
   - Click "Details" to see all possible outcomes
   - Click "Copy" to copy input items to clipboard

4. **Export Data**
   Use the "CSV" button to export results for spreadsheet analysis

### API Usage

The application provides RESTful API endpoints for programmatic access:

**Get Trade-ups**
```http
GET /api/tradeups?max_cost=10.0&collection=Mirage&min_success_pct=80&min_profit_pct=10
```

**Export CSV**
```http
GET /api/tradeups/csv?max_cost=10.0&min_profit_pct=5
```

**Health Check**
```http
GET /healthz
```

See the [API Documentation](#api-documentation) section for complete details.

## 🔬 Trade-up Logic

### How Trade-ups Work

A CS2 trade-up uses **10 input skins** of the same grade/rarity and produces **1 output skin** of the next higher grade. The output is determined by:

1. **Collection Selection**: Choose a collection based on input proportions
   - Probability = (# of inputs from collection) / 10
   
2. **Item Selection**: Uniform random selection within the chosen collection's next-tier items

### Probability Calculation

```
Collection A: 7 inputs → 70% chance
Collection B: 3 inputs → 30% chance

If Collection A is chosen:
  - Each of its 4 higher-tier items: 70% / 4 = 17.5% chance
If Collection B is chosen:
  - Each of its 2 higher-tier items: 30% / 2 = 15% chance
```

### Expected Value Formula

```
EV = Σ(probability × net_price) - total_input_cost

Where:
- net_price = market_price × (1 - fee_rate)
- fee_rate = Steam market fee (default 15%)
```

### Success Rate

- **100%** if all input collections have next-tier items
- **< 100%** if some collections lack higher-tier items (dead outcomes)

## ⚙️ Configuration

The application can be configured by modifying the `CONFIG` dictionary in `app.py`:

```python
CONFIG = {
    'FEE_RATE_DEFAULT': 0.15,        # Default market fee (15%)
    'CACHE_TTL_HOURS': 12,           # Cache refresh interval
    'MAX_POOL_PER_BUCKET': 50,       # Max items per rarity/collection
    'HARD_CAP_RESULTS': 500,         # Maximum results to return
    'DATA_URL': 'https://...'        # Market data source URL
}
```

### Environment Variables

- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: true)
- `SECRET_KEY`: Flask secret key for sessions

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_rules.py -v
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_prices.py -v

# Run with coverage
python -m pytest tests/ --cov=tradeups --cov-report=html
```

## 📊 API Documentation

### GET /api/tradeups

Find profitable trade-up opportunities.

**Query Parameters:**
- `max_cost` (float): Maximum price per input item
- `collection` (string): Collection filter ("any" for all)
- `min_success_pct` (float): Minimum success percentage (0-100)
- `min_profit_pct` (float): Minimum profit margin percentage
- `stattrak` (string): "true", "false", or "both" (default)
- `fee_rate` (float): Market fee rate (0.0-1.0, default 0.15)
- `page` (int): Page number (default 1)
- `page_size` (int): Items per page (1-100, default 25)

**Response:**
```json
{
  "meta": {
    "page": 1,
    "page_size": 25,
    "total": 137,
    "fee_rate": 0.15,
    "data_age_minutes": 42
  },
  "filters": { ...echoed_filters... },
  "results": [
    {
      "rarity": "Mil-Spec Grade",
      "stattrak": false,
      "collection_mix": [{"collection": "Mirage Collection", "count": 10}],
      "avg_input_price": 1.12,
      "total_input_cost": 11.20,
      "success_pct": 100.0,
      "ev": 13.05,
      "margin_pct": 16.52,
      "outcomes": [
        {
          "name": "AK-47 | Point Disarray (Field-Tested)",
          "collection": "Mirage Collection",
          "price": 3.45,
          "prob": 0.25,
          "ev_contrib": 0.74
        }
      ],
      "inputs_example": ["P90 | Elite Build (FT)", "..."]
    }
  ]
}
```

### GET /api/tradeups/csv

Export trade-ups as CSV file. Accepts same parameters as `/api/tradeups` but returns CSV data instead of JSON.

### GET /healthz

Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "data_age_minutes": 42,
  "collections_count": 156,
  "engine_initialized": true
}
```

## 🏗️ Architecture

### Project Structure

```
cs2-trade-up/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── tradeups/                 # Core business logic
│   ├── __init__.py
│   ├── data_loader.py        # Data fetching and caching
│   ├── price_model.py        # Item normalization and pricing
│   ├── rules.py              # Trade-up rules and validation
│   └── engine.py             # Candidate generation and calculations
├── templates/                # Jinja2 HTML templates
│   ├── base.html
│   └── index.html
├── static/                   # Static assets
├── data/                     # Data cache directory
│   └── skins_database.json   # Cached market data
└── tests/                    # Unit tests
    ├── test_rules.py
    ├── test_engine.py
    └── test_prices.py
```

### Key Modules

- **data_loader.py**: Fetches market data with caching and schema flexibility
- **price_model.py**: Normalizes items and handles price selection strategies
- **rules.py**: Implements CS2 trade-up rules and target enumeration
- **engine.py**: Generates candidates using greedy algorithms and calculates EV

## 🔧 Development

### Adding New Features

1. **New Filters**: Add form fields in `templates/index.html` and API parameters in `app.py`
2. **New Algorithms**: Extend the `CandidateGenerator` class in `engine.py`
3. **New Data Sources**: Modify schema mapping in `data_loader.py`

### Performance Optimization

The application uses several optimization strategies:

- **Item Pool Limits**: `MAX_POOL_PER_BUCKET` limits items per rarity/collection
- **Result Caps**: `HARD_CAP_RESULTS` prevents excessive computation
- **Greedy Selection**: Prioritizes cheapest items for faster candidate generation
- **Caching**: 12-hour TTL reduces API calls and improves response times

## ⚠️ Important Notes

### Accuracy Disclaimer

- **Market Volatility**: Prices change frequently; results are estimates
- **Fee Variations**: Actual Steam fees may vary by item and region
- **Trade-up Mechanics**: Based on community understanding; Valve may change rules
- **Use at Your Own Risk**: This tool is for educational purposes

### Limitations

- **Data Freshness**: Market data updates every 12 hours by default
- **Collection Coverage**: Limited to items in the source database
- **Wear Tiers**: Currently uses single price per item (not per wear)
- **Regional Pricing**: Uses global averages, not region-specific prices

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Ensure tests pass**: `python -m pytest tests/ -v`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/cs2-trade-up.git
cd cs2-trade-up

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black ruff

# Run tests
python -m pytest tests/ -v

# Format code
black tradeups/ tests/
ruff check tradeups/ tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CS2 Price Database**: Market data provided by [TheLime1/cs2-price-database](https://github.com/TheLime1/cs2-price-database)
- **Flask Framework**: Web framework by the Pallets team
- **Tailwind CSS**: UI styling framework
- **CS2 Community**: For trade-up mechanics research and testing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/TheLime1/cs2-trade-up/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TheLime1/cs2-trade-up/discussions)
- **Documentation**: This README and inline code comments

---

**Happy Trading!** 🎮💰

> Remember: Past performance does not guarantee future results. Trade responsibly!