# CS2 Profitable Trade-Ups Calculator - Implementation Roadmap

## Overview
A Flask web application that calculates profitable CS2 trade-ups using real-time market data from the CS2 price database.

## Core Features
- Load and cache remote JSON data (12-hour refresh cycle)
- Model trade-up outcomes with correct probability calculations
- Filter by multiple criteria (max price, collection, success %, profitability %, StatTrak)
- Show expected value (EV), profit margin %, success probability, and outcome breakdowns
- CSV export functionality
- Graceful error handling and offline capability

## Architecture

### Project Structure
```
cs2-trade-up/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── roadmap.md                # This file
├── README.md                 # Usage documentation
├── tradeups/                 # Core business logic modules
│   ├── __init__.py
│   ├── data_loader.py        # Remote JSON fetch, cache, schema mapping
│   ├── price_model.py        # Item normalization, price selection
│   ├── rules.py              # Rarity progression, target enumeration
│   └── engine.py             # Candidate generation, EV calculations
├── templates/                # Jinja2 HTML templates
│   ├── base.html
│   └── index.html
├── static/                   # Static assets (optional CSS)
│   └── styles.css
├── data/                     # Local data cache
│   └── skins_database.json   # Cached market data
└── tests/                    # Unit tests
    ├── test_rules.py
    ├── test_engine.py
    └── test_prices.py
```

### Module Responsibilities

#### 1. data_loader.py
- Fetch data from: `https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json`
- Cache management with 12-hour TTL
- Schema variance handling (flexible field mapping)
- Error handling for network failures

#### 2. price_model.py
- `NormalizedItem` data class with type hints
- StatTrak detection logic
- Price selection strategy (prefer Steam lowest_sell, fallback to averages)
- Rarity rank mapping

#### 3. rules.py
- Rarity progression order: Consumer < Industrial < Mil-Spec < Restricted < Classified < Covert
- Alias mapping for different naming conventions
- Target enumeration per collection and rarity
- Trade-up validation rules

#### 4. engine.py
- Candidate generation using greedy algorithms
- Collection mix probability calculations
- Expected Value (EV) computation with configurable fees
- Filtering by user criteria
- Pagination support

### API Endpoints

#### GET /
- Main UI with filter form and results table
- Real-time filtering with JavaScript/HTMX
- Expandable outcome details
- CSV export functionality

#### GET /api/tradeups
Query parameters:
- `max_cost` (float): Maximum input item price
- `collection` (string): Target collection or "any"
- `min_success_pct` (float): Minimum success probability
- `min_profit_pct` (float): Minimum profitability percentage
- `stattrak` ("true"|"false"|"both"): StatTrak filter
- `fee_rate` (float, default 0.15): Market fee rate
- `page`, `page_size`: Pagination controls

Response format:
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
      "rarity": "Mil-Spec",
      "stattrak": false,
      "collection_mix": [{"collection": "Mirage 2021", "count": 10}],
      "avg_input_price": 1.12,
      "total_input_cost": 11.20,
      "success_pct": 100.0,
      "ev": 13.05,
      "margin_pct": 16.52,
      "outcomes": [...],
      "inputs_example": [...]
    }
  ]
}
```

#### GET /healthz
- Service health check
- Data freshness indicator

### Trade-Up Logic

#### Probability Calculation
1. **Collection Selection**: `P(collection) = count_from_collection / 10`
2. **Item Selection**: Uniform distribution within selected collection's valid targets
3. **Success Rate**: Only collections with valid next-tier items contribute to success
4. **Dead Outcomes**: Collections without next-tier items contribute to failure probability

#### Expected Value Formula
```
EV = Σ(probability(target) × net_price(target)) - total_input_cost
where net_price = market_price × (1 - fee_rate)

Profitability% = (EV / total_input_cost) × 100
```

#### Candidate Generation Strategy
1. **Mono-collection stacks**: 10 items from same collection
2. **Bi-collection mixes**: 7/3, 6/4, 5/5 splits between collections
3. **Performance caps**: 
   - MAX_POOL_PER_BUCKET = 50 items per rarity/collection
   - HARD_CAP_RESULTS = 500 total results
   - Greedy selection (cheapest items first)

### Configuration
```python
CONFIG = {
    'FEE_RATE_DEFAULT': 0.15,
    'CACHE_TTL_HOURS': 12,
    'MAX_POOL_PER_BUCKET': 50,
    'HARD_CAP_RESULTS': 500,
    'DATA_URL': 'https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json'
}
```

### Error Handling
- Network failures → use cached data with banner notification
- Schema changes → best-effort mapping with logging
- Invalid parameters → sanitization and defaults
- Missing data → graceful degradation

### Performance Targets
- Initial data load: < 2 seconds
- Filter operations: < 200ms
- API responses: < 500ms
- Memory usage: < 100MB for typical datasets

## Implementation Phases

### Phase 1: Core Infrastructure (todos 1-4)
- Project setup and structure
- Data loading and caching
- Item normalization and price modeling
- Basic trade-up rules

### Phase 2: Calculation Engine (todos 5-6)
- Candidate generation algorithms
- EV and probability calculations
- Filtering and pagination logic

### Phase 3: Web Interface (todos 7-9)
- Flask application setup
- HTML templates with Tailwind
- JavaScript for dynamic filtering
- CSV export functionality

### Phase 4: Testing and Documentation (todos 10-11)
- Comprehensive unit tests
- Integration testing
- Documentation and examples
- Performance optimization

## Technology Stack
- **Backend**: Flask 3.0+
- **Frontend**: Jinja2 templates, Vanilla JS/HTMX, Tailwind CSS (CDN)
- **Data**: JSON cache, Pydantic models
- **Testing**: pytest
- **Dependencies**: requests, python-dateutil, pydantic

## Success Criteria
✅ Runs with simple `python app.py`  
✅ Accurate trade-up probability and EV calculations  
✅ Real-time filtering with sub-200ms response times  
✅ Graceful handling of network failures and schema changes  
✅ Comprehensive test coverage  
✅ Clean, extensible code architecture  
✅ CSV export functionality  
✅ Mobile-responsive UI  

## Future Enhancements
- Client-side filter presets (localStorage)
- Wear tier price differentiation
- Historical profit tracking
- Advanced collection mixing strategies
- WebSocket for real-time price updates