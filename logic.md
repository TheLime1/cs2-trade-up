# CS2/CS:GO Trade-Up Calculator Logic Documentation

This document provides a comprehensive explanation of the calculation rules, formulas, and logic used in the CS2/CS:GO Trade-Up Analyzer. All calculations follow the canonical CS2 trade-up mechanics as documented in the official game specifications.

## Table of Contents

1. [Trade-Up Mechanics Overview](#trade-up-mechanics-overview)
2. [Data Model and Normalization](#data-model-and-normalization)
3. [Eligibility and Constraints](#eligibility-and-constraints)
4. [Collection Indexing](#collection-indexing)
5. [Outcome Probability Calculations](#outcome-probability-calculations)
6. [Float Value Calculations](#float-value-calculations)
7. [Steam Market Fee Calculations](#steam-market-fee-calculations)
8. [Input Selection Algorithm](#input-selection-algorithm)
9. [Expected Value and ROI Calculations](#expected-value-and-roi-calculations)
10. [Trade-Up Candidate Generation](#trade-up-candidate-generation)
11. [Result Filtering and Ranking](#result-filtering-and-ranking)

---

## Trade-Up Mechanics Overview

### Core Requirements
A CS2 trade-up contract requires exactly **10 input weapon skins** with the following constraints:

- All inputs must be of the **same rarity grade** (Consumer, Industrial, Mil-Spec, Restricted, or Classified)
- All inputs must be either **all StatTrak™** OR **all non-StatTrak™** (never mixed)
- The output is a **single skin** of the **next higher rarity grade**
- The output collection is determined probabilistically based on input collections

### Rarity Ladder
The progression follows this strict hierarchy:
```
Consumer Grade → Industrial Grade → Mil-Spec Grade → Restricted → Classified → Covert
```

**Important Notes:**
- Consumer Grade inputs are **enabled by default** to align with popular calculators like csgo.exchange and TradeUpSpy
- Set `allow_consumer_inputs=false` to exclude them if needed (some sources dispute Consumer eligibility due to community ambiguity about whether Consumer→Industrial trade-ups are valid)
- Covert items can be outputs but never inputs
- Souvenir items, Knives, and Contraband items are always excluded
- StatTrak trade-ups require StatTrak variants to exist at the target rarity - collections without StatTrak outputs at the target tier are automatically excluded

---

## Data Model and Normalization

### SkinData Structure
Each skin is normalized to this unified schema:

```python
{
    "market_name": str,           # e.g., "AK-47 | Redline (Field-Tested)"
    "weapon": str,                # e.g., "AK-47"
    "skin": str,                  # e.g., "Redline"
    "exterior": str,              # FN/MW/FT/WW/BS
    "stattrak": bool,             # StatTrak™ prefix detection
    "souvenir": bool,             # Souvenir prefix detection
    "collection": str,            # e.g., "The Cache Collection"
    "rarity": str,                # Standardized rarity names
    "min_float": float,           # Minimum float value for this skin
    "max_float": float,           # Maximum float value for this skin
    "steam_price": float,         # Current market price (buyer-pays)
    "availability": int,          # Number available on market
    "last_update": str            # Timestamp of price data
}
```

### Market Name Parsing Algorithm
The system parses market names using this logic:

1. **Prefix Detection:**
   - `StatTrak™ ` prefix → `stattrak = true`
   - `Souvenir ` prefix → `souvenir = true`

2. **Exterior Extraction:**
   - Find rightmost parentheses: `(Factory New)`, `(Field-Tested)`, etc.
   - Map to standardized exterior names

3. **Weapon/Skin Separation:**
   - Split on ` | ` delimiter
   - Left side = weapon name
   - Right side = skin finish name

---

## Eligibility and Constraints

### Input Eligibility Rules
An item is eligible as trade-up input if ALL conditions are met:

```python
def is_eligible_input(skin):
    # Must have required data
    if not (skin.collection and skin.rarity and skin.steam_price):
        return False
    
    # Must be available on market
    if skin.availability is None or skin.availability <= 0:
        return False
    
    # Cannot be Souvenir
    if skin.souvenir:
        return False
    
    # Cannot be knives (heuristic detection)
    if skin.weapon and any(knife in skin.weapon.lower() 
                          for knife in ['knife', 'bayonet', 'karambit']):
        return False
    
    # Cannot be Contraband
    if 'contraband' in skin.rarity.lower():
        return False
    
    # Cannot be Covert (output only)
    if skin.rarity == 'Covert':
        return False
    
    # Consumer Grade only if explicitly allowed
    if skin.rarity == 'Consumer Grade' and not allow_consumer_inputs:
        return False
    
    return True
```

### Output Eligibility Rules
An item is eligible as trade-up output if:

- It belongs to a collection that has eligible inputs at the lower rarity
- It has the correct rarity (one tier above inputs)
- It matches the StatTrak™ status of inputs **and** has a StatTrak variant available (if required)
- It has valid pricing data
- StatTrak outputs must actually exist in the target collection/rarity combination

---

## Collection Indexing

### Collection Mapping Structure
The system builds these indexes for efficient lookups:

```python
# Input items by rarity and StatTrak status
eligible_inputs: Dict[(rarity, stattrak)] = List[SkinData]

# Output items by collection, rarity, and StatTrak status  
collection_outputs: Dict[(collection, rarity, stattrak)] = List[SkinData]
```

### Collection Validation
For each collection at a given rarity:

- Count total possible outputs (`mC`)
- Verify at least one output exists
- Filter by availability and pricing data
- When mixing collections, probability mass per collection is proportional to its input count

---

## Outcome Probability Calculations

### Probability Formula
For a trade-up with inputs from collections `C1, C2, ..., Ck`:

**Variables:**
- `nC` = number of inputs from collection C (out of 10 total)
- `mC` = number of possible output skins in collection C at target rarity

**Probability of a specific output skin `s` in collection `C`:**
```
P(s) = (nC / 10) × (1 / mC)
```

### Probability Validation
The probabilities must sum to 1.0:
```
Σ_all_collections [ (nC / 10) × (mC × 1/mC) ] = Σ_all_collections (nC / 10) = 1
```

### Example Calculation
Trade-up with 8 items from Collection A (4 possible outputs) and 2 items from Collection B (3 possible outputs):

**Collection A outcomes:**
- Each of 4 skins: `P = (8/10) × (1/4) = 0.20` (20% each)

**Collection B outcomes:**  
- Each of 3 skins: `P = (2/10) × (1/3) = 0.0667` (6.67% each)

**Validation:** `4 × 0.20 + 3 × 0.0667 = 0.80 + 0.20 = 1.00` ✓

---

## Float Value Calculations

### Output Float Formula
Given input floats `[f1, f2, ..., f10]` and output skin range `[minOut, maxOut]`:

1. **Calculate average input float:**
   ```
   X = (f1 + f2 + ... + f10) / 10
   ```

2. **Map to output range:**
   ```
   outFloat = minOut + (maxOut - minOut) × X
   ```

### Exterior Mapping
Float values map to exterior conditions using these ranges:

| Exterior       | Float Range |
| -------------- | ----------- |
| Factory New    | 0.00 - 0.07 |
| Minimal Wear   | 0.07 - 0.15 |
| Field-Tested   | 0.15 - 0.38 |
| Well-Worn      | 0.38 - 0.45 |
| Battle-Scarred | 0.45 - 1.00 |

**Important:** Some skins have restricted ranges. The actual float is clamped to the skin's `[min_float, max_float]` before exterior determination.

### Float Recommendation Algorithm
For input selection, the system recommends floats using this strategy:

```python
def calculate_recommended_float(min_float, max_float):
    """Recommend float that's 25% into the range for cost/quality balance"""
    float_range = max_float - min_float
    return min_float + (float_range * 0.25)
```

This provides slightly better than average quality while keeping costs reasonable.

---

## Steam Market Fee Calculations

### Fee Structure
Steam Community Market applies a **15% total fee** with precise component-based logic:
- **5%** Steam platform fee (minimum $0.01)
- **10%** CS2 game fee (minimum $0.01)
- **$0.03** minimum listing price enforced

### Precise Fee Calculation
The system uses Steam's exact fee logic instead of simplified percentage calculations:

```python
def exact_fee_split(buyer_pays):
    # Enforce minimum listing price
    if buyer_pays < 0.03:
        buyer_pays = 0.03
    
    # Calculate individual fee components with minimums
    steam_fee = max(round(buyer_pays * 0.05, 2), 0.01)
    game_fee = max(round(buyer_pays * 0.10, 2), 0.01)
    
    # Seller receives the remainder
    seller_gets = round(buyer_pays - steam_fee - game_fee, 2)
    return seller_gets
```

### Fee Rounding Behavior
Steam's rounding behavior affects low-priced items significantly:
- Items under $0.03 cannot be listed
- Each fee component has a $0.01 minimum
- For a $0.05 item: steam_fee = $0.01, game_fee = $0.01, seller gets = $0.03
- For a $1.00 item: steam_fee = $0.05, game_fee = $0.10, seller gets = $0.85

### Market Variance and Slippage
The calculator supports optional market variance simulation:
- **Buy Slippage:** Increases input costs to simulate higher purchase prices
- **Sell Slippage:** Decreases output revenue to simulate lower selling prices  
- **Custom Fee Rates:** Override the default 15% for specialized market scenarios

---

## Input Selection Algorithm

### Selection Strategy (Balanced Approach)
The input selection algorithm balances cost-effectiveness with weapon consistency:

1. **Availability Prioritization:** Items with higher availability are preferred, but items with missing availability data are not excluded (just deprioritized)

2. **Price Optimization:** Select cheapest eligible inputs by price within each collection

3. **Weapon Consistency:** Items of the same weapon type use the same wear level and float recommendations for realistic trading scenarios

4. **Float Unification:** All items of the same weapon use unified float constraints to ensure obtainability

### Implementation Details

```python
def select_inputs(collection, required_count, available_items):
    # Sort by availability (higher first) then price (lower first)
    def sort_key(skin):
        availability = skin.availability if skin.availability is not None else 0
        price = skin.steam_price or float('inf')
        return (-availability, price)
    
    available_items.sort(key=sort_key)
    
    selected_weapons = {}  # Track weapon -> specific_variant mapping
    selected_items = []
    
    for i in range(required_count):
        item = available_items[i % len(available_items)]
        weapon_name = extract_weapon_name(item.market_name)
        
        # Use consistent variant for same weapon
        if weapon_name in selected_weapons:
            item = selected_weapons[weapon_name]
        else:
            selected_weapons[weapon_name] = item
            
        selected_items.append(item)
    
    return selected_items
```

### Float Consistency Rules
- All items of the same weapon use the same recommended float value
- Float recommendations use the least restrictive (highest) float among variants
- This ensures all recommended items are actually obtainable on the market

### Liquidity Guard
The system now includes an optional liquidity guard to filter out outputs with very low market availability:

- `min_liquidity` parameter sets minimum availability threshold
- Outputs below this threshold are excluded from consideration
- Helps avoid scenarios where profitable outcomes have no realistic liquidity
Identical items are grouped for user convenience:

```python
# Group identical items
item_counts = {}
for item in selected_items:
    key = item.market_name
    if key not in item_counts:
        item_counts[key] = {'item': item, 'count': 0}
    item_counts[key]['count'] += 1

# Create recommendations with quantities
recommendations = []
for item_data in item_counts.values():
    recommendations.append(BuyRecommendation(
        market_name=item_data['item'].market_name,
        collection=collection,
        price=item_data['item'].steam_price,
        quantity=item_data['count'],
        recommended_float=calculate_recommended_float(...)
    ))
```

---

## Expected Value and ROI Calculations

### Expected Value Formula
For a trade-up candidate `T` with outcomes `s ∈ S`:

```
EV(T) = Σ_s [ P(s) × NetSellPrice(s) ] - TotalInputCost
```

Where:
- `P(s)` = probability of outcome s
- `NetSellPrice(s)` = seller net amount after Steam fees  
- `TotalInputCost` = sum of input acquisition costs

### ROI Calculation
```
ROI(T) = EV(T) / TotalInputCost
```

### Success Rate Calculation
**Corrected Definition:** A trade-up is considered "successful" if the single output value covers the total input basket cost.

```python
def calculate_success_rate(outcomes, total_input_cost):
    profitable_outcomes = [
        o for o in outcomes 
        if o.net_price >= total_input_cost  # Single outcome covers entire input cost
    ]
    return sum(o.probability for o in profitable_outcomes)
```

**Important:** Previous versions incorrectly compared outcomes against average per-item cost (`total_cost / 10`), which overstated success rates. The corrected version provides realistic probability of recovering the full investment.

### Revenue Contribution Calculation
Each outcome's contribution to expected value, renamed for clarity:
```
expected_revenue_contribution = probability × net_selling_price
```

**Terminology Update:** Previously called "contribution", now renamed to "expected_revenue_contribution" to clarify that this represents the revenue component before subtracting input costs.

---

## Trade-Up Candidate Generation

### Composition Patterns
The system generates candidates using these collection mixing patterns:

| Pattern | Description       | Example                    |
| ------- | ----------------- | -------------------------- |
| (10, 0) | Single collection | 10 items from Collection A |
| (8, 2)  | Two collections   | 8 from A, 2 from B         |
| (7, 3)  | Two collections   | 7 from A, 3 from B         |
| (6, 4)  | Two collections   | 6 from A, 4 from B         |
| (5, 5)  | Two collections   | 5 from A, 5 from B         |

### Candidate Evaluation Process

```python
def evaluate_composition(composition, inputs_by_collection, target_rarity):
    # 1. Select cheapest inputs for each collection
    total_cost = 0.0
    selected_inputs = []
    
    for collection, count in composition.items():
        items = select_inputs(collection, count, inputs_by_collection[collection])
        selected_inputs.extend(items)
        total_cost += sum(item.steam_price for item in items)
    
    # 2. Calculate outcomes and probabilities
    outcomes = []
    for collection, count in composition.items():
        outputs = get_collection_outputs(collection, target_rarity)
        collection_prob = count / 10.0
        
        for output in outputs:
            probability = collection_prob / len(outputs)
            net_price = net_from_list_price(output.steam_price)
            contribution = probability * net_price
            
            outcomes.append(TradeUpOutcome(
                market_name=output.market_name,
                collection=collection,
                probability=probability,
                price=output.steam_price,
                net_price=net_price,
                contribution=contribution
            ))
    
    # 3. Calculate metrics
    expected_value = sum(o.contribution for o in outcomes) - total_cost
    roi = expected_value / total_cost if total_cost > 0 else 0.0
    
    return TradeUpCandidate(
        inputs=composition,
        total_cost=total_cost,
        outcomes=outcomes,
        expected_value=expected_value,
        roi=roi,
        # ... other fields
    )
```

---

## Result Filtering and Ranking

### Filtering Criteria
Candidates are filtered by:

1. **Minimum ROI:** `roi >= min_roi_threshold`
2. **Maximum Cost:** `total_cost <= max_cost_limit` (if specified)
3. **Availability:** All required inputs must be available on market

### Ranking Algorithm
Results are sorted by **Expected Value** in descending order:

```python
candidates.sort(key=lambda x: x.expected_value, reverse=True)
```

This prioritizes absolute profit over percentage returns, making larger profitable trades rank higher than smaller ones.

### Alternative Ranking Strategies
The system could be configured for different ranking approaches:

- **By ROI:** `key=lambda x: x.roi` - Prioritizes percentage returns
- **By Success Rate:** `key=lambda x: x.success_rate` - Prioritizes reliability
- **By Cost:** `key=lambda x: x.total_cost` - Prioritizes affordability

---

## Algorithm Complexity and Performance

### Time Complexity
- **Collection Indexing:** O(n) where n = total skins
- **Candidate Generation:** O(c² × p) where c = collections, p = patterns  
- **Outcome Calculation:** O(k × m) where k = candidates, m = avg outcomes per candidate

### Space Complexity
- **Indexes:** O(n) for skin storage and lookup tables
- **Results:** O(k × m) for all candidates and their outcomes

### Performance Optimizations
1. **Caching:** Database and parsed results cached for 24 hours
2. **Early Filtering:** Ineligible items filtered during data loading
3. **Lazy Evaluation:** Outcomes calculated only for viable compositions
4. **Price Sorting:** Input items pre-sorted by price for efficient selection

---

## Validation and Testing

### Unit Test Coverage
The system includes comprehensive validation for:

1. **Probability Math:** Verify probabilities sum to 1.0 for all compositions
2. **Float Calculations:** Test output float mapping and exterior determination  
3. **Fee Calculations:** Validate Steam fee formulas and rounding
4. **Edge Cases:** Handle missing data, zero prices, and boundary conditions

### Example Test Case: Probability Validation
```python
def test_probability_calculation():
    # Setup: 8 items from Collection A (4 outputs), 2 from Collection B (3 outputs)
    composition = {"Collection A": 8, "Collection B": 2}
    
    # Expected probabilities
    collection_a_prob = 8/10 * 1/4  # 0.20 per outcome
    collection_b_prob = 2/10 * 1/3  # 0.0667 per outcome
    
    # Validation
    total_prob = 4 * collection_a_prob + 3 * collection_b_prob
    assert abs(total_prob - 1.0) < 1e-6  # Should sum to 1.0
```

---

## Configuration Options

### Key Parameters
| Parameter                         | Default | Description                             |
| --------------------------------- | ------- | --------------------------------------- |
| `allow_consumer_inputs`           | `false` | Enable Consumer Grade items as inputs   |
| `assume_input_costs_include_fees` | `true`  | Whether input prices include Steam fees |
| `min_roi`                         | `0.0`   | Minimum ROI threshold for filtering     |
| `cache_refresh_minutes`           | `60`    | Database cache refresh interval         |

### Runtime Flags
- `--force_refresh`: Force database refresh regardless of cache age
- `--float_aware`: Enable float-based outcome prediction (experimental)
- `--max_cost`: Limit trade-ups by maximum input cost

---

## Known Limitations

1. **Float Prediction:** While mathematically correct, actual float outcomes may vary due to Valve's implementation details

2. **Market Volatility:** Prices change rapidly; cache may be slightly outdated

3. **Availability Accuracy:** Market availability is snapshot-based and may not reflect real-time status

4. **Network Latency:** Database downloads may fail in poor network conditions

5. **Memory Usage:** Large datasets (10k+ skins) may consume significant RAM

---

## Future Enhancements

### Potential Improvements
1. **Real-time Pricing:** WebSocket integration for live price updates
2. **Portfolio Optimization:** Multi-trade-up strategy optimization  
3. **Risk Analysis:** Monte Carlo simulation for outcome distributions
4. **Market Depth:** Consider volume and liquidity in calculations
5. **Machine Learning:** Predictive modeling for price movements

### Scalability Considerations
- **Database Sharding:** Split data by collection/rarity for faster queries
- **Async Processing:** Background calculation of candidate matrices
- **Result Caching:** Pre-compute popular trade-up scenarios
- **API Rate Limiting:** Protect against excessive database requests

---

*This documentation reflects the current implementation as of the latest version. All formulas and algorithms are based on official CS2/CS:GO trade-up contract mechanics and Steam Community Market policies.*