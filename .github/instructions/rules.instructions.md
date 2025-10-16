# CS2 Trade-Up Calculator - Rules & Guidelines

This document defines the core rules and constraints that govern the CS2 Trade-Up Calculator's behavior and recommendations.

## Rule 1: Weapon Consistency Requirement

**Rule**: If there is more than 1 weapon of the same type in the input selection, all instances of that weapon must have:
- **Same wear condition** (Factory New, Minimal Wear, Field-Tested, Well-Worn, or Battle-Scarred)
- **Same or lower float constraint** (all items must be obtainable with the unified float recommendation)

**Rationale**: This ensures realistic and practical buying recommendations where users can purchase items in bulk with consistent specifications.

**Implementation**: 
- The system identifies weapons by name (e.g., "AK-47", "Nova", "P250")
- When multiple items of the same weapon are selected, it uses the same variant for all
- Float recommendations are unified using the least restrictive (highest) float among variants
- Example: If selecting 4x Nova items, all will be "Nova | Blaze Orange (Well-Worn)" with "float≤0.40"

**Example Output**:
```
Items to Buy:
  • 4 x Nova | Blaze Orange (Well-Worn) $15.41, float≤0.40 (Total: $61.64)
  • 3 x MP7 | Teal Blossom (Battle-Scarred) $0.40, float≤0.59 (Total: $1.20)
```

## Rule 2: Core Trade-Up Mechanics

**Rule**: All trade-ups must follow CS2's canonical trade-up contract requirements:
- Exactly 10 input items
- All inputs must be the same rarity grade
- All inputs must be either all StatTrak™ OR all non-StatTrak™ (never mixed)
- Output is always the next higher rarity tier
- Collections can be mixed, affecting probability distribution

## Rule 3: Success Rate Definition

**Rule**: A trade-up is considered "successful" if the single output item's net selling price covers the total input basket cost.

**Formula**: `success_rate = sum(outcome.probability for outcome in outcomes if outcome.net_price >= total_input_cost)`

**Rationale**: This provides realistic probability of recovering the full investment, not just individual item costs.

## Rule 4: Fee Calculations

**Rule**: Use Steam Community Market's precise fee structure:
- 5% Steam platform fee (minimum $0.01)
- 10% CS2 game fee (minimum $0.01)
- $0.03 minimum listing price enforcement
- Component-based rounding for accuracy

## Rule 5: StatTrak Output Validation

**Rule**: StatTrak trade-ups are only recommended when StatTrak variants actually exist at the target rarity in the collection.

**Implementation**: Collections without valid StatTrak outputs are automatically excluded from StatTrak trade-up calculations.

## Rule 6: Consumer Grade Default Policy

**Rule**: Consumer Grade items are allowed as inputs by default to align with popular community calculators.

**Rationale**: Matches behavior of established tools like csgo.exchange and TradeUpSpy, despite some community ambiguity about Consumer→Industrial eligibility.

**Override**: Use `--allow_consumer_inputs false` to exclude Consumer Grade inputs if needed.

## Rule 7: Availability Handling

**Rule**: Items with missing availability data are deprioritized but not excluded from analysis.

**Implementation**: 
- Sort by availability (higher first), then price (lower first)
- Treat missing availability as 0 for sorting purposes
- This ensures broader market coverage while preferring liquid items

## Rule 8: Float Display Precision

**Rule**: Float recommendations are displayed with 2 decimal places for readability.

**Example**: `float≤0.40` instead of `float≤0.398`

**Note**: Internal calculations maintain full precision; only display is rounded.

## Rule 9: Collection Filtering

**Rule**: Only display collections that actually contribute inputs to the trade-up.

**Implementation**: Filter out collections with 0 input count from result displays to avoid showing "Collection Name (x0)".

## Rule 10: Market Simulation Parameters

**Rule**: Optional enhancement parameters for advanced market modeling:

- `--buy_slippage_pct`: Simulate higher purchase costs due to market variance
- `--sell_slippage_pct`: Simulate lower selling prices due to market variance  
- `--custom_fee_rate`: Override default 15% fee for specialized scenarios
- `--min_liquidity`: Filter outputs with insufficient market availability

**Default**: All parameters default to 0 or standard values for conservative estimates.

---

## Compliance Notes

These rules ensure the calculator provides:
- **Realistic recommendations** that can be executed in practice
- **Accurate calculations** matching Steam's actual fee behavior
- **Consistent behavior** aligned with established community tools
- **Transparent logic** with clear success probability definitions

Any deviations from these rules should be explicitly documented and justified.