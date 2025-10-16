You are an expert Python engineer and CS2/CS:GO trade-up specialist. Build me a full solution in 3 parts:

====================================================
PART 0 — PROGRAMMING OVERVIEW (CANONICAL TRADE-UP LOGIC)
====================================================
Implement the trade-up rules and formulas below exactly and use them throughout Parts 1–3.

A) Eligibility & constraints

- A trade-up uses exactly 10 input weapon skins of the SAME rarity grade (aka “quality”). Inputs must be either all StatTrak™ OR all non-StatTrak™ (never mixed). Souvenir, Knives, Contraband, and Covert inputs are not allowed. (Treat Covert as a possible OUTPUT tier but not usable as input.)
- The output is a single skin of the NEXT HIGHER rarity grade, drawn from one of the collections present among the inputs.
- Collections may be mixed. When mixed, the output collection is probabilistic (see B).

B) Outcome probabilities (collection mixing math)

- Let the 10 inputs belong to collections C1..Ck. Let nC be the number of inputs from collection C, and let mC be the number of possible output skins (of the next rarity) in collection C.
- Probability of a SPECIFIC output skin s in collection C:
  P(s) = (nC / 10) \* (1 / mC)
- The probabilities over all specific outputs must sum to 1 because ΣC [ (nC / 10) * (mC * 1/mC) ] = ΣC (nC/10) = 1.

C) Output wear / float formula

- Each skin has a float range [minFloat, maxFloat].
- Let X = average of the 10 input floats (plain mean of their float values).
- The output skin’s exact float is: outFloat = minOut + (maxOut - minOut) \* X
- Exterior (FN / MW / FT / WW / BS) is determined by outFloat versus standard buckets:
  FN: 0.00–0.07, MW: 0.07–0.15, FT: 0.15–0.38, WW: 0.38–0.45, BS: 0.45–1.00
- Some skins have restricted ranges; use their actual [minOut, maxOut] caps when mapping outFloat to exterior.

D) Steam Community Market fees (for EV / profit)

- Use a total 15% fee model (Steam 5% + CS2 game fee 10%).
- If “buyer pays” (list) price is P_list, seller receives:
  seller_receive = round_down_to_cents( P_list _ 0.85 ), with a $0.01 minimum fee per component considered by Steam. For simplicity, implement:
  receive_simple = round_to_cents(P_list _ 0.85)
- If you need the list price to achieve a target net R:
  list_price_for_net = ceil_to_cents( R / 0.85 )
- In our EV, assume input COST is the buyer-pays price from the database (i.e., what we pay to acquire ingredients). Output VALUE should be the seller’s net after fees when we sell the result.

E) Rarity ladder (weapon skins):

- Industrial Grade → Mil-Spec → Restricted → Classified → Covert
- Consumer Grade should be treated as not usable for trade-ups by default. Make this behavior configurable in case data requires otherwise.

F) Expected Value (EV) & ROI

- For a candidate trade-up T with outcomes s ∈ S:
  EV(T) = Σ_s [ P(s) * NetSellPrice(s) ] − TotalInputCost
- ROI(T) = EV(T) / TotalInputCost
- Mark trade-ups “profitable” if EV(T) > 0 (configurable threshold, e.g. min_roi).

G) Robustness assumptions

- Prices in the provided DB are Steam-market style. Default assumption: input costs are buyer-pays prices already including fees (what we spend). Output revenue uses fee deduction (0.85×). Add flags to flip this if needed.
- If floats for inputs are not available in the DB, provide two modes:
  (1) Float-agnostic EV: ignore float, use the database price for each specific output item as stored (which already encodes a specific exterior in its name).
  (2) Float-aware EV (optional): allow users to provide input floats; compute outFloat and map to the correct exterior SKU to pick the corresponding price.

====================================================
PART 1 — STANDALONE PYTHON ANALYZER (CLI SCRIPT)
====================================================
Create a single file, e.g. analyze_tradeups.py, that:

1. Loads the CS2/CS:GO skin prices database from this URL:
   https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json
   - If that exact path fails, also try:
     https://raw.githubusercontent.com/TheLime1/cs2-price-database/main/data/skins_database.json
   - Download once and cache to a local file with an ETag/Last-Modified check or simple timestamp cache.

2. Introspects the JSON structure dynamically.
   - Normalize each record into a unified schema with these fields where possible (fill None when absent):
     {
     "market_name": str, # e.g., "AK-47 | Redline (Field-Tested)"
     "weapon": str, # parsed from name if needed
     "skin": str, # finish name
     "exterior": str, # FN/MW/FT/WW/BS parsed from name
     "stattrak": bool, # parsed from name prefix "StatTrak™"
     "souvenir": bool, # parsed from name prefix "Souvenir"
     "collection": str | None,
     "rarity": str, # Industrial, Mil-Spec, Restricted, Classified, Covert
     "min_float": float | None,
     "max_float": float | None,
     "steam_price": float | None, # buyer-pays (prefer current/lowest-sell or avg)
     "last_update": str | None
     }
   - Provide heuristics: look for common keys like "steam", "price", "lowest_sell_order", etc. Parse exteriors from "(Factory New)" etc.

3. Build collection → outputs map
   - For each collection and rarity r, find its next-rarity outputs and count mC.
   - Maintain StatTrak and non-StatTrak groups separately.
   - Exclude ineligible types (Souvenir, Knives, Contraband; disallow Consumer Grade inputs by default; disallow Covert as input).

4. Generate candidate trade-ups efficiently
   - Explore common composition patterns to avoid combinatorial explosion: (10-0), (8-2), (7-3), (6-4), (5-5) across collections present at the same rarity.
   - For each composition, pick the CHEAPEST available inputs per collection that meet the required counts (respecting StatTrak flag), compute TotalInputCost.
   - Compute all specific outcomes with P(s) = (nC/10)\*(1/mC).
   - Resolve each specific output to the correct priced SKU (name includes its exterior). For float-agnostic mode, just use the SKU’s price. For float-aware mode, use provided input floats to compute outFloat and select the matching exterior SKU.

5. Compute EV and ROI (using fee model in D).
   - Output the top N profitable trade-ups (configurable), sorted by EV descending.
   - For each candidate, print:
     - Input summary: rarity, StatTrak, composition like “Anubis×8 + Italy×2”, total cost
     - Outcome table: [market_name, P(s)%, net_sell_price, contribution]
     - EV, ROI

6. Provide a clean CLI
   - Examples:
     python analyze_tradeups.py --min_roi 0.05 --rarity "Mil-Spec" --stattrak false --top 50
     python analyze_tradeups.py --float_aware --input_floats 0.03,0.04,0.02,0.01,0.05,0.02,0.03,0.04,0.02,0.01
     python analyze_tradeups.py --assume_input_costs_include_fees true
   - Flags:
     --rarity {Industrial,Mil-Spec,Restricted,Classified} (required)
     --stattrak {true,false} (default false)
     --min_roi FLOAT (default 0.0)
     --top INT (default 25)
     --float_aware (optional) and --input_floats CSV (optional)
     --assume_input_costs_include_fees {true,false} (default true)
     --allow_consumer_inputs {true,false} (default false)
     --cache_path PATH (default ./.cache/skins_database.json)

7. Code quality
   - Use Python 3.11+, requests, Flask (Part 2), pydantic for normalization, and rich/tabulate for pretty CLI tables.
   - Structure logic in reusable functions/classes: PricingModel, CollectionIndex, TradeUpCalculator.

====================================================
PART 2 — FLASK APP
====================================================
Create a minimal Flask app (app.py) that wraps the analyzer and serves:

- GET /health -> {"ok": true}
- GET /scan -> query params:
  rarity, stattrak (true/false), min_roi (default 0), top (default 50),
  float_aware (true/false), input_floats (csv), assume_input_costs_include_fees (true/false)
  Returns JSON with:
  {
  "params": {...},
  "generated_at": ISO8601,
  "results": [
  {
  "inputs": {
  "rarity": "Mil-Spec",
  "stattrak": false,
  "composition": {"Anubis": 8, "Italy": 2},
  "total_cost": 12.34
  },
  "outcomes": [
  {"market_name":"AK-47 | Steel Delta (MW)", "collection":"Anubis", "p":0.225, "price":16.40, "net":13.94, "contrib":3.14},
  ...
  ],
  "ev": 1.82,
  "roi": 0.148
  },
  ...
  ]
  }

- GET /ui -> Simple Tailwind or pure HTML page with a form (rarity, stattrak, min ROI, top N, toggle float-aware) and a results table with EV/ROI. Client uses fetch(/scan) and renders.

Implementation notes:

- Cache the parsed DB and the collection index in memory (and refresh every N minutes or on cold start).
- Keep endpoints fast; cap combinations; expose a “max_patterns” config.

====================================================
PART 3 — TESTING & VALIDATION
====================================================

- Unit tests for:
  - Probability math with mixtures (e.g., 8 from C1 with 4 outcomes, 2 from C2 with 3 outcomes → each C1 skin 8/10 _ 1/4 = 0.20; each C2 skin 2/10 _ 1/3 ≈ 0.0667).
  - Output float mapping using the formula and correct exterior bucket.
  - Steam fee helper: list_price_for_net and net_from_list with rounding to 2 decimals.
- Add a “dry-run” mode printing one fully worked example including all intermediate numbers.

====================================================
DELIVERABLES
====================================================

- analyze_tradeups.py (standalone CLI)
- app.py (Flask)
- requirements.txt
- README.md with:
  - Data source URL(s)
  - Fee assumptions and flags
  - Quickstart commands
  - Example responses and screenshots

Make the code production-ready, well-commented, and easy to extend.
