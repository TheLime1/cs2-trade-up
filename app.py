#!/usr/bin/env python3
"""
CS2/CS:GO Trade-Up Analyzer - Flask Web Application
Provides REST API and web interface for trade-up analysis.
"""

import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, render_template_string

from analyze_tradeups import (
    TradeUpAnalyzer, TradeUpCandidate
)


class AnalyzerCache:
    """Thread-safe cache for the analyzer instance."""

    def __init__(self, cache_refresh_minutes: int = 60):
        self.analyzer = None
        self.last_refresh = 0
        self.cache_refresh_minutes = cache_refresh_minutes
        self.lock = threading.Lock()

    def get_analyzer(self, allow_consumer_inputs: bool = True) -> TradeUpAnalyzer:
        """Get cached analyzer instance, refreshing if needed."""
        with self.lock:
            now = time.time()

            if (self.analyzer is None or
                    now - self.last_refresh > self.cache_refresh_minutes * 60):

                print(f"Refreshing analyzer cache at {datetime.now()}")
                self.analyzer = TradeUpAnalyzer()
                self.analyzer.load_data(
                    allow_consumer_inputs=allow_consumer_inputs)
                self.last_refresh = now

            return self.analyzer


# Global analyzer cache
analyzer_cache = AnalyzerCache()

# Flask app
app = Flask(__name__)


# HTML template for the web UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CS2 Trade-Up Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 min-h-screen text-white">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center text-blue-400 mb-8">CS2 Trade-Up Analyzer</h1>
        
        <div class="max-w-7xl mx-auto">
            <!-- Enhanced Filter Form -->
            <div class="bg-gray-800 rounded-lg border border-gray-600 p-6 mb-8">
                <form id="analysisForm" class="space-y-6">
                    <!-- Top row filters -->
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <!-- Cost Filter -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Cost</label>
                            <select id="cost_range" name="cost_range" class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="">All Costs</option>
                                <option value="0-5">$0 - $5</option>
                                <option value="5-10">$5 - $10</option>
                                <option value="10-20">$10 - $20</option>
                                <option value="20-30">$20 - $30</option>
                                <option value="30-50">$30 - $50</option>
                                <option value="50+">$50+</option>
                            </select>
                        </div>
                        
                        <!-- Collection Filter -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Collection</label>
                            <select id="collection" name="collection" class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="">All Collections</option>
                                <!-- Will be populated dynamically -->
                            </select>
                        </div>
                        
                        <!-- Min Profit % -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Min. Profit %</label>
                            <input type="number" id="min_roi" name="min_roi" value="0" step="1" placeholder="e.g. 120"
                                   class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400">
                        </div>
                        
                        <!-- StatTrak Filter -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">StatTrak™</label>
                            <select id="stattrak" name="stattrak" class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="">All</option>
                                <option value="false" selected>Normal</option>
                                <option value="true">StatTrak™</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Sort By row -->
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <!-- Sort By -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Sort By</label>
                            <select id="sort_by" name="sort_by" class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="highest_profit" selected>Highest Profit %</option>
                                <option value="lowest_profit">Lowest Profit %</option>
                                <option value="highest_cost">Highest Input Cost</option>
                                <option value="lowest_cost">Lowest Input Cost</option>
                                <option value="highest_ev">Highest Expected Value</option>
                                <option value="lowest_ev">Lowest Expected Value</option>
                            </select>
                        </div>
                        
                        <!-- Rarity -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Rarity</label>
                            <select id="rarity" name="rarity" class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="" selected>All Rarities</option>
                                <option value="Consumer">Consumer Grade</option>
                                <option value="Industrial">Industrial Grade</option>
                                <option value="Mil-Spec">Mil-Spec Grade</option>
                                <option value="Restricted">Restricted</option>
                                <option value="Classified">Classified</option>
                                <option value="Covert">Covert</option>
                            </select>
                        </div>
                        
                        <!-- Top Results -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Top Results</label>
                            <input type="number" id="top" name="top" value="25" min="1" max="100" 
                                   class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                        </div>
                        
                        <!-- Input Costs Include Fees -->
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Input Costs Include Fees</label>
                            <select id="assume_input_costs_include_fees" name="assume_input_costs_include_fees" 
                                    class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white">
                                <option value="true" selected>Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                    </div>
                    
                    <button type="submit" class="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 transition duration-200 font-medium">
                        Analyze Trade-Ups
                    </button>
                </form>
            </div>
            
            <!-- Loading Indicator -->
            <div id="loading" class="hidden text-center py-8">
                <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
                <p class="mt-4 text-gray-300">Analyzing trade-ups...</p>
            </div>
            
            <!-- Results -->
            <div id="results" class="space-y-6"></div>
        </div>
    </div>

    <script>
        // Load collections on page load
        async function loadCollections() {
            try {
                const response = await fetch('/collections');
                const data = await response.json();
                
                const collectionSelect = document.getElementById('collection');
                data.collections.forEach(collection => {
                    const option = document.createElement('option');
                    option.value = collection;
                    option.textContent = collection;
                    collectionSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load collections:', error);
            }
        }
        
        // Load collections when page loads
        document.addEventListener('DOMContentLoaded', loadCollections);

        // Handle form submission
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const params = new URLSearchParams();
            
            for (const [key, value] of formData.entries()) {
                if (value) {
                    params.append(key, value);
                }
            }
            
            // Handle cost range
            const costRange = params.get('cost_range');
            if (costRange) {
                params.delete('cost_range');
                if (costRange === '0-5') {
                    params.set('min_cost', '0');
                    params.set('max_cost', '5');
                } else if (costRange === '5-10') {
                    params.set('min_cost', '5');
                    params.set('max_cost', '10');
                } else if (costRange === '10-20') {
                    params.set('min_cost', '10');
                    params.set('max_cost', '20');
                } else if (costRange === '20-30') {
                    params.set('min_cost', '20');
                    params.set('max_cost', '30');
                } else if (costRange === '30-50') {
                    params.set('min_cost', '30');
                    params.set('max_cost', '50');
                } else if (costRange === '50+') {
                    params.set('min_cost', '50');
                }
            }
            
            // Convert percentages from whole numbers to decimals
            const minRoi = parseFloat(params.get('min_roi') || '0') / 100;
            params.set('min_roi', minRoi.toString());
            
            // Show loading
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch(`/scan?${params.toString()}`);
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    displayError(data.error || 'An error occurred');
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        });

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (!data.results || data.results.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="bg-yellow-900 border-l-4 border-yellow-500 text-yellow-200 p-4 rounded">
                        <h3 class="font-bold">No Results</h3>
                        <p>No profitable trade-ups found with the given parameters.</p>
                    </div>
                `;
                return;
            }
            
            let html = `
                <div class="bg-green-900 border-l-4 border-green-500 text-green-200 p-4 rounded mb-6">
                    <h3 class="font-bold">Analysis Complete</h3>
                    <p>Found ${data.results.length} profitable trade-up${data.results.length === 1 ? '' : 's'}</p>
                    <p class="text-sm mt-1">Generated at: ${new Date(data.generated_at).toLocaleString()}</p>
                </div>
            `;
            
            data.results.forEach((result, index) => {
                const composition = Object.entries(result.inputs.composition)
                    .map(([coll, count]) => `${coll} (x${count})`)
                    .join(', ');
                
                html += `
                    <div class="bg-gray-800 border border-gray-600 rounded-lg p-6 mb-6">
                        <!-- Collections header -->
                        <div class="mb-4">
                            <h4 class="text-sm font-medium text-gray-300 mb-2">Collections:</h4>
                            <p class="text-white font-medium">${composition}</p>
                        </div>
                        
                        <!-- Key Metrics Grid - 6 cards without Success Rate -->
                        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
                            <!-- 1. Rarity -->
                            <div class="bg-blue-900 border-l-4 border-blue-400 rounded-lg p-3">
                                <div class="text-xs text-blue-300 font-medium uppercase">Rarity:</div>
                                <div class="text-sm font-semibold text-blue-100">${result.inputs.rarity}</div>
                            </div>
                            
                            <!-- 2. Avg Input Float -->
                            <div class="bg-gray-700 border-l-4 border-gray-400 rounded-lg p-3">
                                <div class="text-xs text-gray-300 font-medium uppercase">Avg Input Float:</div>
                                <div class="text-sm font-semibold text-gray-100">${result.inputs.avg_input_float ? result.inputs.avg_input_float.toFixed(4) : 'N/A'}</div>
                            </div>
                            
                            <!-- 3. Input Cost -->
                            <div class="bg-yellow-900 border-l-4 border-yellow-400 rounded-lg p-3">
                                <div class="text-xs text-yellow-300 font-medium uppercase">Input Cost:</div>
                                <div class="text-sm font-semibold text-yellow-100">$${result.inputs.total_cost.toFixed(2)}</div>
                            </div>
                            
                            <!-- 4. Expected Value -->
                            <div class="bg-orange-900 border-l-4 border-orange-400 rounded-lg p-3">
                                <div class="text-xs text-orange-300 font-medium uppercase">Expected Value:</div>
                                <div class="text-sm font-semibold text-orange-100">$${(result.ev + result.inputs.total_cost).toFixed(2)}</div>
                            </div>
                            
                            <!-- 5. Profit -->
                            <div class="bg-green-900 border-l-4 border-green-400 rounded-lg p-3">
                                <div class="text-xs text-green-300 font-medium uppercase">Profit:</div>
                                <div class="text-sm font-semibold text-green-100">$${result.ev.toFixed(2)}</div>
                            </div>
                            
                            <!-- 6. Profitability -->
                            <div class="bg-cyan-900 border-l-4 border-cyan-400 rounded-lg p-3">
                                <div class="text-xs text-cyan-300 font-medium uppercase">Profitability:</div>
                                <div class="text-sm font-semibold text-cyan-100">${(result.roi * 100).toFixed(2)}%</div>
                            </div>
                        </div>
                        
                        <!-- Buy Recommendations -->
                        ${result.buy_recommendations && result.buy_recommendations.length > 0 ? `
                        <div class="mb-6">
                            <h4 class="font-medium text-gray-200 mb-3">Input Items</h4>
                            <div class="space-y-2">
                                ${result.buy_recommendations.map(rec => {
                                    const floatInfo = rec.recommended_float ? `, float≤${rec.recommended_float.toFixed(2)}` : '';
                                    const quantityText = rec.quantity > 1 ? `${rec.quantity} x ` : '';
                                    
                                    // Extract wear level from market name if present
                                    const wearMatch = rec.market_name.match(/\((Factory New|Minimal Wear|Field-Tested|Well-Worn|Battle-Scarred)\)/);
                                    const wear = wearMatch ? wearMatch[1] : '';
                                    
                                    // Create the display format: "3 x StatTrak™ MP9 Featherweight [Minimal Wear, $0.29, float≤0.13]"
                                    let displayName = rec.market_name;
                                    let priceAndDetails = '';
                                    
                                    if (wear) {
                                        // Remove wear from main name and put it in brackets with price and float
                                        displayName = rec.market_name.replace(/\s*\([^)]+\)/, '');
                                        priceAndDetails = `[${wear}, $${rec.price.toFixed(2)}${floatInfo}]`;
                                    } else {
                                        priceAndDetails = `[$${rec.price.toFixed(2)}${floatInfo}]`;
                                    }
                                    
                                    return `
                                        <div class="flex items-center justify-between bg-blue-900 border border-blue-600 rounded-lg p-3">
                                            <div class="flex-1">
                                                <div class="text-white">
                                                    <span class="text-blue-300 font-medium">${quantityText}</span>
                                                    <span class="font-semibold">${displayName}</span>
                                                    <span class="text-yellow-300 font-medium ml-2">${priceAndDetails}</span>
                                                </div>
                                            </div>
                                            <button class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 ml-3">
                                                Buy
                                            </button>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        <div class="overflow-x-auto">
                            <h4 class="font-medium text-gray-200 mb-2">Possible Outcomes</h4>
                            <table class="min-w-full table-auto">
                                <thead>
                                    <tr class="bg-gray-700">
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase">Probability</th>
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase">Item</th>
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase">Collection</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-300 uppercase">Price</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-300 uppercase">Net</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-300 uppercase">Contribution</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-gray-600">
                `;
                
                // Sort outcomes by probability (highest first)
                const sortedOutcomes = result.outcomes.sort((a, b) => b.p - a.p);
                
                sortedOutcomes.forEach(outcome => {
                    html += `
                        <tr class="bg-gray-800">
                            <td class="px-4 py-2 text-sm font-medium text-white">${(outcome.p * 100).toFixed(1)}%</td>
                            <td class="px-4 py-2 text-sm text-gray-200">${outcome.market_name}</td>
                            <td class="px-4 py-2 text-sm text-gray-300">${outcome.collection}</td>
                            <td class="px-4 py-2 text-sm text-right text-white">$${outcome.price.toFixed(2)}</td>
                            <td class="px-4 py-2 text-sm text-right text-white">$${outcome.net.toFixed(2)}</td>
                            <td class="px-4 py-2 text-sm text-right text-white">$${outcome.contrib.toFixed(2)}</td>
                        </tr>
                    `;
                });
                
                html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }

        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="bg-red-900 border-l-4 border-red-500 text-red-200 p-4 rounded">
                    <h3 class="font-bold">Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>
"""


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"ok": True})


@app.route('/collections')
def get_collections():
    """Get list of available collections."""
    try:
        analyzer = analyzer_cache.get_analyzer()
        collections = set()

        for skin in analyzer.skins:
            if skin.collection:
                collections.add(skin.collection)

        return jsonify({"collections": sorted(list(collections))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/scan')
def scan():
    """Analyze trade-ups with query parameters."""
    try:
        # Parse parameters
        rarity = request.args.get('rarity', 'Mil-Spec')
        stattrak = request.args.get('stattrak', 'false').lower() == 'true'
        min_roi = float(request.args.get('min_roi', '0'))
        min_cost = float(request.args.get('min_cost', '0'))
        max_cost = float(request.args.get('max_cost', '0'))
        collection = request.args.get('collection', '')
        sort_by = request.args.get('sort_by', 'highest_profit')
        top = int(request.args.get('top', '50'))
        assume_input_costs_include_fees = request.args.get(
            'assume_input_costs_include_fees', 'true').lower() == 'true'

        # Validate rarity
        valid_rarities = ['Consumer', 'Industrial',
                          'Mil-Spec', 'Restricted', 'Classified', 'Covert', '']
        if rarity not in valid_rarities:
            return jsonify({"error": f"Invalid rarity. Must be one of: {valid_rarities}"}), 400

        # Get analyzer
        analyzer = analyzer_cache.get_analyzer()

        # Handle "All Rarities" case - for now, default to Mil-Spec if empty
        if not rarity:
            rarity = 'Mil-Spec'

        # Perform analysis (with caching enabled)
        results = analyzer.analyze(
            rarity=rarity,
            stattrak=stattrak,
            min_roi=min_roi,
            top=top,
            assume_input_costs_include_fees=assume_input_costs_include_fees,
            use_cache=True  # Always use cache in Flask app for speed
        )

        # Apply additional filters
        filtered_results = []
        for candidate in results:
            # Filter by cost range
            if min_cost > 0 and candidate.total_cost < min_cost:
                continue
            if max_cost > 0 and candidate.total_cost > max_cost:
                continue

            # Filter by collection (if specified)
            if collection:
                collections_in_candidate = set(candidate.inputs.keys())
                if collection not in collections_in_candidate:
                    continue

            filtered_results.append(candidate)        # Sort results
        if sort_by == 'highest_profit':
            filtered_results.sort(key=lambda x: x.roi, reverse=True)
        elif sort_by == 'lowest_profit':
            filtered_results.sort(key=lambda x: x.roi)
        elif sort_by == 'highest_cost':
            filtered_results.sort(key=lambda x: x.total_cost, reverse=True)
        elif sort_by == 'lowest_cost':
            filtered_results.sort(key=lambda x: x.total_cost)
        elif sort_by == 'highest_ev':
            filtered_results.sort(key=lambda x: x.expected_value, reverse=True)
        elif sort_by == 'lowest_ev':
            filtered_results.sort(key=lambda x: x.expected_value)
        # Default is already sorted by profit (highest first)

        # Limit results after filtering and sorting
        filtered_results = filtered_results[:top]

        # Format response
        response_data = {
            "params": {
                "rarity": rarity,
                "stattrak": stattrak,
                "min_roi": min_roi,
                "min_cost": min_cost,
                "max_cost": max_cost,
                "collection": collection,
                "sort_by": sort_by,
                "top": top,
                "assume_input_costs_include_fees": assume_input_costs_include_fees
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": []
        }

        for candidate in filtered_results:
            result = {
                "inputs": {
                    "rarity": candidate.rarity,
                    "stattrak": candidate.stattrak,
                    "composition": candidate.inputs,
                    "total_cost": candidate.total_cost,
                    "avg_input_float": candidate.avg_input_float,
                    "success_rate": candidate.success_rate
                },
                "buy_recommendations": [
                    {
                        "market_name": rec.market_name,
                        "collection": rec.collection,
                        "price": rec.price,
                        "recommended_float": rec.recommended_float,
                        "quantity": rec.quantity
                    }
                    for rec in (candidate.buy_recommendations or [])
                ],
                "outcomes": [
                    {
                        "market_name": outcome.market_name,
                        "collection": outcome.collection,
                        "p": outcome.probability,
                        "price": outcome.price,
                        "net": outcome.net_price,
                        "contrib": outcome.expected_revenue_contribution
                    }
                    for outcome in candidate.outcomes
                ],
                "ev": candidate.expected_value,
                "roi": candidate.roi
            }
            response_data["results"].append(result)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ui')
def ui():
    """Serve the web UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/')
def index():
    """Redirect root to UI."""
    return ui()


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    import argparse

    # Add argument parsing for precompute option
    parser = argparse.ArgumentParser(
        description='CS2 Trade-Up Analyzer Flask Server')
    parser.add_argument('--precompute', action='store_true',
                        help='Pre-compute trade-up cache before starting server')
    args = parser.parse_args()

    print("Starting CS2 Trade-Up Analyzer Web Server...")
    print("Initializing analyzer cache...")

    # Pre-load the analyzer cache
    try:
        analyzer = analyzer_cache.get_analyzer()
        print("Cache initialized successfully")

        # Pre-compute trade-up combinations if requested
        if args.precompute:
            print("Pre-computing trade-up combinations for faster responses...")
            analyzer.precompute_cache()
            print("Pre-computation complete!")

    except Exception as e:
        print(f"Warning: Failed to initialize cache: {e}")
        print("Cache will be loaded on first request")

    print("Server starting on http://localhost:5000")
    print("API endpoints:")
    print("  GET /health - Health check")
    print("  GET /scan - Analyze trade-ups")
    print("  GET /ui - Web interface")

    if args.precompute:
        print("\nTrade-up cache has been pre-computed for faster responses!")

    app.run(debug=True, host='0.0.0.0', port=5000)
