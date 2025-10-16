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

    def get_analyzer(self, allow_consumer_inputs: bool = False) -> TradeUpAnalyzer:
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
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center text-blue-600 mb-8">CS2 Trade-Up Analyzer</h1>
        
        <div class="max-w-4xl mx-auto">
            <!-- Input Form -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-2xl font-semibold mb-4">Analysis Parameters</h2>
                <form id="analysisForm" class="space-y-4">
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Rarity</label>
                            <select id="rarity" name="rarity" class="w-full p-2 border border-gray-300 rounded-md">
                                <option value="Industrial">Industrial Grade</option>
                                <option value="Mil-Spec" selected>Mil-Spec Grade</option>
                                <option value="Restricted">Restricted</option>
                                <option value="Classified">Classified</option>
                            </select>
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">StatTrak</label>
                            <select id="stattrak" name="stattrak" class="w-full p-2 border border-gray-300 rounded-md">
                                <option value="false" selected>Normal</option>
                                <option value="true">StatTrak™</option>
                            </select>
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Min ROI (%)</label>
                            <input type="number" id="min_roi" name="min_roi" value="0" step="0.1" 
                                   class="w-full p-2 border border-gray-300 rounded-md">
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Top Results</label>
                            <input type="number" id="top" name="top" value="25" min="1" max="100" 
                                   class="w-full p-2 border border-gray-300 rounded-md">
                        </div>
                        
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Input Costs Include Fees</label>
                            <select id="assume_input_costs_include_fees" name="assume_input_costs_include_fees" 
                                    class="w-full p-2 border border-gray-300 rounded-md">
                                <option value="true" selected>Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                    </div>
                    
                    <button type="submit" class="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 transition duration-200">
                        Analyze Trade-Ups
                    </button>
                </form>
            </div>
            
            <!-- Loading Indicator -->
            <div id="loading" class="hidden text-center py-8">
                <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <p class="mt-4 text-gray-600">Analyzing trade-ups...</p>
            </div>
            
            <!-- Results -->
            <div id="results" class="space-y-6"></div>
        </div>
    </div>

    <script>
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
            
            // Convert min_roi from percentage to decimal
            const minRoi = parseFloat(params.get('min_roi')) / 100;
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
                    <div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded">
                        <h3 class="font-bold">No Results</h3>
                        <p>No profitable trade-ups found with the given parameters.</p>
                    </div>
                `;
                return;
            }
            
            let html = `
                <div class="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 rounded mb-6">
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
                    <div class="bg-white rounded-lg shadow-md p-6 mb-6">
                        <div class="mb-4">
                            <h3 class="text-xl font-semibold text-gray-800 mb-2">Trade-up #${index + 1}</h3>
                            <p class="text-sm text-gray-600"><strong>Collections:</strong> ${composition}</p>
                        </div>
                        
                        <!-- Key Metrics Grid -->
                        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
                            <div class="bg-blue-50 border border-blue-200 rounded-lg p-3">
                                <div class="text-xs text-blue-600 font-medium uppercase">Rarity</div>
                                <div class="text-sm font-semibold text-blue-800">${result.inputs.rarity}</div>
                            </div>
                            ${result.inputs.avg_input_float ? `
                            <div class="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <div class="text-xs text-gray-600 font-medium uppercase">Avg Input Float</div>
                                <div class="text-sm font-semibold text-gray-800">${result.inputs.avg_input_float.toFixed(4)}</div>
                            </div>
                            ` : ''}
                            <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                                <div class="text-xs text-yellow-600 font-medium uppercase">Input Cost</div>
                                <div class="text-sm font-semibold text-yellow-800">$${result.inputs.total_cost.toFixed(2)}</div>
                            </div>
                            <div class="bg-orange-50 border border-orange-200 rounded-lg p-3">
                                <div class="text-xs text-orange-600 font-medium uppercase">Expected Value</div>
                                <div class="text-sm font-semibold text-orange-800">$${(result.ev + result.inputs.total_cost).toFixed(2)}</div>
                            </div>
                            <div class="bg-green-50 border border-green-200 rounded-lg p-3">
                                <div class="text-xs text-green-600 font-medium uppercase">Profit</div>
                                <div class="text-sm font-semibold text-green-800">$${result.ev.toFixed(2)}</div>
                            </div>
                            <div class="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
                                <div class="text-xs text-cyan-600 font-medium uppercase">Profitability</div>
                                <div class="text-sm font-semibold text-cyan-800">${(result.roi * 100).toFixed(2)}%</div>
                            </div>
                            ${result.inputs.success_rate ? `
                            <div class="bg-teal-50 border border-teal-200 rounded-lg p-3">
                                <div class="text-xs text-teal-600 font-medium uppercase">Success Rate</div>
                                <div class="text-sm font-semibold text-teal-800">${(result.inputs.success_rate * 100).toFixed(1)}%</div>
                            </div>
                            ` : ''}
                        </div>
                        
                        <!-- Buy Recommendations -->
                        ${result.buy_recommendations && result.buy_recommendations.length > 0 ? `
                        <div class="mb-6">
                            <h4 class="font-medium text-gray-700 mb-3">Items to Buy</h4>
                            <div class="space-y-2">
                                ${result.buy_recommendations.map(rec => {
                                    const floatInfo = rec.recommended_float ? `, float≤${rec.recommended_float.toFixed(3)}` : '';
                                    const quantityText = rec.quantity > 1 ? `${rec.quantity} x ` : '';
                                    const totalPrice = rec.price * rec.quantity;
                                    const totalInfo = rec.quantity > 1 ? ` (Total: $${totalPrice.toFixed(2)})` : '';
                                    
                                    return `
                                        <div class="flex items-center justify-between bg-blue-50 border border-blue-200 rounded-lg p-3">
                                            <div>
                                                <span class="text-blue-600 font-medium">${quantityText}</span>
                                                <span class="font-semibold">${rec.market_name}</span>
                                                <span class="text-sm text-gray-600">$${rec.price.toFixed(2)}${floatInfo}${totalInfo}</span>
                                            </div>
                                            <button class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">
                                                Buy
                                            </button>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        <div class="overflow-x-auto">
                            <h4 class="font-medium text-gray-700 mb-2">Possible Outcomes</h4>
                            <table class="min-w-full table-auto">
                                <thead>
                                    <tr class="bg-gray-50">
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Probability</th>
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Item</th>
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Collection</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Price</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Net</th>
                                        <th class="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Contribution</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-gray-200">
                `;
                
                // Sort outcomes by probability (highest first)
                const sortedOutcomes = result.outcomes.sort((a, b) => b.p - a.p);
                
                sortedOutcomes.forEach(outcome => {
                    html += `
                        <tr>
                            <td class="px-4 py-2 text-sm font-medium">${(outcome.p * 100).toFixed(1)}%</td>
                            <td class="px-4 py-2 text-sm text-gray-900">${outcome.market_name}</td>
                            <td class="px-4 py-2 text-sm text-gray-600">${outcome.collection}</td>
                            <td class="px-4 py-2 text-sm text-right">$${outcome.price.toFixed(2)}</td>
                            <td class="px-4 py-2 text-sm text-right">$${outcome.net.toFixed(2)}</td>
                            <td class="px-4 py-2 text-sm text-right">$${outcome.contrib.toFixed(2)}</td>
                        </tr>
                    `;
                });
                
                html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            });            resultsDiv.innerHTML = html;
        }

        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded">
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


@app.route('/scan')
def scan():
    """Analyze trade-ups with query parameters."""
    try:
        # Parse parameters
        rarity = request.args.get('rarity', 'Mil-Spec')
        stattrak = request.args.get('stattrak', 'false').lower() == 'true'
        min_roi = float(request.args.get('min_roi', '0'))
        top = int(request.args.get('top', '50'))
        assume_input_costs_include_fees = request.args.get(
            'assume_input_costs_include_fees', 'true').lower() == 'true'

        # Validate rarity
        valid_rarities = ['Industrial', 'Mil-Spec', 'Restricted', 'Classified']
        if rarity not in valid_rarities:
            return jsonify({"error": f"Invalid rarity. Must be one of: {valid_rarities}"}), 400

        # Get analyzer
        analyzer = analyzer_cache.get_analyzer()

        # Perform analysis
        results = analyzer.analyze(
            rarity=rarity,
            stattrak=stattrak,
            min_roi=min_roi,
            top=top,
            assume_input_costs_include_fees=assume_input_costs_include_fees
        )

        # Format response
        response_data = {
            "params": {
                "rarity": rarity,
                "stattrak": stattrak,
                "min_roi": min_roi,
                "top": top,
                "assume_input_costs_include_fees": assume_input_costs_include_fees
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": []
        }

        for candidate in results:
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
                        "contrib": outcome.contribution
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
    print("Starting CS2 Trade-Up Analyzer Web Server...")
    print("Initializing analyzer cache...")

    # Pre-load the analyzer cache
    try:
        analyzer_cache.get_analyzer()
        print("Cache initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize cache: {e}")
        print("Cache will be loaded on first request")

    print("Server starting on http://localhost:5000")
    print("API endpoints:")
    print("  GET /health - Health check")
    print("  GET /scan - Analyze trade-ups")
    print("  GET /ui - Web interface")

    app.run(debug=True, host='0.0.0.0', port=5000)
