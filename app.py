"""
Flask application for CS2 Trade-ups Calculator.

Main entry point for the web application.
"""

import csv
import io
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, make_response

from tradeups.data_loader import load_skins_data
from tradeups.price_model import normalize_items
from tradeups.engine import TradeUpEngine

# Configuration
CONFIG = {
    'FEE_RATE_DEFAULT': 0.15,
    'CACHE_TTL_HOURS': 12,
    'MAX_POOL_PER_BUCKET': 50,
    'HARD_CAP_RESULTS': 500,
    'DATA_URL': 'https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json'
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Global variables for caching
_engine: TradeUpEngine = None
_data_last_updated: datetime = None
_available_collections: List[str] = []


def get_engine() -> TradeUpEngine:
    """Get or initialize the trade-up engine."""
    global _engine, _data_last_updated, _available_collections

    if _engine is None:
        logger.info("Initializing trade-up engine...")

        # Load data
        raw_data, last_updated = load_skins_data(
            data_url=CONFIG['DATA_URL'],
            cache_path='data/skins_database.json',
            cache_ttl_hours=CONFIG['CACHE_TTL_HOURS']
        )

        # Normalize items
        items = normalize_items(raw_data)
        logger.info("Normalized %d items", len(items))

        # Build engine
        _engine = TradeUpEngine(
            items,
            max_pool_per_bucket=CONFIG['MAX_POOL_PER_BUCKET'],
            hard_cap_results=CONFIG['HARD_CAP_RESULTS']
        )

        # Cache metadata
        _data_last_updated = last_updated

        # Extract available collections
        collections = set()
        for item in items:
            if item.collection and item.is_tradeable:
                collections.add(item.collection)
        _available_collections = sorted(list(collections))

        logger.info("Engine initialized with %d collections",
                    len(_available_collections))

    return _engine


def get_data_age_minutes() -> int:
    """Get the age of cached data in minutes."""
    if _data_last_updated is None:
        return 0

    age = datetime.now() - _data_last_updated
    return int(age.total_seconds() / 60)


@app.route('/')
def index():
    """Main page with trade-up calculator interface."""
    try:
        # Initialize engine to get collections
        get_engine()

        return render_template(
            'index.html',
            collections=_available_collections,
            default_fee_rate=CONFIG['FEE_RATE_DEFAULT'],
            data_age_minutes=get_data_age_minutes()
        )
    except Exception as e:
        logger.error("Error loading main page: %s", str(e))
        return render_template(
            'index.html',
            collections=[],
            default_fee_rate=CONFIG['FEE_RATE_DEFAULT'],
            data_age_minutes=0,
            error=f"Error loading data: {str(e)}"
        )


@app.route('/api/tradeups')
def api_tradeups():
    """API endpoint for trade-up calculations."""
    try:
        # Parse query parameters
        max_cost = request.args.get('max_cost', type=float)
        collection = request.args.get('collection')
        min_success_pct = request.args.get('min_success_pct', type=float)
        min_profit_pct = request.args.get('min_profit_pct', type=float)
        stattrak = request.args.get('stattrak', 'both')
        fee_rate = request.args.get(
            'fee_rate', CONFIG['FEE_RATE_DEFAULT'], type=float)
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 25, type=int)

        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 25
        if fee_rate < 0 or fee_rate > 1:
            fee_rate = CONFIG['FEE_RATE_DEFAULT']

        # Get engine and find trade-ups
        engine = get_engine()
        candidates, total_count = engine.find_profitable_tradeups(
            max_cost=max_cost,
            collection=collection,
            min_success_pct=min_success_pct,
            min_profit_pct=min_profit_pct,
            stattrak=stattrak,
            page=page,
            page_size=page_size
        )

        # Convert candidates to API response format
        results = []
        for candidate in candidates:
            # Prepare outcomes
            outcomes = []
            for outcome in candidate.outcomes:
                outcomes.append({
                    'name': outcome['name'],
                    'collection': outcome['collection'],
                    'price': round(outcome['price'], 2),
                    'prob': round(outcome['probability'], 4),
                    'ev_contrib': round(outcome['probability'] * outcome['price'] * (1 - fee_rate), 2)
                })

            results.append({
                'rarity': candidate.rarity,
                'stattrak': candidate.stattrak,
                'collection_mix': candidate.collection_mix,
                'avg_input_price': round(candidate.avg_input_price, 2),
                'total_input_cost': round(candidate.total_input_cost, 2),
                'success_pct': round(candidate.success_pct, 2),
                'ev': round(candidate.ev, 2),
                'margin_pct': round(candidate.margin_pct, 2),
                'outcomes': outcomes,
                'inputs_example': candidate.inputs_example
            })

        # Prepare response
        response = {
            'meta': {
                'page': page,
                'page_size': page_size,
                'total': total_count,
                'fee_rate': fee_rate,
                'data_age_minutes': get_data_age_minutes()
            },
            'filters': {
                'max_cost': max_cost,
                'collection': collection,
                'min_success_pct': min_success_pct,
                'min_profit_pct': min_profit_pct,
                'stattrak': stattrak
            },
            'results': results
        }

        return jsonify(response)

    except Exception as e:
        logger.error("Error in API endpoint: %s", str(e))
        return jsonify({
            'error': str(e),
            'meta': {'data_age_minutes': get_data_age_minutes()},
            'results': []
        }), 500


@app.route('/api/tradeups/csv')
def api_tradeups_csv():
    """Export trade-ups as CSV."""
    try:
        # Get same parameters as regular API
        max_cost = request.args.get('max_cost', type=float)
        collection = request.args.get('collection')
        min_success_pct = request.args.get('min_success_pct', type=float)
        min_profit_pct = request.args.get('min_profit_pct', type=float)
        stattrak = request.args.get('stattrak', 'both')
        fee_rate = request.args.get(
            'fee_rate', CONFIG['FEE_RATE_DEFAULT'], type=float)

        # Get all results (no pagination for CSV)
        engine = get_engine()
        candidates, _ = engine.find_profitable_tradeups(
            max_cost=max_cost,
            collection=collection,
            min_success_pct=min_success_pct,
            min_profit_pct=min_profit_pct,
            stattrak=stattrak,
            page=1,
            page_size=CONFIG['HARD_CAP_RESULTS']  # Get all available
        )

        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Rarity',
            'StatTrak',
            'Collections',
            'Avg Input Price',
            'Total Input Cost',
            'Success %',
            'Expected Value',
            'Profit Margin %',
            'Top Outcome',
            'Top Outcome Price',
            'Top Outcome Probability'
        ])

        # Data rows
        for candidate in candidates:
            # Get top outcome
            top_outcome = max(
                candidate.outcomes, key=lambda x: x['probability']) if candidate.outcomes else None

            # Collection mix as string
            collections_str = ', '.join([
                f"{mix['count']}× {mix['collection']}"
                for mix in candidate.collection_mix
            ])

            writer.writerow([
                candidate.rarity,
                'Yes' if candidate.stattrak else 'No',
                collections_str,
                f"${candidate.avg_input_price:.2f}",
                f"${candidate.total_input_cost:.2f}",
                f"{candidate.success_pct:.2f}%",
                f"${candidate.ev:.2f}",
                f"{candidate.margin_pct:.2f}%",
                top_outcome['name'] if top_outcome else 'N/A',
                f"${top_outcome['price']:.2f}" if top_outcome else 'N/A',
                f"{top_outcome['probability'] * 100:.2f}%" if top_outcome else 'N/A'
            ])

        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=cs2_tradeups.csv'

        return response

    except Exception as e:
        logger.error("Error exporting CSV: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/healthz')
def health_check():
    """Health check endpoint."""
    try:
        return jsonify({
            'ok': True,
            'data_age_minutes': get_data_age_minutes(),
            'collections_count': len(_available_collections),
            'engine_initialized': _engine is not None
        })
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(error))
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'

    logger.info("Starting CS2 Trade-ups Calculator on port %d", port)

    try:
        # Pre-initialize engine for faster first requests
        logger.info("Pre-initializing engine...")
        get_engine()
        logger.info("Engine pre-initialization complete")
    except Exception as e:
        logger.error("Failed to pre-initialize engine: %s", str(e))
        logger.info("Engine will be initialized on first request")

    app.run(host='0.0.0.0', port=port, debug=debug)
