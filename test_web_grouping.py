#!/usr/bin/env python3

import requests
import json


def test_web_grouping():
    """Test if web interface is getting grouped recommendations"""
    try:
        # Make request to web API
        response = requests.get(
            'http://localhost:5000/scan?rarity=Mil-Spec&stattrak=false&top=1')
        data = response.json()

        print("=== WEB API RESPONSE ===")
        if 'results' in data and len(data['results']) > 0:
            first_result = data['results'][0]
            buy_recs = first_result.get('buy_recommendations', [])

            print(f"Found {len(buy_recs)} buy recommendations:")
            for i, rec in enumerate(buy_recs):
                print(
                    f"{i+1}. {rec['market_name']} - Qty: {rec['quantity']} - Price: ${rec['price']:.2f}")

            # Check if we have multiple quantities (which would indicate grouping)
            quantities = [rec['quantity'] for rec in buy_recs]
            max_quantity = max(quantities) if quantities else 0

            print(f"\nMax quantity found: {max_quantity}")
            if max_quantity > 1:
                print("✅ Grouping is working!")
            else:
                print("❌ No grouping detected - all quantities are 1")

        else:
            print("No results found")

    except requests.exceptions.ConnectionError:
        print("❌ Server not running")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_web_grouping()
