#!/usr/bin/env python3
"""Debug script to analyze loaded skins."""

import sys
from analyze_tradeups import TradeUpAnalyzer
from rich.console import Console

console = Console()


def main():
    analyzer = TradeUpAnalyzer()
    analyzer.load_data()

    print(f"Total skins loaded: {len(analyzer.skins)}")

    # Count by rarity
    rarity_counts = {}
    collection_counts = {}
    stattrak_counts = {}

    for skin in analyzer.skins:
        rarity = skin.rarity or "Unknown"
        collection = skin.collection or "Unknown"

        rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
        collection_counts[collection] = collection_counts.get(
            collection, 0) + 1
        stattrak_counts[skin.stattrak] = stattrak_counts.get(
            skin.stattrak, 0) + 1

    print("\n=== RARITY DISTRIBUTION ===")
    for rarity, count in sorted(rarity_counts.items()):
        print(f"{rarity}: {count}")

    print("\n=== STATTRAK DISTRIBUTION ===")
    for stattrak, count in sorted(stattrak_counts.items()):
        print(f"StatTrak {stattrak}: {count}")

    print("\n=== TOP 10 COLLECTIONS ===")
    sorted_collections = sorted(
        collection_counts.items(), key=lambda x: x[1], reverse=True)
    for collection, count in sorted_collections[:10]:
        print(f"{collection}: {count}")

    # Check a few Mil-Spec examples
    print("\n=== FIRST 5 MIL-SPEC EXAMPLES ===")
    mil_spec_skins = [
        skin for skin in analyzer.skins if skin.rarity == "Mil-Spec Grade"][:5]
    for skin in mil_spec_skins:
        print(f"Name: {skin.market_name}")
        print(f"  Collection: {skin.collection}")
        print(f"  Price: ${skin.steam_price}")
        print(f"  Available: {skin.availability}")
        print(f"  StatTrak: {skin.stattrak}")
        print()


if __name__ == "__main__":
    main()
