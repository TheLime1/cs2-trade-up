#!/usr/bin/env python3

from analyze_tradeups import TradeUpAnalyzer


def debug_detailed_selection():
    """Debug the detailed item selection process"""
    print("=== DEBUGGING DETAILED SELECTION ===")

    # Create a custom analyzer with debug output
    analyzer = TradeUpAnalyzer()
    analyzer.load_data()

    # Let's manually check what items are available in The Militia Collection
    inputs_by_collection = analyzer.inputs_by_collection

    if "The Militia Collection" in inputs_by_collection:
        militia_items = inputs_by_collection["The Militia Collection"]
        print(f"Found {len(militia_items)} items in The Militia Collection:")

        # Group by weapon type
        weapons_by_type = {}
        for skin in militia_items:
            weapon_name = skin.market_name.split(
                ' | ')[0].replace('StatTrak™ ', '')
            if weapon_name not in weapons_by_type:
                weapons_by_type[weapon_name] = []
            weapons_by_type[weapon_name].append(skin)

        for weapon_name, weapon_items in weapons_by_type.items():
            print(f"\n{weapon_name}:")
            weapon_items.sort(key=lambda x: x.steam_price or 0.0)
            for item in weapon_items:
                print(f"  - {item.market_name}: ${item.steam_price:.2f}")

            # Show what the algorithm would select
            cheapest_variant = weapon_items[0]
            print(
                f"  → Algorithm would select: {cheapest_variant.market_name} (${cheapest_variant.steam_price:.2f})")


if __name__ == "__main__":
    debug_detailed_selection()
