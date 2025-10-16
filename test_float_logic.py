#!/usr/bin/env python3
"""Test script to investigate float and pricing issues."""

from analyze_tradeups import TradeUpAnalyzer

def main():
    print("=== INVESTIGATING FLOAT AND PRICING ISSUES ===\n")
    
    analyzer = TradeUpAnalyzer()
    analyzer.load_data()
    
    # Find some Nova items to test
    nova_items = []
    for skin in analyzer.skins:
        if skin.weapon == "Nova" and "Blaze Orange" in skin.market_name:
            nova_items.append(skin)
    
    print("=== NOVA BLAZE ORANGE ITEMS ===")
    for item in nova_items[:10]:  # Show first 10
        print(f"Name: {item.market_name}")
        print(f"  Exterior: {item.exterior}")
        print(f"  Float Range: {item.min_float} - {item.max_float}")
        print(f"  Price: ${item.steam_price}")
        print(f"  Available: {item.availability}")
        print()
    
    # Test the unified float logic
    print("=== TESTING UNIFIED FLOAT LOGIC ===")
    
    # Simulate what happens in the buy recommendation logic
    test_items = [
        ("Nova | Blaze Orange (Field-Tested)", 0.07, 0.15, 16.21),
        ("Nova | Blaze Orange (Well-Worn)", 0.38, 0.45, 15.41), 
        ("Nova | Blaze Orange (Battle-Scarred)", 0.45, 0.75, 15.50),
    ]
    
    print("Original float ranges:")
    for name, min_f, max_f, price in test_items:
        float_range = max_f - min_f
        recommended = min_f + (float_range * 0.25)
        print(f"{name}: {min_f:.3f}-{max_f:.3f} → recommended: {recommended:.3f}")
    
    # Apply unified logic
    print("\nAfter unified float logic:")
    recommended_floats = []
    for name, min_f, max_f, price in test_items:
        float_range = max_f - min_f
        recommended = min_f + (float_range * 0.25)
        recommended_floats.append(recommended)
    
    unified_float = max(recommended_floats)
    print(f"Unified float (maximum): {unified_float:.3f}")
    
    for name, min_f, max_f, price in test_items:
        print(f"{name}: float≤{unified_float:.3f}, ${price}")

if __name__ == "__main__":
    main()