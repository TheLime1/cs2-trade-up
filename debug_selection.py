#!/usr/bin/env python3

from analyze_tradeups import TradeUpAnalyzer

def debug_item_selection():
    """Debug what items are being selected for trade-ups"""
    print("=== DEBUGGING ITEM SELECTION ===")
    
    analyzer = TradeUpAnalyzer()
    print("Loading database...")
    analyzer.load_data()  # Load the data first
    
    # Get the top result
    results = analyzer.analyze(
        rarity='Mil-Spec',
        stattrak=False,
        top=1
    )
    
    if results:
        candidate = results[0]
        recommendations = candidate.buy_recommendations or []
        print(f"\nFound trade-up with {len(recommendations)} buy recommendations:")
        
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec.quantity}x {rec.market_name} - ${rec.price:.2f} each")
            
        # Check if any weapon appears in multiple wear levels
        weapon_wear_map = {}
        for rec in recommendations:
            weapon_name = rec.market_name.split(' | ')[0].replace('StatTrak™ ', '')
            
            # Extract wear level from name
            wear_match = None
            for wear in ['Factory New', 'Minimal Wear', 'Field-Tested', 'Well-Worn', 'Battle-Scarred']:
                if wear in rec.market_name:
                    wear_match = wear
                    break
            
            if weapon_name not in weapon_wear_map:
                weapon_wear_map[weapon_name] = set()
            if wear_match:
                weapon_wear_map[weapon_name].add(wear_match)
                
        print("\nWeapon wear level analysis:")
        for weapon, wears in weapon_wear_map.items():
            if len(wears) > 1:
                print(f"❌ {weapon}: Multiple wears - {', '.join(wears)}")
            else:
                print(f"✅ {weapon}: Single wear - {next(iter(wears)) if wears else 'Unknown'}")

if __name__ == "__main__":
    debug_item_selection()