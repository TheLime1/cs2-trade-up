#!/usr/bin/env python3
"""
Database Structure Analyzer
Analyzes the CS2 skin database to understand its structure and content.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path


def analyze_database_structure():
    """Analyze the database structure and content."""

    # Load the cached database
    cache_path = Path("./.cache/skins_database.json")

    if not cache_path.exists():
        print("❌ No cached database found. Please run the analyzer first to download the database.")
        return

    print(f"📁 Loading database from: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Raw data type: {type(data)}")
    print(
        f"📊 Raw data keys (if dict): {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

    # Handle different JSON structures
    if isinstance(data, list):
        records = data
        print(f"📊 Structure: Direct list with {len(records)} records")
    elif isinstance(data, dict):
        if 'items' in data:
            records = data['items']
            print(
                f"📊 Structure: Dict with 'items' key containing {len(records)} records")
        elif 'skins' in data:
            records = data['skins']
            print(
                f"📊 Structure: Dict with 'skins' key containing {len(records)} records")
        else:
            # Assume values are the records
            records = list(data.values())
            print(
                f"📊 Structure: Dict with values as records, {len(records)} records")
    else:
        print(f"❌ Unknown data structure: {type(data)}")
        return

    if not records:
        print("❌ No records found in database")
        return

    print(f"\n🔍 ANALYZING {len(records)} RECORDS...\n")

    # Analyze first few records
    print("📝 SAMPLE RECORDS:")
    print("=" * 80)
    for i, record in enumerate(records[:3]):
        print(f"\nRecord #{i+1}:")
        if isinstance(record, dict):
            for key, value in record.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {record}")

    if not isinstance(records[0], dict):
        print("❌ Records are not dictionaries, cannot analyze further")
        return

    # Analyze field distribution
    print("\n📊 FIELD ANALYSIS:")
    print("=" * 80)

    all_fields = set()
    field_counts = Counter()
    field_samples = defaultdict(set)

    for record in records:
        if isinstance(record, dict):
            for key, value in record.items():
                all_fields.add(key)
                if value is not None and value != "":
                    field_counts[key] += 1
                    if len(field_samples[key]) < 5:  # Keep sample values
                        # Truncate long values
                        field_samples[key].add(str(value)[:50])

    print(f"Total unique fields: {len(all_fields)}")
    print("\nField distribution:")
    for field in sorted(all_fields):
        count = field_counts[field]
        percentage = (count / len(records)) * 100
        samples = ", ".join(list(field_samples[field])[:3])
        print(
            f"  {field:30} {count:5d}/{len(records)} ({percentage:5.1f}%) - Sample: {samples}")

    # Analyze rarity distribution
    print("\n🎯 RARITY ANALYSIS:")
    print("=" * 80)

    rarity_fields = ['rarity', 'quality', 'tier', 'grade']
    rarity_values = Counter()

    for record in records:
        for field in rarity_fields:
            if field in record and record[field]:
                rarity_values[record[field]] += 1
                break

    if rarity_values:
        print("Rarity distribution:")
        for rarity, count in rarity_values.most_common():
            percentage = (count / len(records)) * 100
            print(f"  {rarity:30} {count:5d} ({percentage:5.1f}%)")
    else:
        print("❌ No rarity information found")

    # Analyze collections
    print("\n🏛️ COLLECTION ANALYSIS:")
    print("=" * 80)

    collection_fields = ['collection', 'case', 'container']
    collection_values = Counter()

    for record in records:
        for field in collection_fields:
            if field in record and record[field]:
                collection_values[record[field]] += 1
                break

    if collection_values:
        print(f"Found {len(collection_values)} unique collections:")
        for collection, count in collection_values.most_common(10):  # Top 10
            percentage = (count / len(records)) * 100
            print(f"  {collection:40} {count:5d} ({percentage:5.1f}%)")
        if len(collection_values) > 10:
            print(f"  ... and {len(collection_values) - 10} more collections")
    else:
        print("❌ No collection information found")

    # Analyze prices
    print("\n💰 PRICE ANALYSIS:")
    print("=" * 80)

    price_fields = ['steam_price', 'price',
                    'lowest_sell_order', 'steam', 'current_price']
    prices = []

    for record in records:
        for field in price_fields:
            if field in record and record[field] is not None:
                try:
                    price = float(record[field])
                    if price > 0:
                        prices.append(price)
                        break
                except (ValueError, TypeError):
                    continue

    if prices:
        prices.sort()
        print(
            f"Records with valid prices: {len(prices)}/{len(records)} ({len(prices)/len(records)*100:.1f}%)")
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"Median price: ${prices[len(prices)//2]:.2f}")
        print(f"Prices under $1: {len([p for p in prices if p < 1])}")
        print(f"Prices $1-$10: {len([p for p in prices if 1 <= p < 10])}")
        print(f"Prices $10+: {len([p for p in prices if p >= 10])}")
    else:
        print("❌ No valid price information found")

    # Analyze availability
    print("\n📦 AVAILABILITY ANALYSIS:")
    print("=" * 80)

    availability_fields = ['availability',
                           'available', 'quantity', 'stock', 'count']
    availabilities = []

    for record in records:
        for field in availability_fields:
            if field in record and record[field] is not None:
                try:
                    avail = int(record[field])
                    availabilities.append(avail)
                    break
                except (ValueError, TypeError):
                    continue

    if availabilities:
        print(
            f"Records with availability data: {len(availabilities)}/{len(records)} ({len(availabilities)/len(records)*100:.1f}%)")
        print(
            f"Available items (>0): {len([a for a in availabilities if a > 0])}")
        print(
            f"Out of stock (0): {len([a for a in availabilities if a == 0])}")
        print(f"Max availability: {max(availabilities)}")
    else:
        print("❌ No availability information found")

    # Analyze StatTrak
    print("\n⭐ STATTRAK ANALYSIS:")
    print("=" * 80)

    stattrak_count = 0
    for record in records:
        market_name = record.get('market_name', record.get('name', ''))
        if 'StatTrak' in market_name:
            stattrak_count += 1

    print(
        f"StatTrak items: {stattrak_count}/{len(records)} ({stattrak_count/len(records)*100:.1f}%)")

    # Look for specific examples
    print("\n🔍 SPECIFIC EXAMPLES:")
    print("=" * 80)

    print("\nMil-Spec examples:")
    mil_spec_count = 0
    for record in records:
        rarity = record.get('rarity', record.get('quality', ''))
        if rarity and 'mil-spec' in rarity.lower():
            mil_spec_count += 1
            if mil_spec_count <= 3:
                name = record.get('market_name', record.get('name', 'N/A'))
                collection = record.get('collection', 'N/A')
                price = record.get('steam_price', record.get('price', 'N/A'))
                print(f"  {name} | {collection} | ${price}")

    print(f"\nTotal Mil-Spec items found: {mil_spec_count}")

    # Check for float data
    print("\n🎲 FLOAT DATA ANALYSIS:")
    print("=" * 80)

    float_records = 0
    for record in records:
        if (record.get('min_float') is not None or
            record.get('max_float') is not None or
            record.get('float_min') is not None or
                record.get('float_max') is not None):
            float_records += 1

    print(
        f"Records with float data: {float_records}/{len(records)} ({float_records/len(records)*100:.1f}%)")


if __name__ == "__main__":
    analyze_database_structure()
