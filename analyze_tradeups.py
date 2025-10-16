#!/usr/bin/env python3
"""
CS2/CS:GO Trade-Up Analyzer
A comprehensive tool for analyzing profitable CS2/CS:GO trade-up contracts.

Usage:
    python analyze_tradeups.py --rarity "Mil-Spec" --stattrak false --min_roi 0.05 --max_cost 20.0 --top 50
    python analyze_tradeups.py --float_aware --input_floats 0.03,0.04,0.02,0.01,0.05,0.02,0.03,0.04,0.02,0.01
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from tabulate import tabulate


class SkinData(BaseModel):
    """Normalized skin data structure."""
    market_name: str
    weapon: Optional[str] = None
    skin: Optional[str] = None
    exterior: Optional[str] = None
    stattrak: bool = False
    souvenir: bool = False
    collection: Optional[str] = None
    rarity: Optional[str] = None
    min_float: Optional[float] = None
    max_float: Optional[float] = None
    steam_price: Optional[float] = None
    availability: Optional[int] = None  # Number of items available on market
    last_update: Optional[str] = None


@dataclass
class TradeUpOutcome:
    """Represents a specific trade-up outcome."""
    market_name: str
    collection: str
    probability: float
    price: float
    net_price: float
    expected_revenue_contribution: float  # probability × net_price


@dataclass
class BuyRecommendation:
    """Represents a specific item to buy for trade-up."""
    market_name: str
    collection: str
    price: float
    recommended_float: Optional[float] = None
    quantity: int = 1


@dataclass
class TradeUpCandidate:
    """Represents a candidate trade-up configuration."""
    inputs: Dict[str, int]  # collection -> count
    total_cost: float
    outcomes: List[TradeUpOutcome]
    expected_value: float
    roi: float
    rarity: str
    stattrak: bool
    buy_recommendations: Optional[List[BuyRecommendation]] = None
    avg_input_float: Optional[float] = None
    success_rate: Optional[float] = None  # Percentage of profitable outcomes


class SteamFeeCalculator:
    """Handles Steam Community Market fee calculations with precise Steam logic."""

    STEAM_FEE_RATE = 0.05  # 5%
    GAME_FEE_RATE = 0.10   # 10%
    MINIMUM_LISTING_PRICE = 0.03  # $0.03 minimum listing price
    MINIMUM_FEE_COMPONENT = 0.01  # $0.01 minimum per fee component

    @classmethod
    def exact_fee_split(cls, buyer_pays: float) -> float:
        """
        Calculate exact seller amount using Steam's precise fee logic.

        Args:
            buyer_pays: The price the buyer pays (listing price)

        Returns:
            The amount the seller receives after fees
        """
        # Enforce minimum listing price
        if buyer_pays < cls.MINIMUM_LISTING_PRICE:
            buyer_pays = cls.MINIMUM_LISTING_PRICE

        # Calculate individual fee components with minimums
        steam_fee = max(round(buyer_pays * cls.STEAM_FEE_RATE,
                        2), cls.MINIMUM_FEE_COMPONENT)
        game_fee = max(round(buyer_pays * cls.GAME_FEE_RATE, 2),
                       cls.MINIMUM_FEE_COMPONENT)

        # Total fees
        total_fee = steam_fee + game_fee

        # Seller receives the remainder
        seller_gets = round(buyer_pays - total_fee, 2)

        return seller_gets

    @classmethod
    def net_from_list_price(cls, list_price: float) -> float:
        """Calculate net amount seller receives after fees using exact Steam logic."""
        return cls.exact_fee_split(list_price)

    @classmethod
    def list_price_for_net(cls, net_amount: float) -> float:
        """Calculate list price needed to achieve target net amount."""
        # Start with simple calculation and iterate to find exact price
        # This accounts for the minimum fee components and rounding
        estimated_price = round(
            net_amount / (1 - cls.STEAM_FEE_RATE - cls.GAME_FEE_RATE), 2)

        # Ensure minimum listing price
        estimated_price = max(estimated_price, cls.MINIMUM_LISTING_PRICE)

        # Fine-tune to get exact net amount
        for adjustment in [0.00, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]:
            test_price = estimated_price + adjustment
            if test_price >= cls.MINIMUM_LISTING_PRICE:
                actual_net = cls.exact_fee_split(test_price)
                if actual_net >= net_amount:
                    return test_price

        return estimated_price

    @classmethod
    def get_total_fee_rate(cls, buyer_pays: float) -> float:
        """Get the effective total fee rate for a given price."""
        if buyer_pays < cls.MINIMUM_LISTING_PRICE:
            buyer_pays = cls.MINIMUM_LISTING_PRICE

        steam_fee = max(round(buyer_pays * cls.STEAM_FEE_RATE,
                        2), cls.MINIMUM_FEE_COMPONENT)
        game_fee = max(round(buyer_pays * cls.GAME_FEE_RATE, 2),
                       cls.MINIMUM_FEE_COMPONENT)
        total_fee = steam_fee + game_fee

        return total_fee / buyer_pays if buyer_pays > 0 else 0


class FloatCalculator:
    """Handles CS2 float value calculations and exterior mapping."""

    EXTERIOR_RANGES = {
        'Factory New': (0.00, 0.07),
        'Minimal Wear': (0.07, 0.15),
        'Field-Tested': (0.15, 0.38),
        'Well-Worn': (0.38, 0.45),
        'Battle-Scarred': (0.45, 1.00)
    }

    EXTERIOR_ABBREVIATIONS = {
        'Factory New': 'FN',
        'Minimal Wear': 'MW',
        'Field-Tested': 'FT',
        'Well-Worn': 'WW',
        'Battle-Scarred': 'BS'
    }

    @classmethod
    def calculate_output_float(cls, input_floats: List[float], min_out: float, max_out: float) -> float:
        """Calculate output float based on input float average."""
        avg_input = sum(input_floats) / len(input_floats)
        return min_out + (max_out - min_out) * avg_input

    @classmethod
    def float_to_exterior(cls, float_value: float, min_float: float = 0.0, max_float: float = 1.0) -> str:
        """Map float value to exterior condition."""
        # Clamp float to skin's actual range
        clamped_float = max(min_float, min(max_float, float_value))

        for exterior, (min_range, max_range) in cls.EXTERIOR_RANGES.items():
            if min_range <= clamped_float < max_range:
                return exterior

        # Handle edge case for exactly 1.0
        if clamped_float >= 0.45:
            return 'Battle-Scarred'

        return 'Factory New'  # Default fallback


class DatabaseLoader:
    """Handles loading and caching of the CS2 skin database."""

    PRIMARY_URL = "https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json"
    FALLBACK_URL = "https://raw.githubusercontent.com/TheLime1/cs2-price-database/main/data/skins_database.json"

    def __init__(self, cache_path: str = "./.cache/skins_database.json"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def load_database(self, force_refresh: bool = False) -> Dict:
        """Load database from cache or download if needed."""
        if not force_refresh and self._is_cache_valid():
            self.console.print("[green]Loading from cache...[/green]")
            return self._load_from_cache()

        self.console.print("[yellow]Downloading database...[/yellow]")
        data = self._download_database()
        self._save_to_cache(data)
        return data

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is recent enough (24 hours)."""
        if not self.cache_path.exists():
            return False

        cache_age = time.time() - self.cache_path.stat().st_mtime
        return cache_age < 24 * 3600  # 24 hours

    def _load_from_cache(self) -> Dict:
        """Load data from cache file."""
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_to_cache(self, data: Dict) -> None:
        """Save data to cache file."""
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _download_database(self) -> Dict:
        """Download database from remote URL."""
        urls = [self.PRIMARY_URL, self.FALLBACK_URL]

        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.console.print(
                    f"[red]Failed to download from {url}: {e}[/red]")
                continue

        raise Exception("Failed to download database from any URL")


class DataNormalizer:
    """Normalizes various database formats to unified schema."""

    @staticmethod
    def parse_market_name(market_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool, bool]:
        """Parse weapon, skin, exterior, stattrak, and souvenir from market name."""
        original_name = market_name
        stattrak = market_name.startswith("StatTrak™ ")
        souvenir = market_name.startswith("Souvenir ")

        # Remove prefixes
        if stattrak:
            market_name = market_name[10:]  # Remove "StatTrak™ "
        elif souvenir:
            market_name = market_name[9:]   # Remove "Souvenir "

        # Extract exterior from parentheses
        exterior = None
        if '(' in market_name and ')' in market_name:
            start = market_name.rfind('(')
            end = market_name.rfind(')')
            if start < end:
                exterior_text = market_name[start+1:end]
                # Map common exterior names
                exterior_map = {
                    'Factory New': 'Factory New',
                    'Minimal Wear': 'Minimal Wear',
                    'Field-Tested': 'Field-Tested',
                    'Well-Worn': 'Well-Worn',
                    'Battle-Scarred': 'Battle-Scarred'
                }
                exterior = exterior_map.get(exterior_text, exterior_text)
                market_name = market_name[:start].strip()

        # Split weapon and skin
        weapon = None
        skin = None
        if ' | ' in market_name:
            weapon, skin = market_name.split(' | ', 1)
        else:
            weapon = market_name

        return weapon, skin, exterior, stattrak, souvenir

    @staticmethod
    def normalize_record(record: Dict) -> List[SkinData]:
        """Normalize a database record to multiple SkinData entries (one per variant)."""
        results = []

        # Get base information
        base_weapon = record.get('weapon', '')
        base_skin = record.get('skin_name', '')
        collection = record.get('collection', '')
        raw_rarity = record.get('rarity', '')

        # Normalize rarity - handle database format differences
        rarity_aliases = {
            'Consumer': 'Consumer Grade',
            'Industrial': 'Industrial Grade',
            'Mil-Spec': 'Mil-Spec Grade',
            'Mil-spec': 'Mil-Spec Grade',  # Database format
            'Military': 'Mil-Spec Grade',
            'Mil-Spec Grade': 'Mil-Spec Grade',
            'Restricted': 'Restricted',
            'Classified': 'Classified',
            'Covert': 'Covert'
        }
        rarity = rarity_aliases.get(raw_rarity, raw_rarity)

        variants = record.get('variants', [])

        # Process each variant (exterior) separately
        for variant in variants:
            if not isinstance(variant, dict):
                continue

            exterior = variant.get('wear', '')
            float_range = variant.get('float_range', [])
            min_float = float_range[0] if len(float_range) >= 1 else None
            max_float = float_range[1] if len(float_range) >= 2 else None

            # Check if variant is available
            available = variant.get('available', False)
            if not available:
                continue

            prices = variant.get('prices', {})

            # Process normal variant
            normal_price_data = prices.get('normal', {})
            normal_price = normal_price_data.get('usd')
            has_normal_listings = variant.get('has_normal_listings', False)

            if normal_price and has_normal_listings:
                market_name = f"{base_weapon} | {base_skin} ({exterior})"

                results.append(SkinData(
                    market_name=market_name,
                    weapon=base_weapon,
                    skin=base_skin,
                    exterior=exterior,
                    stattrak=False,
                    souvenir=False,
                    collection=collection,
                    rarity=rarity,
                    min_float=min_float,
                    max_float=max_float,
                    steam_price=normal_price,
                    availability=1 if has_normal_listings else 0,  # Boolean to int
                    last_update=None
                ))

            # Process StatTrak variant
            stattrak_price_data = prices.get('stattrak', {})
            stattrak_price = stattrak_price_data.get('usd')
            has_stattrak_listings = variant.get('has_stattrak_listings', False)
            stattrak_available = variant.get('stattrak_available', False)

            if stattrak_price and has_stattrak_listings and stattrak_available:
                stattrak_market_name = f"StatTrak™ {base_weapon} | {base_skin} ({exterior})"

                results.append(SkinData(
                    market_name=stattrak_market_name,
                    weapon=base_weapon,
                    skin=base_skin,
                    exterior=exterior,
                    stattrak=True,
                    souvenir=False,
                    collection=collection,
                    rarity=rarity,
                    min_float=min_float,
                    max_float=max_float,
                    steam_price=stattrak_price,
                    availability=1 if has_stattrak_listings else 0,  # Boolean to int
                    last_update=None
                ))

        return results


class CollectionIndex:
    """Builds and maintains collection mappings for trade-up analysis."""

    RARITY_LADDER = [
        'Consumer Grade',
        'Industrial Grade',
        'Mil-Spec Grade',
        'Restricted',
        'Classified',
        'Covert'
    ]

    # Alternative names for rarities
    RARITY_ALIASES = {
        'Consumer': 'Consumer Grade',
        'Industrial': 'Industrial Grade',
        'Mil-Spec': 'Mil-Spec Grade',
        'Mil-spec': 'Mil-Spec Grade',  # Database format
        'Military': 'Mil-Spec Grade',
        'Mil-Spec Grade': 'Mil-Spec Grade',
        'Restricted': 'Restricted',
        'Classified': 'Classified',
        'Covert': 'Covert'
    }

    def __init__(self, skins: List[SkinData], allow_consumer_inputs: bool = True):
        self.skins = skins
        self.allow_consumer_inputs = allow_consumer_inputs
        self.console = Console()

        # Build indexes
        # (collection, rarity, stattrak) -> List[SkinData]
        self.collection_outputs = {}
        self.eligible_inputs = {}     # (rarity, stattrak) -> List[SkinData]
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build collection and eligibility indexes."""
        eligible_skins = []

        for skin in self.skins:
            # Skip ineligible types
            if self._is_ineligible(skin):
                continue

            eligible_skins.append(skin)

            # Index by collection and rarity for outputs
            if skin.collection and skin.rarity:
                key = (skin.collection, skin.rarity, skin.stattrak)
                if key not in self.collection_outputs:
                    self.collection_outputs[key] = []
                self.collection_outputs[key].append(skin)

        # Build eligible inputs by rarity
        for skin in eligible_skins:
            if self._is_valid_input(skin):
                key = (skin.rarity, skin.stattrak)
                if key not in self.eligible_inputs:
                    self.eligible_inputs[key] = []
                self.eligible_inputs[key].append(skin)

        self.console.print(
            f"[green]Indexed {len(eligible_skins)} eligible skins[/green]")
        self.console.print(
            f"[green]Found {len(self.eligible_inputs)} input categories[/green]")

    def _is_ineligible(self, skin: SkinData) -> bool:
        """Check if skin is ineligible for any trade-up use."""
        # Skip items without required data
        if not skin.collection or not skin.rarity or skin.steam_price is None:
            return True

        # Skip items with explicitly zero availability (but allow None/missing availability)
        if skin.availability is not None and skin.availability <= 0:
            return True

        # Skip souvenir items
        if skin.souvenir:
            return True

        # Skip knives (simple heuristic)
        if skin.weapon and any(knife in skin.weapon.lower() for knife in ['knife', 'bayonet', 'karambit']):
            return True

        # Skip contraband
        if skin.rarity and 'contraband' in skin.rarity.lower():
            return True

        return False

    def _is_valid_input(self, skin: SkinData) -> bool:
        """Check if skin can be used as trade-up input."""
        if not skin.rarity:
            return False

        # Normalize rarity name
        rarity = self.RARITY_ALIASES.get(skin.rarity, skin.rarity)

        # Consumer Grade inputs only if explicitly allowed
        if rarity == 'Consumer Grade' and not self.allow_consumer_inputs:
            return False

        # Covert cannot be used as input (only as output)
        if rarity == 'Covert':
            return False

        return True

    def get_next_rarity(self, current_rarity: str) -> Optional[str]:
        """Get the next rarity tier for trade-up output."""
        normalized = self.RARITY_ALIASES.get(current_rarity, current_rarity)

        try:
            current_index = self.RARITY_LADDER.index(normalized)
            if current_index + 1 < len(self.RARITY_LADDER):
                return self.RARITY_LADDER[current_index + 1]
        except ValueError:
            pass

        return None

    def get_inputs_by_rarity(self, rarity: str, stattrak: bool) -> List[SkinData]:
        """Get all eligible inputs for a given rarity and StatTrak status."""
        normalized = self.RARITY_ALIASES.get(rarity, rarity)
        key = (normalized, stattrak)
        return self.eligible_inputs.get(key, [])

    def get_outputs_by_collection(self, collection: str, rarity: str, stattrak: bool) -> List[SkinData]:
        """Get all possible outputs for a collection at a given rarity."""
        normalized = self.RARITY_ALIASES.get(rarity, rarity)
        key = (collection, normalized, stattrak)
        outputs = self.collection_outputs.get(key, [])

        # For StatTrak trade-ups, ensure at least one StatTrak variant exists
        # If no StatTrak outputs exist for this collection+rarity, return empty list
        if stattrak and not outputs:
            return []

        return outputs


class TradeUpCalculator:
    """Calculates trade-up probabilities and expected values."""

    def __init__(self, collection_index: CollectionIndex, assume_input_costs_include_fees: bool = True,
                 buy_slippage_pct: float = 0.0, sell_slippage_pct: float = 0.0,
                 custom_fee_rate: Optional[float] = None, min_liquidity: int = 1):
        self.collection_index = collection_index
        self.assume_input_costs_include_fees = assume_input_costs_include_fees
        # Additional cost for buying (market variance)
        self.buy_slippage_pct = buy_slippage_pct
        # Reduced revenue for selling (market variance)
        self.sell_slippage_pct = sell_slippage_pct
        self.custom_fee_rate = custom_fee_rate  # Override default 15% fee rate
        self.min_liquidity = min_liquidity  # Minimum availability for outputs
        self.console = Console()

    def apply_market_adjustments(self, price: float, is_buying: bool) -> float:
        """Apply slippage and custom fee rates to prices."""
        if is_buying:
            # Apply buy slippage (increases cost)
            adjusted_price = price * (1 + self.buy_slippage_pct / 100)
        else:
            # For selling, first apply custom fee rate if specified
            if self.custom_fee_rate is not None:
                # Use custom fee rate instead of Steam's default
                net_price = price * (1 - self.custom_fee_rate / 100)
            else:
                # Use Steam's precise fee calculation
                net_price = SteamFeeCalculator.exact_fee_split(price)

            # Then apply sell slippage (reduces revenue)
            adjusted_price = net_price * (1 - self.sell_slippage_pct / 100)

        return adjusted_price

    def generate_candidates(self, rarity: str, stattrak: bool) -> List[TradeUpCandidate]:
        """Generate all viable trade-up candidates using database prices and availability."""
        inputs = self.collection_index.get_inputs_by_rarity(rarity, stattrak)
        if not inputs:
            self.console.print(
                f"[red]No eligible inputs found for {rarity} (StatTrak: {stattrak})[/red]")
            return []

        next_rarity = self.collection_index.get_next_rarity(rarity)
        if not next_rarity:
            self.console.print(f"[red]No next rarity found for {rarity}[/red]")
            return []

        # Group inputs by collection and sort by availability + price
        inputs_by_collection = {}
        for skin in inputs:
            if skin.collection not in inputs_by_collection:
                inputs_by_collection[skin.collection] = []
            inputs_by_collection[skin.collection].append(skin)

        # Sort each collection by availability (available first) then by price (cheapest first)
        # This deprioritizes items with missing/low availability but doesn't exclude them
        for collection in inputs_by_collection:
            def sort_key(skin):
                # Primary sort: availability (higher is better, None treated as 0)
                availability = skin.availability if skin.availability is not None else 0
                # Secondary sort: price (lower is better)
                price = skin.steam_price or float('inf')
                return (-availability, price)

            inputs_by_collection[collection].sort(key=sort_key)

        candidates = []
        collections = list(inputs_by_collection.keys())

        # Generate composition patterns: (10-0), (8-2), (7-3), (6-4), (5-5)
        patterns = [
            [(10, 0)],
            [(8, 2)],
            [(7, 3)],
            [(6, 4)],
            [(5, 5)]
        ]

        for pattern in patterns:
            if len(collections) >= len(pattern[0]):
                for collection_combo in combinations_with_replacement(collections, len(pattern[0])):
                    for counts in pattern:
                        if len(counts) == len(collection_combo):
                            candidate = self._evaluate_composition(
                                dict(zip(collection_combo, counts)),
                                inputs_by_collection,
                                next_rarity,
                                rarity,
                                stattrak
                            )
                            if candidate:
                                candidates.append(candidate)

        return candidates

    def _evaluate_composition(self, composition: Dict[str, int], inputs_by_collection: Dict[str, List[SkinData]],
                              next_rarity: str, input_rarity: str, stattrak: bool) -> Optional[TradeUpCandidate]:
        """Evaluate a specific composition of inputs using database data."""
        # Calculate total input cost and generate buy recommendations
        total_cost = 0.0
        selected_inputs = []
        buy_recommendations = []

        for collection, count in composition.items():
            if collection not in inputs_by_collection:
                return None

            available = inputs_by_collection[collection]
            if len(available) < count:
                return None

            # Enforce same weapon = same wear level for consistency
            collection_items = []
            selected_weapons = {}  # Track which specific variant we chose for each weapon

            for i in range(count):
                skin = available[i % len(available)]

                # Extract weapon name
                weapon_name = skin.market_name.split(
                    ' | ')[0].replace('StatTrak™ ', '')

                # If we've seen this weapon before, use the same variant
                if weapon_name in selected_weapons:
                    skin = selected_weapons[weapon_name]
                else:
                    # First time seeing this weapon, record our choice
                    selected_weapons[weapon_name] = skin

                cost = skin.steam_price or 0.0
                if not self.assume_input_costs_include_fees:
                    cost = SteamFeeCalculator.list_price_for_net(cost)

                # Apply buy slippage
                cost = self.apply_market_adjustments(cost, is_buying=True)

                total_cost += cost
                selected_inputs.append(skin)
                collection_items.append(skin)

            # Group identical items and create buy recommendations
            item_counts = {}
            for item in collection_items:
                key = item.market_name
                if key not in item_counts:
                    item_counts[key] = {'item': item, 'count': 0}
                item_counts[key]['count'] += 1

            for item_data in item_counts.values():
                item = item_data['item']
                quantity = item_data['count']

                # Calculate optimal recommended float for this item
                recommended_float = None
                if item.min_float is not None and item.max_float is not None:
                    # Recommend a float that's 25% into the range (slightly better than mid-range)
                    # This balances cost vs output quality
                    float_range = item.max_float - item.min_float
                    recommended_float = item.min_float + (float_range * 0.25)

                buy_recommendations.append(BuyRecommendation(
                    market_name=item.market_name,
                    collection=collection,
                    price=item.steam_price or 0.0,
                    recommended_float=recommended_float,
                    quantity=quantity
                ))

        # Apply unified float constraints for same weapons
        weapon_float_map = {}

        # First pass: determine the LEAST restrictive (highest) float for each weapon
        # This ensures all items of the same weapon are actually obtainable
        for rec in buy_recommendations:
            # Extract weapon name from market name (e.g., "Nova | Blaze Orange (Field-Tested)" -> "Nova")
            weapon_name = rec.market_name.split(
                ' | ')[0].replace('StatTrak™ ', '')

            if weapon_name not in weapon_float_map:
                weapon_float_map[weapon_name] = rec.recommended_float
            elif rec.recommended_float is not None and weapon_float_map[weapon_name] is not None:
                # Use the less restrictive (higher) float for all items of this weapon
                weapon_float_map[weapon_name] = max(
                    weapon_float_map[weapon_name], rec.recommended_float)

        # Second pass: apply unified float constraints
        for rec in buy_recommendations:
            weapon_name = rec.market_name.split(
                ' | ')[0].replace('StatTrak™ ', '')
            if weapon_name in weapon_float_map:
                rec.recommended_float = weapon_float_map[weapon_name]

        if len(selected_inputs) != 10:
            return None

        # Calculate average input float for display using recommended floats
        avg_input_float = None
        recommended_floats = []
        for rec in buy_recommendations:
            if rec.recommended_float is not None:
                recommended_floats.extend(
                    [rec.recommended_float] * rec.quantity)
        if len(recommended_floats) == 10:
            avg_input_float = sum(recommended_floats) / len(recommended_floats)

        # Calculate outcomes
        outcomes = []
        collections_with_outputs = set()

        for collection, count in composition.items():
            outputs = self.collection_index.get_outputs_by_collection(
                collection, next_rarity, stattrak)
            if not outputs:
                continue

            collections_with_outputs.add(collection)

            # Calculate probability for each specific output
            collection_prob = count / 10.0
            num_outputs = len(outputs)

            for output in outputs:
                if output.steam_price is None:
                    continue

                # Apply liquidity guard - skip outputs with very low availability
                if (output.availability is not None and
                        output.availability < self.min_liquidity):
                    continue

                # Handle float-aware calculations for better price estimation
                net_price = output.steam_price
                if avg_input_float is not None and output.min_float is not None and output.max_float is not None:
                    # Calculate expected output float and map to correct exterior
                    output_float = FloatCalculator.calculate_output_float(
                        [avg_input_float] * 10, output.min_float, output.max_float
                    )
                    target_exterior = FloatCalculator.float_to_exterior(
                        output_float, output.min_float, output.max_float
                    )

                    # Try to find the output with the correct exterior for better price accuracy
                    for potential_output in outputs:
                        if potential_output.exterior == target_exterior and potential_output.steam_price:
                            net_price = potential_output.steam_price
                            break

                # Apply sell slippage and fee calculations
                net_price = self.apply_market_adjustments(
                    net_price, is_buying=False)

                probability = collection_prob / num_outputs
                expected_revenue_contribution = probability * net_price

                outcomes.append(TradeUpOutcome(
                    market_name=output.market_name,
                    collection=collection,
                    probability=probability,
                    price=output.steam_price,
                    net_price=net_price,
                    expected_revenue_contribution=expected_revenue_contribution
                ))

        if not outcomes:
            return None

        # Calculate expected value and ROI
        expected_value = sum(
            outcome.expected_revenue_contribution for outcome in outcomes) - total_cost
        roi = expected_value / total_cost if total_cost > 0 else 0.0

        # Calculate success rate (percentage of outcomes where output covers total input cost)
        # A trade-up is "successful" if the single output covers the total input basket cost
        profitable_outcomes = [
            o for o in outcomes if o.net_price >= total_cost]
        success_rate = sum(
            o.probability for o in profitable_outcomes) if profitable_outcomes else 0.0

        return TradeUpCandidate(
            inputs=composition,
            total_cost=total_cost,
            outcomes=outcomes,
            expected_value=expected_value,
            roi=roi,
            rarity=input_rarity,
            stattrak=stattrak,
            buy_recommendations=buy_recommendations,
            avg_input_float=avg_input_float,
            success_rate=success_rate
        )


class TradeUpAnalyzer:
    """Main analyzer class that orchestrates the trade-up analysis."""

    def __init__(self, cache_path: str = "./.cache/skins_database.json"):
        self.cache_path = cache_path
        self.console = Console()
        self.database_loader = DatabaseLoader(cache_path)
        self.skins = []
        self.collection_index = None

    def load_data(self, force_refresh: bool = False, allow_consumer_inputs: bool = True) -> None:
        """Load and normalize skin data."""
        self.console.print(
            "[bold blue]Loading CS2 skin database...[/bold blue]")

        raw_data = self.database_loader.load_database(force_refresh)

        # Handle different JSON structures
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict) and 'items' in raw_data:
            records = raw_data['items']
        elif isinstance(raw_data, dict) and 'skins' in raw_data:
            records = raw_data['skins']
        else:
            # Assume it's a dict where values are the records
            records = list(raw_data.values())

        self.console.print(
            f"[yellow]Processing {len(records)} records...[/yellow]")

        # Normalize all records
        self.skins = []
        for record in records:
            try:
                # normalize_record now returns a list of SkinData
                skin_variants = DataNormalizer.normalize_record(record)
                self.skins.extend(skin_variants)
            except Exception:
                # Skip problematic records
                continue

        self.console.print(f"[green]Loaded {len(self.skins)} skins[/green]")

        # Build collection index
        self.collection_index = CollectionIndex(
            self.skins, allow_consumer_inputs)

    def analyze(self, rarity: str, stattrak: bool = False, min_roi: float = 0.0,
                top: int = 25, assume_input_costs_include_fees: bool = True, max_cost: float = 0.0,
                buy_slippage_pct: float = 0.0, sell_slippage_pct: float = 0.0,
                custom_fee_rate: Optional[float] = None, min_liquidity: int = 1) -> List[TradeUpCandidate]:
        """Perform trade-up analysis using database prices and availability."""
        if not self.collection_index:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.console.print(
            f"[bold blue]Analyzing {rarity} trade-ups (StatTrak: {stattrak})[/bold blue]")

        calculator = TradeUpCalculator(
            self.collection_index, assume_input_costs_include_fees,
            buy_slippage_pct, sell_slippage_pct, custom_fee_rate, min_liquidity)
        candidates = calculator.generate_candidates(rarity, stattrak)

        # Filter by minimum ROI and maximum cost
        profitable = [c for c in candidates if c.roi >= min_roi]
        if max_cost > 0:
            profitable = [c for c in profitable if c.total_cost <= max_cost]

        # Sort by expected value (descending)
        profitable.sort(key=lambda x: x.expected_value, reverse=True)

        self.console.print(
            f"[green]Found {len(profitable)} profitable trade-ups[/green]")

        return profitable[:top]

    def print_results(self, candidates: List[TradeUpCandidate]) -> None:
        """Print analysis results in a formatted table."""
        if not candidates:
            self.console.print(
                "[yellow]No profitable trade-ups found[/yellow]")
            return

        for i, candidate in enumerate(candidates, 1):
            self.console.print(f"\n[bold cyan]Trade-up #{i}[/bold cyan]")

            # Collections summary
            collections_summary = ", ".join(
                [f"{coll} (x{count})" for coll, count in candidate.inputs.items()])
            self.console.print(
                f"[bold]Collections:[/bold] {collections_summary}")

            # Key metrics in a grid
            self.console.print(
                f"\n[blue]Rarity:[/blue] {candidate.rarity} ({'StatTrak™' if candidate.stattrak else 'Normal'})")
            if candidate.avg_input_float:
                self.console.print(
                    f"[blue]Avg Input Float:[/blue] {candidate.avg_input_float:.4f}")
            self.console.print(
                f"[yellow]Input Cost:[/yellow] ${candidate.total_cost:.2f}")
            self.console.print(
                f"[orange3]Expected Value:[/orange3] ${candidate.expected_value + candidate.total_cost:.2f}")
            self.console.print(
                f"[green]Profit:[/green] ${candidate.expected_value:.2f}")
            self.console.print(
                f"[cyan]Profitability:[/cyan] {candidate.roi:.2%}")
            if candidate.success_rate:
                self.console.print(
                    f"[cyan]Success Rate:[/cyan] {candidate.success_rate:.1%}")

            # Buy recommendations - what the user should purchase
            if candidate.buy_recommendations:
                self.console.print(
                    f"\n[bold yellow]Items to Buy:[/bold yellow]")

                for rec in candidate.buy_recommendations:
                    float_info = ""
                    if rec.recommended_float is not None:
                        float_info = f", float≤{rec.recommended_float:.2f}"

                    quantity_text = f"{rec.quantity} x " if rec.quantity > 1 else ""

                    # Create a styled display similar to your image
                    price_per_item = rec.price
                    total_price = price_per_item * rec.quantity

                    item_display = f"[cyan]{quantity_text}[/cyan][bold]{rec.market_name}[/bold] [dim]${price_per_item:.2f}{float_info}[/dim]"
                    if rec.quantity > 1:
                        item_display += f" [dim](Total: ${total_price:.2f})[/dim]"

                    self.console.print(f"  • {item_display}")

            # Possible outcomes table
            self.console.print(
                f"\n[bold yellow]Possible Outcomes:[/bold yellow]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Probability", justify="right", style="cyan")
            table.add_column("Item", style="white")
            table.add_column("Collection", style="green")
            table.add_column("Price", justify="right", style="yellow")
            table.add_column("Net", justify="right", style="green")
            table.add_column("Contribution", justify="right", style="blue")

            # Sort outcomes by probability (highest first)
            sorted_outcomes = sorted(
                candidate.outcomes, key=lambda x: x.probability, reverse=True)

            for outcome in sorted_outcomes:
                table.add_row(
                    f"{outcome.probability:.1%}",
                    outcome.market_name,
                    outcome.collection,
                    f"${outcome.price:.2f}",
                    f"${outcome.net_price:.2f}",
                    f"${outcome.expected_revenue_contribution:.2f}"
                )

            self.console.print(table)

            # Summary line
            profit_color = "green" if candidate.expected_value > 0 else "red"
            self.console.print(
                f"\n[bold {profit_color}]Summary: Invest ${candidate.total_cost:.2f} → Expected ${candidate.expected_value + candidate.total_cost:.2f} (Profit: ${candidate.expected_value:.2f}, ROI: {candidate.roi:.1%})[/bold {profit_color}]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CS2/CS:GO Trade-Up Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_tradeups.py --rarity "Mil-Spec" --stattrak false --min_roi 0.05 --top 50
  python analyze_tradeups.py --rarity "Restricted" --min_roi 0.10 --top 25
  python analyze_tradeups.py --assume_input_costs_include_fees true
        """
    )

    parser.add_argument(
        '--rarity',
        required=True,
        choices=['Industrial', 'Mil-Spec', 'Restricted', 'Classified'],
        help='Input rarity tier for trade-up'
    )

    parser.add_argument(
        '--stattrak',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Whether to use StatTrak items (true/false)'
    )

    parser.add_argument(
        '--min_roi',
        type=float,
        default=0.0,
        help='Minimum ROI threshold (default: 0.0)'
    )

    parser.add_argument(
        '--max_cost',
        type=float,
        default=0.0,
        help='Maximum total input cost filter (default: 0.0 = no limit)'
    )

    parser.add_argument(
        '--top',
        type=int,
        default=25,
        help='Number of top results to show (default: 25)'
    )

    parser.add_argument(
        '--assume_input_costs_include_fees',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Whether input costs include Steam fees (true/false)'
    )

    parser.add_argument(
        '--allow_consumer_inputs',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Allow Consumer Grade items as inputs (true/false) - default: true'
    )

    parser.add_argument(
        '--buy_slippage_pct',
        type=float,
        default=0.0,
        help='Buy slippage percentage to simulate market variance (default: 0.0)'
    )

    parser.add_argument(
        '--sell_slippage_pct',
        type=float,
        default=0.0,
        help='Sell slippage percentage to simulate market variance (default: 0.0)'
    )

    parser.add_argument(
        '--custom_fee_rate',
        type=float,
        default=None,
        help='Custom fee rate percentage (overrides Steam\'s 15%, e.g. 10.0 for 10%)'
    )

    parser.add_argument(
        '--min_liquidity',
        type=int,
        default=1,
        help='Minimum availability/listings required for outputs (default: 1)'
    )

    parser.add_argument(
        '--cache_path',
        default='./.cache/skins_database.json',
        help='Path to database cache file'
    )

    parser.add_argument(
        '--force_refresh',
        action='store_true',
        help='Force refresh of cached database'
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = TradeUpAnalyzer(args.cache_path)

        # Load data
        analyzer.load_data(args.force_refresh, args.allow_consumer_inputs)

        # Perform analysis
        results = analyzer.analyze(
            rarity=args.rarity,
            stattrak=args.stattrak,
            min_roi=args.min_roi,
            max_cost=args.max_cost,
            top=args.top,
            assume_input_costs_include_fees=args.assume_input_costs_include_fees,
            buy_slippage_pct=args.buy_slippage_pct,
            sell_slippage_pct=args.sell_slippage_pct,
            custom_fee_rate=args.custom_fee_rate,
            min_liquidity=args.min_liquidity
        )

        # Print results
        analyzer.print_results(results)

    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
