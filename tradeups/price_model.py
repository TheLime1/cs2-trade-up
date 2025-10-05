"""
Price model module for CS2 trade-ups calculator.

Provides normalized item structure, price selection, and StatTrak detection.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NormalizedItem:
    """Normalized representation of a CS2 skin item."""

    id: str
    market_hash_name: str
    weapon: Optional[str]
    skin: Optional[str]
    stattrak: bool
    collection: Optional[str]
    rarity: str
    rarity_rank: int
    price: float
    wear: Optional[str] = None
    paint_index: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and clean up data after initialization."""
        # Ensure price is non-negative
        if self.price < 0:
            self.price = 0.0

        # Generate ID if not provided
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this item."""
        base = self.market_hash_name.lower()
        # Remove special characters and spaces
        clean_base = re.sub(r'[^\w\s-]', '', base)
        clean_base = re.sub(r'\s+', '_', clean_base)
        return clean_base

    @property
    def is_tradeable(self) -> bool:
        """Check if item is tradeable (has valid price and collection)."""
        return self.price > 0 and self.collection is not None

    @property
    def display_name(self) -> str:
        """Get display-friendly name."""
        return self.market_hash_name


class RarityMapper:
    """Maps rarity names to standardized ranks."""

    # Standard rarity order (lower rank = lower rarity)
    RARITY_ORDER = [
        "Consumer Grade",
        "Industrial Grade",
        "Mil-Spec Grade",
        "Restricted",
        "Classified",
        "Covert"
    ]

    # Aliases for different naming conventions
    RARITY_ALIASES = {
        # Color-based names
        "white": "Consumer Grade",
        "light_blue": "Industrial Grade",
        "blue": "Mil-Spec Grade",
        "purple": "Restricted",
        "pink": "Classified",
        "red": "Covert",

        # Short names
        "consumer": "Consumer Grade",
        "industrial": "Industrial Grade",
        "milspec": "Mil-Spec Grade",
        "mil_spec": "Mil-Spec Grade",
        "restricted": "Restricted",
        "classified": "Classified",
        "covert": "Covert",

        # Alternative formats
        "grade_consumer": "Consumer Grade",
        "grade_industrial": "Industrial Grade",
        "grade_milspec": "Mil-Spec Grade",
        "grade_restricted": "Restricted",
        "grade_classified": "Classified",
        "grade_covert": "Covert"
    }

    @classmethod
    def get_rarity_rank(cls, rarity: str) -> int:
        """
        Get the numeric rank for a rarity name.

        Args:
            rarity: Rarity name or alias

        Returns:
            Numeric rank (0 = lowest rarity)
        """
        if not rarity:
            return 0

        # Normalize the rarity name
        normalized = cls._normalize_rarity_name(rarity)

        # Check aliases first
        if normalized in cls.RARITY_ALIASES:
            # Get the aliased name and normalize it too
            aliased_name = cls.RARITY_ALIASES[normalized]
            normalized = cls._normalize_rarity_name(aliased_name)

        # Find in order list
        for i, standard_name in enumerate(cls.RARITY_ORDER):
            if normalized == cls._normalize_rarity_name(standard_name):
                return i

        # If not found, try partial matching
        for i, standard_name in enumerate(cls.RARITY_ORDER):
            if normalized in standard_name.lower() or standard_name.lower() in normalized:
                logger.warning(
                    f"Used partial match for rarity '{rarity}' -> '{standard_name}'")
                return i

        # Default to consumer grade if unknown
        logger.warning(
            f"Unknown rarity '{rarity}', defaulting to Consumer Grade")
        return 0

    @classmethod
    def get_standard_rarity_name(cls, rarity: str) -> str:
        """Get the standard rarity name for any input."""
        rank = cls.get_rarity_rank(rarity)
        return cls.RARITY_ORDER[min(rank, len(cls.RARITY_ORDER) - 1)]

    @classmethod
    def _normalize_rarity_name(cls, rarity: str) -> str:
        """Normalize rarity name for comparison."""
        return rarity.lower().strip().replace(' ', '_').replace('-', '_')


class PriceSelector:
    """Selects the best price from various price sources."""

    # Price field preferences (in order)
    PRICE_PREFERENCES = [
        'steam.lowest_sell',
        'lowest_sell',
        'steam_price',
        'market_price',
        'price',
        'avg_price',
        'average_price',
        'median_price',
        'highest_buy'
    ]

    @classmethod
    def select_price(cls, item_data: Dict[str, Any]) -> float:
        """
        Select the best available price from item data.

        Args:
            item_data: Raw item data containing price information

        Returns:
            Selected price as float, or 0.0 if no valid price found
        """
        # Try direct price fields first
        for field in cls.PRICE_PREFERENCES:
            price = cls._extract_price_field(item_data, field)
            if price is not None and price > 0:
                return price

        # Try nested price objects
        if 'prices' in item_data:
            return cls._extract_from_prices_object(item_data['prices'])

        if 'price_data' in item_data:
            return cls._extract_from_prices_object(item_data['price_data'])

        # Try any field containing 'price'
        for key, value in item_data.items():
            if 'price' in key.lower() and isinstance(value, (int, float)):
                price = float(value)
                if price > 0:
                    return price

        logger.warning(
            f"No valid price found for item: {item_data.get('name', 'Unknown')}")
        return 0.0

    @classmethod
    def _extract_price_field(cls, data: Dict[str, Any], field_path: str) -> Optional[float]:
        """Extract price from a potentially nested field path."""
        try:
            value = data
            for part in field_path.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None

            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Try to parse string as number
                return float(value.replace('$', '').replace(',', ''))

        except (ValueError, TypeError, AttributeError):
            pass

        return None

    @classmethod
    def _extract_from_prices_object(cls, prices_obj: Any) -> float:
        """Extract price from a complex prices object."""
        if isinstance(prices_obj, (int, float)):
            return float(prices_obj)

        if isinstance(prices_obj, dict):
            for field in cls.PRICE_PREFERENCES:
                price = cls._extract_price_field(prices_obj, field)
                if price is not None and price > 0:
                    return price

        return 0.0


class StatTrakDetector:
    """Detects StatTrak items from various indicators."""

    STATTRAK_INDICATORS = [
        'stattrak™',
        'stattrak',
        'stat-trak',
        'stat_trak',
        'st '
    ]

    @classmethod
    def is_stattrak(cls, item_data: Dict[str, Any]) -> bool:
        """
        Determine if an item is StatTrak.

        Args:
            item_data: Item data dictionary

        Returns:
            True if item is StatTrak
        """
        # Check explicit boolean field
        stattrak_field = item_data.get('stattrak')
        if isinstance(stattrak_field, bool):
            return stattrak_field

        # Check name-based detection
        name = item_data.get('name', item_data.get('market_hash_name', ''))
        if name:
            return cls._detect_from_name(name)

        # Check other possible fields
        for field in ['is_stattrak', 'stat_trak', 'st']:
            if field in item_data:
                value = item_data[field]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes']
                elif isinstance(value, (int, float)):
                    return bool(value)

        return False

    @classmethod
    def _detect_from_name(cls, name: str) -> bool:
        """Detect StatTrak from item name."""
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in cls.STATTRAK_INDICATORS)


def normalize_item(raw_data: Dict[str, Any]) -> NormalizedItem:
    """
    Convert raw item data to NormalizedItem.

    Args:
        raw_data: Raw item data from JSON

    Returns:
        NormalizedItem instance
    """
    # Extract basic fields
    name = raw_data.get('name', raw_data.get('market_hash_name', ''))
    weapon = raw_data.get('weapon')
    skin = raw_data.get('skin')
    collection = raw_data.get('collection')
    rarity_raw = raw_data.get('rarity', 'Consumer')
    wear = raw_data.get('wear')
    paint_index = raw_data.get('paint_index')

    # Process derived fields
    stattrak = StatTrakDetector.is_stattrak(raw_data)
    price = PriceSelector.select_price(raw_data)
    rarity = RarityMapper.get_standard_rarity_name(rarity_raw)
    rarity_rank = RarityMapper.get_rarity_rank(rarity_raw)

    # Generate ID
    item_id = raw_data.get('id', '')
    if not item_id and name:
        # Generate from name
        item_id = re.sub(r'[^\w\s-]', '', name.lower())
        item_id = re.sub(r'\s+', '_', item_id)

    return NormalizedItem(
        id=item_id,
        market_hash_name=name,
        weapon=weapon,
        skin=skin,
        stattrak=stattrak,
        collection=collection,
        rarity=rarity,
        rarity_rank=rarity_rank,
        price=price,
        wear=wear,
        paint_index=paint_index,
        raw_data=raw_data
    )


def normalize_items(raw_items: List[Dict[str, Any]]) -> List[NormalizedItem]:
    """
    Normalize a list of raw items.

    Args:
        raw_items: List of raw item dictionaries

    Returns:
        List of NormalizedItem instances
    """
    normalized = []

    for raw_item in raw_items:
        try:
            item = normalize_item(raw_item)
            normalized.append(item)
        except Exception as e:
            logger.error("Failed to normalize item %s: %s",
                         raw_item.get('name', 'Unknown') if raw_item else 'None', str(e))
            # Continue processing other items
            continue

    logger.info(
        f"Successfully normalized {len(normalized)} out of {len(raw_items)} items")
    return normalized


if __name__ == "__main__":
    # Test the price model
    logging.basicConfig(level=logging.INFO)

    # Test rarity mapping
    print("Testing rarity mapping:")
    test_rarities = ["Consumer", "Blue",
                     "Mil-Spec", "purple", "Classified", "red"]
    for rarity in test_rarities:
        rank = RarityMapper.get_rarity_rank(rarity)
        standard = RarityMapper.get_standard_rarity_name(rarity)
        print(f"  {rarity} -> rank: {rank}, standard: {standard}")

    # Test StatTrak detection
    print("\nTesting StatTrak detection:")
    test_names = [
        "AK-47 | Redline (Field-Tested)",
        "StatTrak™ AK-47 | Redline (Field-Tested)",
        "M4A4 | Howl (Factory New)"
    ]
    for name in test_names:
        is_st = StatTrakDetector._detect_from_name(name)
        print(f"  {name} -> StatTrak: {is_st}")

    # Test price selection
    print("\nTesting price selection:")
    test_item = {
        'name': 'AK-47 | Redline (FT)',
        'prices': {
            'steam': {'lowest_sell': 25.50},
            'market_price': 30.00
        },
        'avg_price': 27.75
    }
    price = PriceSelector.select_price(test_item)
    print(f"  Selected price: ${price}")
