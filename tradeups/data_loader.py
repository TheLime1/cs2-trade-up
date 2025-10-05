"""
Data loader module for CS2 trade-ups calculator.

Handles fetching, caching, and schema mapping of the remote skins database.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dateutil import parser

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and caching of CS2 skins database."""

    def __init__(
        self,
        data_url: str,
        cache_path: str,
        cache_ttl_hours: int = 12
    ):
        self.data_url = data_url
        self.cache_path = Path(cache_path)
        self.cache_ttl_hours = cache_ttl_hours
        self._data: Optional[List[Dict[str, Any]]] = None
        self._last_updated: Optional[datetime] = None

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def get_data(self, force_refresh: bool = False) -> tuple[List[Dict[str, Any]], datetime]:
        """
        Get the skins data, either from cache or by fetching from remote.

        Args:
            force_refresh: If True, bypasses cache and fetches fresh data

        Returns:
            Tuple of (data_list, last_updated_datetime)
        """
        if not force_refresh and self._should_use_cache():
            logger.info("Using cached data")
            return self._data, self._last_updated

        # Try to fetch fresh data
        try:
            fresh_data = self._fetch_remote_data()
            self._save_to_cache(fresh_data)
            self._data = fresh_data
            self._last_updated = datetime.now()
            logger.info("Successfully fetched fresh data")
            return self._data, self._last_updated

        except Exception as e:
            logger.error(f"Failed to fetch remote data: {e}")

            # Fall back to cache if available
            if self._load_from_cache():
                logger.warning("Using cached data due to fetch failure")
                return self._data, self._last_updated
            else:
                logger.error("No cached data available")
                raise RuntimeError(
                    "Unable to load data from remote or cache") from e

    def _should_use_cache(self) -> bool:
        """Check if we should use cached data."""
        if self._data is not None and self._last_updated is not None:
            age = datetime.now() - self._last_updated
            return age < timedelta(hours=self.cache_ttl_hours)

        return self._load_from_cache()

    def _load_from_cache(self) -> bool:
        """Load data from cache file."""
        if not self.cache_path.exists():
            return False

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Extract metadata and data
            if isinstance(cache_data, dict) and 'data' in cache_data:
                self._data = cache_data['data']
                timestamp_str = cache_data.get('timestamp')
                if timestamp_str:
                    self._last_updated = parser.parse(timestamp_str)
                else:
                    # Fallback to file modification time
                    stat = self.cache_path.stat()
                    self._last_updated = datetime.fromtimestamp(stat.st_mtime)
            else:
                # Legacy format - data directly in file
                self._data = cache_data
                stat = self.cache_path.stat()
                self._last_updated = datetime.fromtimestamp(stat.st_mtime)

            # Check if cache is still valid
            age = datetime.now() - self._last_updated
            if age < timedelta(hours=self.cache_ttl_hours):
                logger.info(f"Loaded valid cache data (age: {age})")
                return True
            else:
                logger.info(f"Cache data is stale (age: {age})")
                return False

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def _fetch_remote_data(self) -> List[Dict[str, Any]]:
        """Fetch data from remote URL."""
        logger.info(f"Fetching data from {self.data_url}")

        response = requests.get(self.data_url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, dict) and 'items' in data:
            return data['items']
        else:
            logger.warning("Unexpected data format, using as-is")
            return [data] if isinstance(data, dict) else []

    def _save_to_cache(self, data: List[Dict[str, Any]]) -> None:
        """Save data to cache file."""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'meta': {
                'source_url': self.data_url,
                'items_count': len(data)
            }
        }

        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} items to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_cache_age_minutes(self) -> Optional[int]:
        """Get the age of cached data in minutes."""
        if self._last_updated is None:
            return None

        age = datetime.now() - self._last_updated
        return int(age.total_seconds() / 60)


class SchemaMapper:
    """Maps different schema variations to a normalized format."""

    # Common field name variations
    FIELD_MAPPINGS = {
        'name': ['market_hash_name', 'name', 'item_name', 'full_name'],
        'weapon': ['weapon', 'weapon_name', 'gun', 'weapon_type'],
        'skin': ['skin', 'skin_name', 'paint', 'finish'],
        'collection': ['collection', 'case', 'case_name', 'container'],
        'rarity': ['rarity', 'grade', 'quality', 'tier'],
        'price': ['price', 'steam_price', 'market_price', 'lowest_sell'],
        'stattrak': ['stattrak', 'stat_trak', 'st', 'is_stattrak'],
        'wear': ['wear', 'wear_name', 'condition', 'exterior'],
        'paint_index': ['paint_index', 'skin_id', 'paint_id'],
        'prices': ['prices', 'price_data', 'market_data']
    }

    @classmethod
    def map_item(cls, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a raw item from the JSON to normalized fields.

        Args:
            raw_item: Raw item data from JSON

        Returns:
            Dictionary with normalized field names
        """
        mapped = {}

        # Map each field using the mapping table
        for normalized_field, possible_names in cls.FIELD_MAPPINGS.items():
            value = cls._find_field_value(raw_item, possible_names)
            if value is not None:
                mapped[normalized_field] = value

        # Special handling for complex fields
        mapped.update(cls._handle_special_fields(raw_item, mapped))

        # Ensure we have the raw data for debugging
        mapped['_raw'] = raw_item

        return mapped

    @classmethod
    def _find_field_value(cls, data: Dict[str, Any], field_names: List[str]) -> Any:
        """Find the first available field value from a list of possible names."""
        for field_name in field_names:
            if field_name in data and data[field_name] is not None:
                return data[field_name]
        return None

    @classmethod
    def _handle_special_fields(cls, raw_item: Dict[str, Any], mapped: Dict[str, Any]) -> Dict[str, Any]:
        """Handle special field processing that requires custom logic."""
        result = {}

        # Construct full name if not present
        if 'name' not in mapped:
            name_parts = []
            if mapped.get('weapon'):
                name_parts.append(mapped['weapon'])
            if mapped.get('skin'):
                name_parts.append(f"| {mapped['skin']}")
            if mapped.get('wear'):
                name_parts.append(f"({mapped['wear']})")
            if name_parts:
                result['name'] = ' '.join(name_parts)

        # Handle price extraction from complex price objects
        if 'price' not in mapped and mapped.get('prices'):
            result['price'] = cls._extract_best_price(mapped['prices'])

        # Handle StatTrak detection
        if 'stattrak' not in mapped:
            result['stattrak'] = cls._detect_stattrak(mapped.get('name', ''))

        return result

    @classmethod
    def _extract_best_price(cls, prices_data: Any) -> Optional[float]:
        """Extract the best price from a complex prices object."""
        if isinstance(prices_data, (int, float)):
            return float(prices_data)

        if isinstance(prices_data, dict):
            # Try different price fields in order of preference
            price_fields = [
                'steam.lowest_sell', 'lowest_sell', 'steam_price',
                'market_price', 'price', 'avg_price', 'median_price'
            ]

            for field in price_fields:
                if '.' in field:
                    # Handle nested fields like 'steam.lowest_sell'
                    parts = field.split('.')
                    value = prices_data
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    if value is not None:
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            continue
                else:
                    # Handle direct fields
                    if field in prices_data:
                        try:
                            return float(prices_data[field])
                        except (ValueError, TypeError):
                            continue

        return None

    @classmethod
    def _detect_stattrak(cls, name: str) -> bool:
        """Detect if an item is StatTrak based on its name."""
        if not name:
            return False
        return name.lower().startswith('stattrak™') or 'stattrak' in name.lower()


def load_skins_data(
    data_url: str = "https://raw.githubusercontent.com/TheLime1/cs2-price-database/refs/heads/main/data/skins_database.json",
    cache_path: str = "data/skins_database.json",
    cache_ttl_hours: int = 12,
    force_refresh: bool = False
) -> tuple[List[Dict[str, Any]], datetime]:
    """
    Convenience function to load and normalize skins data.

    Args:
        data_url: URL to fetch data from
        cache_path: Path to cache file
        cache_ttl_hours: Cache TTL in hours
        force_refresh: Force refresh from remote

    Returns:
        Tuple of (normalized_items, last_updated)
    """
    loader = DataLoader(data_url, cache_path, cache_ttl_hours)
    raw_data, last_updated = loader.get_data(force_refresh)

    # Normalize all items
    normalized_items = []
    for raw_item in raw_data:
        try:
            normalized = SchemaMapper.map_item(raw_item)
            normalized_items.append(normalized)
        except Exception as e:
            logger.warning(f"Failed to normalize item: {e}")
            # Include raw item for debugging
            normalized_items.append({'_raw': raw_item, '_error': str(e)})

    logger.info(f"Normalized {len(normalized_items)} items")
    return normalized_items, last_updated


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)

    try:
        items, updated = load_skins_data()
        print(f"Loaded {len(items)} items, last updated: {updated}")

        # Show sample items
        for i, item in enumerate(items[:3]):
            print(f"\nSample item {i+1}:")
            for key, value in item.items():
                if key != '_raw':  # Skip raw data for readability
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
