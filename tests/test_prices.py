"""
Tests for price model module.
"""

import pytest
from tradeups.price_model import (
    NormalizedItem, RarityMapper, PriceSelector, StatTrakDetector,
    normalize_item, normalize_items
)


class TestNormalizedItem:
    """Test the NormalizedItem class."""

    def test_creation(self):
        """Test creating a normalized item."""
        item = NormalizedItem(
            id="test_item",
            market_hash_name="AK-47 | Redline (Field-Tested)",
            weapon="AK-47",
            skin="Redline",
            stattrak=False,
            collection="Spectrum Collection",
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=25.50
        )

        assert item.id == "test_item"
        assert item.market_hash_name == "AK-47 | Redline (Field-Tested)"
        assert item.weapon == "AK-47"
        assert item.skin == "Redline"
        assert item.stattrak is False
        assert item.collection == "Spectrum Collection"
        assert item.rarity == "Mil-Spec Grade"
        assert item.rarity_rank == 2
        assert item.price == 25.50

    def test_price_validation(self):
        """Test price validation on creation."""
        item = NormalizedItem(
            id="test",
            market_hash_name="Test Item",
            weapon=None,
            skin=None,
            stattrak=False,
            collection="Test",
            rarity="Consumer Grade",
            rarity_rank=0,
            price=-5.0  # Negative price should be corrected to 0
        )

        assert item.price == 0.0

    def test_id_generation(self):
        """Test ID generation when not provided."""
        item = NormalizedItem(
            id="",  # Empty ID should trigger generation
            market_hash_name="AK-47 | Redline (Field-Tested)",
            weapon="AK-47",
            skin="Redline",
            stattrak=False,
            collection="Test",
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=25.50
        )

        assert item.id != ""
        assert "ak" in item.id.lower()
        assert "redline" in item.id.lower()

    def test_is_tradeable(self):
        """Test tradeable property."""
        # Tradeable item
        tradeable_item = NormalizedItem(
            id="test1",
            market_hash_name="Test Item",
            weapon=None,
            skin=None,
            stattrak=False,
            collection="Test Collection",
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=10.0
        )
        assert tradeable_item.is_tradeable is True

        # Non-tradeable (no price)
        no_price_item = NormalizedItem(
            id="test2",
            market_hash_name="Test Item",
            weapon=None,
            skin=None,
            stattrak=False,
            collection="Test Collection",
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=0.0
        )
        assert no_price_item.is_tradeable is False

        # Non-tradeable (no collection)
        no_collection_item = NormalizedItem(
            id="test3",
            market_hash_name="Test Item",
            weapon=None,
            skin=None,
            stattrak=False,
            collection=None,
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=10.0
        )
        assert no_collection_item.is_tradeable is False


class TestRarityMapper:
    """Test the RarityMapper class."""

    def test_rarity_order(self):
        """Test rarity order mapping."""
        assert RarityMapper.get_rarity_rank("Consumer Grade") == 0
        assert RarityMapper.get_rarity_rank("Industrial Grade") == 1
        assert RarityMapper.get_rarity_rank("Mil-Spec Grade") == 2
        assert RarityMapper.get_rarity_rank("Restricted") == 3
        assert RarityMapper.get_rarity_rank("Classified") == 4
        assert RarityMapper.get_rarity_rank("Covert") == 5

    def test_rarity_aliases(self):
        """Test rarity alias mapping."""
        # Color aliases
        assert RarityMapper.get_rarity_rank(
            "blue") == RarityMapper.get_rarity_rank("Mil-Spec Grade")
        assert RarityMapper.get_rarity_rank(
            "purple") == RarityMapper.get_rarity_rank("Restricted")
        assert RarityMapper.get_rarity_rank(
            "pink") == RarityMapper.get_rarity_rank("Classified")
        assert RarityMapper.get_rarity_rank(
            "red") == RarityMapper.get_rarity_rank("Covert")

        # Short names
        assert RarityMapper.get_rarity_rank(
            "consumer") == RarityMapper.get_rarity_rank("Consumer Grade")
        assert RarityMapper.get_rarity_rank(
            "milspec") == RarityMapper.get_rarity_rank("Mil-Spec Grade")

    def test_standard_rarity_name(self):
        """Test getting standard rarity names."""
        assert RarityMapper.get_standard_rarity_name(
            "blue") == "Mil-Spec Grade"
        assert RarityMapper.get_standard_rarity_name("purple") == "Restricted"
        assert RarityMapper.get_standard_rarity_name(
            "consumer") == "Consumer Grade"

    def test_unknown_rarity(self):
        """Test handling unknown rarity."""
        # Should default to Consumer Grade (rank 0)
        assert RarityMapper.get_rarity_rank("unknown_rarity") == 0
        assert RarityMapper.get_standard_rarity_name(
            "unknown_rarity") == "Consumer Grade"

    def test_partial_matching(self):
        """Test partial rarity matching."""
        # Should match "Mil-Spec" even if input is "mil spec"
        rank1 = RarityMapper.get_rarity_rank("mil spec")
        rank2 = RarityMapper.get_rarity_rank("Mil-Spec Grade")
        assert rank1 == rank2


class TestPriceSelector:
    """Test the PriceSelector class."""

    def test_direct_price_field(self):
        """Test selecting from direct price fields."""
        item_data = {
            'price': 25.50,
            'other_field': 'value'
        }

        price = PriceSelector.select_price(item_data)
        assert abs(price - 25.50) < 0.01

    def test_nested_price_field(self):
        """Test selecting from nested price fields."""
        item_data = {
            'steam': {
                'lowest_sell': 23.45
            },
            'market_price': 30.00
        }

        price = PriceSelector.select_price(item_data)
        # Should prefer steam.lowest_sell
        assert abs(price - 23.45) < 0.01

    def test_prices_object(self):
        """Test selecting from prices object."""
        item_data = {
            'prices': {
                'steam_price': 28.75,
                'market_price': 32.00
            }
        }

        price = PriceSelector.select_price(item_data)
        assert abs(price - 28.75) < 0.01

    def test_string_price_parsing(self):
        """Test parsing string prices."""
        item_data = {
            'price': '$15.25'
        }

        price = PriceSelector.select_price(item_data)
        assert abs(price - 15.25) < 0.01

    def test_no_valid_price(self):
        """Test when no valid price is found."""
        item_data = {
            'name': 'Test Item',
            'rarity': 'Mil-Spec'
        }

        price = PriceSelector.select_price(item_data)
        assert price == 0.0

    def test_zero_price_fallback(self):
        """Test fallback when price is zero."""
        item_data = {
            'steam': {
                'lowest_sell': 0.0
            },
            'market_price': 15.50
        }

        price = PriceSelector.select_price(item_data)
        # Should fallback to market_price since lowest_sell is 0
        assert abs(price - 15.50) < 0.01


class TestStatTrakDetector:
    """Test the StatTrakDetector class."""

    def test_explicit_boolean_field(self):
        """Test explicit StatTrak boolean field."""
        item_data = {'stattrak': True}
        assert StatTrakDetector.is_stattrak(item_data) is True

        item_data = {'stattrak': False}
        assert StatTrakDetector.is_stattrak(item_data) is False

    def test_name_based_detection(self):
        """Test StatTrak detection from item name."""
        # StatTrak item
        st_item = {'name': 'StatTrak™ AK-47 | Redline (Field-Tested)'}
        assert StatTrakDetector.is_stattrak(st_item) is True

        # Non-StatTrak item
        normal_item = {'name': 'AK-47 | Redline (Field-Tested)'}
        assert StatTrakDetector.is_stattrak(normal_item) is False

        # Alternative StatTrak format
        alt_st_item = {'name': 'AK-47 | Redline (Field-Tested) StatTrak'}
        assert StatTrakDetector.is_stattrak(alt_st_item) is True

    def test_alternative_field_names(self):
        """Test detection from alternative field names."""
        # is_stattrak field
        item_data = {'is_stattrak': True}
        assert StatTrakDetector.is_stattrak(item_data) is True

        # stat_trak field
        item_data = {'stat_trak': 1}
        assert StatTrakDetector.is_stattrak(item_data) is True

        # st field
        item_data = {'st': 'true'}
        assert StatTrakDetector.is_stattrak(item_data) is True

    def test_case_insensitive_detection(self):
        """Test case-insensitive detection."""
        item_data = {'name': 'stattrak ak-47 | redline'}
        assert StatTrakDetector.is_stattrak(item_data) is True


class TestNormalizeItem:
    """Test the normalize_item function."""

    def test_basic_normalization(self):
        """Test basic item normalization."""
        raw_data = {
            'name': 'AK-47 | Redline (Field-Tested)',
            'weapon': 'AK-47',
            'skin': 'Redline',
            'collection': 'Spectrum Collection',
            'rarity': 'Mil-Spec Grade',
            'price': 25.50,
            'stattrak': False
        }

        item = normalize_item(raw_data)

        assert item.market_hash_name == 'AK-47 | Redline (Field-Tested)'
        assert item.weapon == 'AK-47'
        assert item.skin == 'Redline'
        assert item.collection == 'Spectrum Collection'
        assert item.rarity == 'Mil-Spec Grade'
        assert item.rarity_rank == RarityMapper.get_rarity_rank(
            'Mil-Spec Grade')
        assert abs(item.price - 25.50) < 0.01
        assert item.stattrak is False

    def test_stattrak_normalization(self):
        """Test StatTrak normalization."""
        raw_data = {
            'name': 'StatTrak™ AK-47 | Redline (Field-Tested)',
            'weapon': 'AK-47',
            'skin': 'Redline',
            'collection': 'Spectrum Collection',
            'rarity': 'Mil-Spec Grade',
            'price': 50.00
        }

        item = normalize_item(raw_data)
        assert item.stattrak is True

    def test_price_normalization(self):
        """Test price normalization from complex data."""
        raw_data = {
            'name': 'Test Item',
            'collection': 'Test Collection',
            'rarity': 'Consumer Grade',
            'prices': {
                'steam': {
                    'lowest_sell': 12.34
                }
            }
        }

        item = normalize_item(raw_data)
        assert abs(item.price - 12.34) < 0.01

    def test_missing_fields(self):
        """Test normalization with missing fields."""
        raw_data = {
            'name': 'Minimal Item'
        }

        item = normalize_item(raw_data)

        # Should handle missing fields gracefully
        assert item.market_hash_name == 'Minimal Item'
        assert item.weapon is None
        assert item.skin is None
        assert item.collection is None
        assert item.price == 0.0  # Default for missing price
        assert item.stattrak is False  # Default for missing StatTrak


class TestNormalizeItems:
    """Test the normalize_items function."""

    def test_normalize_list(self):
        """Test normalizing a list of items."""
        raw_items = [
            {
                'name': 'Item 1',
                'collection': 'Collection A',
                'rarity': 'Mil-Spec Grade',
                'price': 10.0
            },
            {
                'name': 'Item 2',
                'collection': 'Collection B',
                'rarity': 'Restricted',
                'price': 25.0
            }
        ]

        items = normalize_items(raw_items)

        assert len(items) == 2
        assert items[0].market_hash_name == 'Item 1'
        assert items[1].market_hash_name == 'Item 2'
        assert items[0].collection == 'Collection A'
        assert items[1].collection == 'Collection B'

    def test_error_handling(self):
        """Test error handling during normalization."""
        raw_items = [
            {
                'name': 'Valid Item',
                'collection': 'Test Collection',
                'rarity': 'Mil-Spec Grade',
                'price': 10.0
            },
            None,  # This should cause an error but not crash
            {
                'name': 'Another Valid Item',
                'collection': 'Test Collection',
                'rarity': 'Restricted',
                'price': 20.0
            }
        ]

        # Should not crash and should return valid items
        items = normalize_items(raw_items)

        # Should have fewer items than input due to error
        assert len(items) < len(raw_items)
        assert all(item.market_hash_name in [
                   'Valid Item', 'Another Valid Item'] for item in items)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
