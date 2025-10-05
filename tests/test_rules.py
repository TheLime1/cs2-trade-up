"""
Tests for trade-up rules module.
"""

import pytest
from tradeups.rules import TradeUpRules, TargetEnumerator, TradeUpCalculator, build_target_enumerator
from tradeups.price_model import NormalizedItem


class TestTradeUpRules:
    """Test the TradeUpRules class."""

    def test_next_rarity_rank(self):
        """Test getting next rarity rank."""
        # Normal progression
        assert TradeUpRules.get_next_rarity_rank(0) == 1
        assert TradeUpRules.get_next_rarity_rank(1) == 2
        assert TradeUpRules.get_next_rarity_rank(2) == 3

        # At highest rarity
        assert TradeUpRules.get_next_rarity_rank(
            8) is None  # Covert is highest
        assert TradeUpRules.get_next_rarity_rank(
            99) is None  # Invalid high rank

    def test_next_rarity_name(self):
        """Test getting next rarity name."""
        assert TradeUpRules.get_next_rarity_name(
            "Consumer Grade") == "Industrial Grade"
        assert TradeUpRules.get_next_rarity_name(
            "Mil-Spec Grade") == "Restricted"
        assert TradeUpRules.get_next_rarity_name("Covert") is None

    def test_can_trade_up_rarity(self):
        """Test checking if rarity can be traded up."""
        assert TradeUpRules.can_trade_up_rarity(0) is True  # Consumer
        assert TradeUpRules.can_trade_up_rarity(
            4) is True  # Classified -> Covert
        assert TradeUpRules.can_trade_up_rarity(5) is False  # Covert (highest)

    def test_validate_trade_up_inputs_valid(self):
        """Test validation with valid inputs."""
        items = self._create_test_items(10, rarity_rank=2)
        is_valid, message = TradeUpRules.validate_trade_up_inputs(items)
        assert is_valid is True
        assert "Valid" in message

    def test_validate_trade_up_inputs_wrong_count(self):
        """Test validation with wrong number of inputs."""
        items = self._create_test_items(5, rarity_rank=2)
        is_valid, message = TradeUpRules.validate_trade_up_inputs(items)
        assert is_valid is False
        assert "exactly 10" in message

    def test_validate_trade_up_inputs_mismatched_rarity(self):
        """Test validation with mismatched rarities."""
        items = self._create_test_items(10, rarity_rank=2)
        items[0].rarity_rank = 1  # Different rarity
        is_valid, message = TradeUpRules.validate_trade_up_inputs(items)
        assert is_valid is False
        assert "same rarity" in message

    def test_validate_trade_up_inputs_mismatched_stattrak(self):
        """Test validation with mismatched StatTrak."""
        items = self._create_test_items(10, rarity_rank=2)
        items[0].stattrak = True  # Different StatTrak
        is_valid, message = TradeUpRules.validate_trade_up_inputs(items)
        assert is_valid is False
        assert "StatTrak" in message

    def test_validate_trade_up_inputs_untradeable(self):
        """Test validation with untradeable items."""
        items = self._create_test_items(10, rarity_rank=2)
        items[0].price = 0.0  # Make untradeable
        is_valid, message = TradeUpRules.validate_trade_up_inputs(items)
        assert is_valid is False
        assert "not tradeable" in message

    def _create_test_items(self, count: int, rarity_rank: int = 2, stattrak: bool = False) -> list:
        """Create test items for validation."""
        items = []
        for i in range(count):
            item = NormalizedItem(
                id=f"test_{i}",
                market_hash_name=f"Test Item {i}",
                weapon="AK-47",
                skin="Test",
                stattrak=stattrak,
                collection="Test Collection",
                rarity="Mil-Spec Grade",
                rarity_rank=rarity_rank,
                price=10.0
            )
            items.append(item)
        return items


class TestTargetEnumerator:
    """Test the TargetEnumerator class."""

    def test_build_indices(self):
        """Test building lookup indices."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        # Check indices are built
        assert "Collection A" in enumerator.collection_rarity_items
        assert "Collection B" in enumerator.collection_rarity_items
        assert 2 in enumerator.collection_rarity_items["Collection A"]
        assert 3 in enumerator.collection_rarity_items["Collection A"]

    def test_get_targets_for_collection(self):
        """Test getting targets for a collection."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        # Get targets from Collection A, rarity 2 -> 3
        targets = enumerator.get_targets_for_collection(
            "Collection A", 2, False)

        # Should get restricted items from Collection A
        assert len(targets) == 2
        assert all(item.rarity_rank == 3 for item in targets)
        assert all(item.collection == "Collection A" for item in targets)
        assert all(item.stattrak is False for item in targets)

    def test_get_targets_for_collection_stattrak(self):
        """Test getting StatTrak targets."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        # Get StatTrak targets
        targets = enumerator.get_targets_for_collection(
            "Collection A", 2, True)

        # Should get StatTrak restricted items
        assert len(targets) == 1
        assert all(item.stattrak is True for item in targets)

    def test_get_targets_for_collection_no_targets(self):
        """Test getting targets when none exist."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        # Try to get targets from highest rarity (should be none)
        targets = enumerator.get_targets_for_collection(
            "Collection A", 8, False)
        assert len(targets) == 0

    def test_get_all_targets(self):
        """Test getting all targets for multiple collections."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        all_targets = enumerator.get_all_targets(
            ["Collection A", "Collection B"], 2, False)

        assert "Collection A" in all_targets
        assert "Collection B" in all_targets
        assert len(all_targets["Collection A"]) == 2
        assert len(all_targets["Collection B"]) == 2

    def test_has_valid_targets(self):
        """Test checking if collection has valid targets."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)

        # Collection A has rarity 3 items (targets for rarity 2)
        assert enumerator.has_valid_targets("Collection A", 2) is True

        # Collection A has no rarity 9 items (no targets for rarity 8)
        assert enumerator.has_valid_targets("Collection A", 8) is False

        # Unknown collection
        assert enumerator.has_valid_targets("Unknown", 2) is False

    def _create_test_items(self) -> dict:
        """Create test items for target enumeration."""
        items_by_collection = {
            "Collection A": [
                # Mil-Spec (rank 2)
                NormalizedItem("item1", "Item 1", "AK-47", "Test1",
                               False, "Collection A", "Mil-Spec Grade", 2, 10.0),
                NormalizedItem("item2", "Item 2", "M4A4", "Test2",
                               False, "Collection A", "Mil-Spec Grade", 2, 12.0),
                # Restricted (rank 3) - targets for Mil-Spec
                NormalizedItem("item3", "Item 3", "AWP", "Test3",
                               False, "Collection A", "Restricted", 3, 25.0),
                NormalizedItem("item4", "Item 4", "AK-47", "Test4",
                               False, "Collection A", "Restricted", 3, 30.0),
                # StatTrak variant
                NormalizedItem("item5", "Item 5 ST", "AWP", "Test5",
                               True, "Collection A", "Restricted", 3, 50.0),
            ],
            "Collection B": [
                # Mil-Spec (rank 2)
                NormalizedItem("item6", "Item 6", "M4A1-S", "Test6",
                               False, "Collection B", "Mil-Spec Grade", 2, 8.0),
                # Restricted (rank 3) - targets for Mil-Spec
                NormalizedItem("item7", "Item 7", "AK-47", "Test7",
                               False, "Collection B", "Restricted", 3, 20.0),
                NormalizedItem("item8", "Item 8", "M4A4", "Test8",
                               False, "Collection B", "Restricted", 3, 22.0),
            ]
        }
        return items_by_collection


class TestTradeUpCalculator:
    """Test the TradeUpCalculator class."""

    def test_calculate_probabilities(self):
        """Test calculating collection probabilities."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)
        calculator = TradeUpCalculator(enumerator)

        # 7 from Collection A, 3 from Collection B
        inputs = ([items["Collection A"][0]] * 7 +
                  [items["Collection B"][0]] * 3)

        probs = calculator.calculate_probabilities(inputs)

        assert probs["Collection A"] == 0.7
        assert probs["Collection B"] == 0.3

    def test_calculate_outcomes(self):
        """Test calculating detailed outcomes."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)
        calculator = TradeUpCalculator(enumerator)

        # All from Collection A (should have 100% success)
        inputs = [items["Collection A"][0]] * 10

        outcomes, success_prob, dead_prob = calculator.calculate_outcomes(
            inputs)

        # Should have 100% success (Collection A has targets)
        assert success_prob == 1.0
        assert dead_prob == 0.0

        # Should have 2 possible outcomes (2 restricted items in Collection A)
        assert len(outcomes) == 2

        # Each outcome should have 50% probability (uniform within collection)
        for outcome in outcomes:
            assert outcome['probability'] == 0.5
            assert outcome['collection'] == "Collection A"

    def test_calculate_expected_value(self):
        """Test calculating expected value."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)
        calculator = TradeUpCalculator(enumerator)

        # All from Collection A
        inputs = [items["Collection A"][0]] * 10  # 10 * $10 = $100 input cost

        ev, total_cost, margin_pct = calculator.calculate_expected_value(
            inputs, fee_rate=0.15)

        assert total_cost == 100.0  # 10 * $10

        # Expected output value (before fees): 0.5 * $25 + 0.5 * $30 = $27.5
        # After 15% fees: $27.5 * 0.85 = $23.375
        # Net EV: $23.375 - $100 = -$76.625
        expected_ev = 23.375 - 100.0
        assert abs(ev - expected_ev) < 0.01

        # Margin should be negative
        assert margin_pct < 0

    def test_mixed_collection_outcomes(self):
        """Test outcomes with mixed collections."""
        items = self._create_test_items()
        enumerator = TargetEnumerator(items)
        calculator = TradeUpCalculator(enumerator)

        # 5 from each collection
        inputs = ([items["Collection A"][0]] * 5 +
                  [items["Collection B"][0]] * 5)

        outcomes, success_prob, dead_prob = calculator.calculate_outcomes(
            inputs)

        # Should have 100% success (both collections have targets)
        assert success_prob == 1.0
        assert dead_prob == 0.0

        # Should have 4 possible outcomes (2 from each collection)
        assert len(outcomes) == 4

        # Check probabilities
        collection_a_outcomes = [
            o for o in outcomes if o['collection'] == "Collection A"]
        collection_b_outcomes = [
            o for o in outcomes if o['collection'] == "Collection B"]

        # Each collection gets 50% probability, split among its items
        for outcome in collection_a_outcomes:
            assert outcome['probability'] == 0.25  # 0.5 / 2 items
        for outcome in collection_b_outcomes:
            assert outcome['probability'] == 0.25  # 0.5 / 2 items

    def _create_test_items(self) -> dict:
        """Create test items for calculator testing."""
        return {
            "Collection A": [
                # Mil-Spec input
                NormalizedItem("input1", "Input 1", "AK-47", "Test1",
                               False, "Collection A", "Mil-Spec Grade", 2, 10.0),
                # Restricted targets
                NormalizedItem("target1", "Target 1", "AWP", "Test1",
                               False, "Collection A", "Restricted", 3, 25.0),
                NormalizedItem("target2", "Target 2", "AK-47", "Test2",
                               False, "Collection A", "Restricted", 3, 30.0),
            ],
            "Collection B": [
                # Mil-Spec input
                NormalizedItem("input2", "Input 2", "M4A1-S", "Test3",
                               False, "Collection B", "Mil-Spec Grade", 2, 8.0),
                # Restricted targets
                NormalizedItem("target3", "Target 3", "AK-47", "Test3",
                               False, "Collection B", "Restricted", 3, 20.0),
                NormalizedItem("target4", "Target 4", "M4A4", "Test4",
                               False, "Collection B", "Restricted", 3, 22.0),
            ]
        }


class TestBuildTargetEnumerator:
    """Test the build_target_enumerator function."""

    def test_build_from_items_list(self):
        """Test building enumerator from items list."""
        items = [
            NormalizedItem("item1", "Item 1", "AK-47", "Test1",
                           False, "Collection A", "Mil-Spec Grade", 2, 10.0),
            NormalizedItem("item2", "Item 2", "AWP", "Test2",
                           False, "Collection A", "Restricted", 3, 25.0),
            NormalizedItem("item3", "Item 3", "M4A4", "Test3",
                           False, "Collection B", "Mil-Spec Grade", 2, 8.0),
            # Untradeable item (should be filtered out)
            NormalizedItem("item4", "Item 4", "Glock", "Test4",
                           False, "Collection C", "Consumer Grade", 0, 0.0),
        ]

        enumerator = build_target_enumerator(items)

        # Should only include tradeable items with collections
        assert "Collection A" in enumerator.items_by_collection
        assert "Collection B" in enumerator.items_by_collection
        # Untradeable item filtered out
        assert "Collection C" not in enumerator.items_by_collection

        assert len(enumerator.items_by_collection["Collection A"]) == 2
        assert len(enumerator.items_by_collection["Collection B"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
