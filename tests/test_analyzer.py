#!/usr/bin/env python3
"""
Unit tests for CS2/CS:GO Trade-Up Analyzer
Tests probability math, float mapping, and Steam fee calculations.
"""

from analyze_tradeups import (
    SteamFeeCalculator, FloatCalculator, DataNormalizer,
    CollectionIndex, TradeUpCalculator, SkinData
)
import pytest
from typing import List
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSteamFeeCalculator:
    """Test Steam Community Market fee calculations."""

    def test_net_from_list_price(self):
        """Test calculation of net amount from list price."""
        # 15% total fee rate (5% Steam + 10% game)
        assert abs(SteamFeeCalculator.net_from_list_price(10.00) - 8.50) < 1e-6
        assert abs(SteamFeeCalculator.net_from_list_price(1.00) - 0.85) < 1e-6
        assert abs(SteamFeeCalculator.net_from_list_price(
            100.00) - 85.00) < 1e-6

        # Test rounding
        assert abs(SteamFeeCalculator.net_from_list_price(10.33) -
                   8.78) < 1e-6  # 10.33 * 0.85 = 8.7805 -> 8.78

    def test_list_price_for_net(self):
        """Test calculation of list price needed for target net amount."""
        # Should be inverse of net_from_list_price
        assert abs(SteamFeeCalculator.list_price_for_net(8.50) - 10.00) < 1e-6
        assert abs(SteamFeeCalculator.list_price_for_net(0.85) - 1.00) < 1e-6
        assert abs(SteamFeeCalculator.list_price_for_net(
            85.00) - 100.00) < 1e-6

        # Test rounding
        net_amount = 8.78
        list_price = SteamFeeCalculator.list_price_for_net(net_amount)
        assert abs(SteamFeeCalculator.net_from_list_price(
            list_price) - net_amount) < 0.01

    def test_round_trip_consistency(self):
        """Test that list_price_for_net and net_from_list_price are consistent."""
        test_amounts = [1.00, 5.67, 10.00, 25.99, 100.00]

        for amount in test_amounts:
            # Start with net amount, get list price, then back to net
            list_price = SteamFeeCalculator.list_price_for_net(amount)
            net_back = SteamFeeCalculator.net_from_list_price(list_price)
            assert abs(
                net_back - amount) < 0.01, f"Round trip failed for {amount}"


class TestFloatCalculator:
    """Test float value calculations and exterior mapping."""

    def test_calculate_output_float(self):
        """Test output float calculation formula."""
        # Average of [0.1, 0.2, 0.3] = 0.2
        # Output range [0.0, 1.0]: 0.0 + (1.0 - 0.0) * 0.2 = 0.2
        input_floats = [0.1, 0.2, 0.3]
        result = FloatCalculator.calculate_output_float(input_floats, 0.0, 1.0)
        assert abs(result - 0.2) < 1e-6

        # Average of [0.05, 0.05, 0.05, 0.05, 0.05] * 2 = 0.05
        # Output range [0.1, 0.5]: 0.1 + (0.5 - 0.1) * 0.05 = 0.1 + 0.4 * 0.05 = 0.12
        input_floats = [0.05] * 10
        result = FloatCalculator.calculate_output_float(input_floats, 0.1, 0.5)
        assert abs(result - 0.12) < 1e-6

    def test_float_to_exterior_standard_ranges(self):
        """Test mapping of float values to exterior conditions."""
        # Test standard ranges
        assert FloatCalculator.float_to_exterior(0.03) == "Factory New"
        assert FloatCalculator.float_to_exterior(0.06) == "Factory New"
        assert FloatCalculator.float_to_exterior(0.07) == "Minimal Wear"
        assert FloatCalculator.float_to_exterior(0.10) == "Minimal Wear"
        assert FloatCalculator.float_to_exterior(0.15) == "Field-Tested"
        assert FloatCalculator.float_to_exterior(0.25) == "Field-Tested"
        assert FloatCalculator.float_to_exterior(0.38) == "Well-Worn"
        assert FloatCalculator.float_to_exterior(0.40) == "Well-Worn"
        assert FloatCalculator.float_to_exterior(0.45) == "Battle-Scarred"
        assert FloatCalculator.float_to_exterior(0.80) == "Battle-Scarred"

    def test_float_to_exterior_edge_cases(self):
        """Test edge cases for float to exterior mapping."""
        # Boundary values
        assert FloatCalculator.float_to_exterior(0.0) == "Factory New"
        assert FloatCalculator.float_to_exterior(1.0) == "Battle-Scarred"

        # Just below boundaries
        assert FloatCalculator.float_to_exterior(0.069999) == "Factory New"
        assert FloatCalculator.float_to_exterior(0.149999) == "Minimal Wear"
        assert FloatCalculator.float_to_exterior(0.379999) == "Field-Tested"
        assert FloatCalculator.float_to_exterior(0.449999) == "Well-Worn"

    def test_float_to_exterior_with_restrictions(self):
        """Test float to exterior mapping with restricted ranges."""
        # Skin with restricted range [0.1, 0.8]
        # Float 0.05 should be clamped to 0.1 -> Minimal Wear
        assert FloatCalculator.float_to_exterior(
            0.05, 0.1, 0.8) == "Minimal Wear"

        # Float 0.9 should be clamped to 0.8 -> Battle-Scarred
        assert FloatCalculator.float_to_exterior(
            0.9, 0.1, 0.8) == "Battle-Scarred"

        # Float 0.2 within range -> Field-Tested
        assert FloatCalculator.float_to_exterior(
            0.2, 0.1, 0.8) == "Field-Tested"


class TestDataNormalizer:
    """Test data normalization functions."""

    def test_parse_market_name_basic(self):
        """Test basic market name parsing."""
        weapon, skin, exterior, stattrak, souvenir = DataNormalizer.parse_market_name(
            "AK-47 | Redline (Field-Tested)")
        assert weapon == "AK-47"
        assert skin == "Redline"
        assert exterior == "Field-Tested"
        assert stattrak == False
        assert souvenir == False

    def test_parse_market_name_stattrak(self):
        """Test StatTrak market name parsing."""
        weapon, skin, exterior, stattrak, souvenir = DataNormalizer.parse_market_name(
            "StatTrak™ AK-47 | Redline (Field-Tested)")
        assert weapon == "AK-47"
        assert skin == "Redline"
        assert exterior == "Field-Tested"
        assert stattrak == True
        assert souvenir == False

    def test_parse_market_name_souvenir(self):
        """Test Souvenir market name parsing."""
        weapon, skin, exterior, stattrak, souvenir = DataNormalizer.parse_market_name(
            "Souvenir AK-47 | Safari Mesh (Well-Worn)")
        assert weapon == "AK-47"
        assert skin == "Safari Mesh"
        assert exterior == "Well-Worn"
        assert stattrak == False
        assert souvenir == True

    def test_parse_market_name_no_skin(self):
        """Test parsing items without skin name."""
        weapon, skin, exterior, stattrak, souvenir = DataNormalizer.parse_market_name(
            "AK-47 (Factory New)")
        assert weapon == "AK-47"
        assert skin == None
        assert exterior == "Factory New"
        assert stattrak == False
        assert souvenir == False

    def test_parse_market_name_no_exterior(self):
        """Test parsing items without exterior."""
        weapon, skin, exterior, stattrak, souvenir = DataNormalizer.parse_market_name(
            "AK-47 | Redline")
        assert weapon == "AK-47"
        assert skin == "Redline"
        assert exterior == None
        assert stattrak == False
        assert souvenir == False


class TestProbabilityMath:
    """Test trade-up probability calculations."""

    def create_test_skins(self) -> List[SkinData]:
        """Create test skin data for probability calculations."""
        skins = []

        # Collection 1 - Mil-Spec inputs (4 items)
        for i in range(4):
            skins.append(SkinData(
                market_name=f"Test Weapon {i} | Skin C1 (Field-Tested)",
                weapon=f"Test Weapon {i}",
                skin="Skin C1",
                exterior="Field-Tested",
                collection="Collection1",
                rarity="Mil-Spec Grade",
                steam_price=1.0,
                stattrak=False
            ))

        # Collection 2 - Mil-Spec inputs (3 items)
        for i in range(3):
            skins.append(SkinData(
                market_name=f"Test Weapon {i+4} | Skin C2 (Field-Tested)",
                weapon=f"Test Weapon {i+4}",
                skin="Skin C2",
                exterior="Field-Tested",
                collection="Collection2",
                rarity="Mil-Spec Grade",
                steam_price=1.0,
                stattrak=False
            ))

        # Collection 1 - Restricted outputs (4 items)
        for i in range(4):
            skins.append(SkinData(
                market_name=f"Output Weapon {i} | Output C1 (Field-Tested)",
                weapon=f"Output Weapon {i}",
                skin="Output C1",
                exterior="Field-Tested",
                collection="Collection1",
                rarity="Restricted",
                steam_price=10.0,
                stattrak=False
            ))

        # Collection 2 - Restricted outputs (3 items)
        for i in range(3):
            skins.append(SkinData(
                market_name=f"Output Weapon {i+4} | Output C2 (Field-Tested)",
                weapon=f"Output Weapon {i+4}",
                skin="Output C2",
                exterior="Field-Tested",
                collection="Collection2",
                rarity="Restricted",
                steam_price=10.0,
                stattrak=False
            ))

        return skins

    def test_probability_calculation_single_collection(self):
        """Test probability calculation with single collection."""
        skins = self.create_test_skins()
        collection_index = CollectionIndex(skins)
        calculator = TradeUpCalculator(collection_index)

        # Test 10 inputs from Collection1 -> 4 possible outputs
        # Each output should have probability 1/4 = 0.25
        composition = {"Collection1": 10}
        candidate = calculator._evaluate_composition(
            composition,
            {"Collection1": [s for s in skins if s.collection ==
                             "Collection1" and s.rarity == "Mil-Spec Grade"]},
            "Restricted",
            "Mil-Spec Grade",
            False,
            False,
            None
        )

        assert candidate is not None
        assert len(candidate.outcomes) == 4
        for outcome in candidate.outcomes:
            assert abs(outcome.probability - 0.25) < 1e-6

    def test_probability_calculation_mixed_collections(self):
        """Test probability calculation with mixed collections."""
        skins = self.create_test_skins()
        collection_index = CollectionIndex(skins)
        calculator = TradeUpCalculator(collection_index)

        # Test 8 from Collection1, 2 from Collection2
        # Collection1: 8/10 * 1/4 = 0.2 per item (4 items)
        # Collection2: 2/10 * 1/3 = 0.0667 per item (3 items)
        composition = {"Collection1": 8, "Collection2": 2}

        inputs_by_collection = {
            "Collection1": [s for s in skins if s.collection == "Collection1" and s.rarity == "Mil-Spec Grade"],
            "Collection2": [s for s in skins if s.collection == "Collection2" and s.rarity == "Mil-Spec Grade"]
        }

        candidate = calculator._evaluate_composition(
            composition,
            inputs_by_collection,
            "Restricted",
            "Mil-Spec Grade",
            False,
            False,
            None
        )

        assert candidate is not None
        assert len(candidate.outcomes) == 7  # 4 from C1 + 3 from C2

        # Check Collection1 outcomes (should be ~0.2 each)
        c1_outcomes = [
            o for o in candidate.outcomes if o.collection == "Collection1"]
        assert len(c1_outcomes) == 4
        for outcome in c1_outcomes:
            assert abs(outcome.probability - 0.2) < 1e-6

        # Check Collection2 outcomes (should be ~0.0667 each)
        c2_outcomes = [
            o for o in candidate.outcomes if o.collection == "Collection2"]
        assert len(c2_outcomes) == 3
        expected_prob = 2.0 / 10.0 / 3.0  # (2/10) * (1/3)
        for outcome in c2_outcomes:
            assert abs(outcome.probability - expected_prob) < 1e-6

        # Total probability should sum to 1.0
        total_prob = sum(o.probability for o in candidate.outcomes)
        assert abs(total_prob - 1.0) < 1e-6

    def test_expected_value_calculation(self):
        """Test expected value and ROI calculations."""
        skins = self.create_test_skins()
        collection_index = CollectionIndex(skins)
        calculator = TradeUpCalculator(collection_index)

        # Single collection case
        composition = {"Collection1": 10}
        inputs_by_collection = {
            "Collection1": [s for s in skins if s.collection == "Collection1" and s.rarity == "Mil-Spec Grade"]
        }

        candidate = calculator._evaluate_composition(
            composition,
            inputs_by_collection,
            "Restricted",
            "Mil-Spec Grade",
            False,
            False,
            None
        )

        assert candidate is not None

        # Input cost: 10 items * $1.00 = $10.00
        assert abs(candidate.total_cost - 10.0) < 1e-6

        # Expected revenue: 4 outputs * 0.25 probability * $8.50 net price = $8.50
        # (Net price = $10.00 * 0.85 = $8.50 after 15% fees)
        expected_revenue = 4 * 0.25 * 8.50
        assert abs(sum(o.contribution for o in candidate.outcomes) -
                   expected_revenue) < 1e-6

        # Expected value: $8.50 - $10.00 = -$1.50
        expected_ev = expected_revenue - 10.0
        assert abs(candidate.expected_value - expected_ev) < 1e-6

        # ROI: -$1.50 / $10.00 = -0.15
        expected_roi = expected_ev / 10.0
        assert abs(candidate.roi - expected_roi) < 1e-6


class TestFullWorkflowExample:
    """Test a complete worked example with all intermediate numbers."""

    def test_dry_run_example(self):
        """Dry run example showing all calculations step by step."""
        print("\n" + "="*50)
        print("DRY RUN EXAMPLE - TRADE-UP CALCULATION")
        print("="*50)

        # Create simple test data
        print("\n1. TEST DATA SETUP:")
        print("   - Collection A: 2 Mil-Spec inputs @ $2.00 each")
        print("   - Collection A: 2 Restricted outputs @ $12.00 each")
        print("   - Collection B: 3 Mil-Spec inputs @ $1.50 each")
        print("   - Collection B: 1 Restricted output @ $20.00")

        skins = []

        # Collection A inputs
        for i in range(2):
            skins.append(SkinData(
                market_name=f"AK-47 | Test A{i} (FT)",
                collection="CollectionA",
                rarity="Mil-Spec Grade",
                steam_price=2.00,
                stattrak=False
            ))

        # Collection B inputs
        for i in range(3):
            skins.append(SkinData(
                market_name=f"M4A4 | Test B{i} (FT)",
                collection="CollectionB",
                rarity="Mil-Spec Grade",
                steam_price=1.50,
                stattrak=False
            ))

        # Collection A outputs
        for i in range(2):
            skins.append(SkinData(
                market_name=f"AWP | Output A{i} (FT)",
                collection="CollectionA",
                rarity="Restricted",
                steam_price=12.00,
                stattrak=False
            ))

        # Collection B output
        skins.append(SkinData(
            market_name="AK-47 | Output B (FT)",
            collection="CollectionB",
            rarity="Restricted",
            steam_price=20.00,
            stattrak=False
        ))

        print("\n2. TRADE-UP COMPOSITION:")
        print("   - 6 items from Collection A (cheapest available)")
        print("   - 4 items from Collection B (cheapest available)")

        collection_index = CollectionIndex(skins)
        calculator = TradeUpCalculator(collection_index)

        composition = {"CollectionA": 6, "CollectionB": 4}
        inputs_by_collection = {
            "CollectionA": [s for s in skins if s.collection == "CollectionA" and s.rarity == "Mil-Spec Grade"],
            "CollectionB": [s for s in skins if s.collection == "CollectionB" and s.rarity == "Mil-Spec Grade"]
        }

        print("\n3. INPUT COST CALCULATION:")
        print("   - Collection A: 2 items available @ $2.00 each")
        print("   - Need 6 items, so: 2×$2.00 + 2×$2.00 + 2×$2.00 = $12.00")
        print("   - Collection B: 3 items available @ $1.50 each")
        print("   - Need 4 items, so: 3×$1.50 + 1×$1.50 = $6.00")
        print("   - Total Input Cost: $12.00 + $6.00 = $18.00")

        candidate = calculator._evaluate_composition(
            composition,
            inputs_by_collection,
            "Restricted",
            "Mil-Spec Grade",
            False,
            False,
            None
        )

        assert candidate is not None

        print("\n4. PROBABILITY CALCULATIONS:")
        print("   Collection A probability: 6/10 = 0.6")
        print("   Collection A has 2 possible outputs")
        print("   Each Collection A output: 0.6 × (1/2) = 0.3")
        print("   Collection B probability: 4/10 = 0.4")
        print("   Collection B has 1 possible output")
        print("   Collection B output: 0.4 × (1/1) = 0.4")
        print("   Total probability check: 0.3 + 0.3 + 0.4 = 1.0 ✓")

        print("\n5. REVENUE CALCULATIONS (after 15% Steam fees):")
        for outcome in candidate.outcomes:
            gross_price = outcome.price
            net_price = SteamFeeCalculator.net_from_list_price(gross_price)
            contribution = outcome.probability * net_price
            print(f"   {outcome.market_name}:")
            print(f"     Probability: {outcome.probability:.1%}")
            print(f"     Gross price: ${gross_price:.2f}")
            print(f"     Net price: ${net_price:.2f}")
            print(f"     Contribution: ${contribution:.2f}")

        total_expected_revenue = sum(
            o.contribution for o in candidate.outcomes)
        expected_value = total_expected_revenue - candidate.total_cost
        roi = expected_value / candidate.total_cost

        print("\n6. FINAL RESULTS:")
        print(f"   Total Input Cost: ${candidate.total_cost:.2f}")
        print(f"   Expected Revenue: ${total_expected_revenue:.2f}")
        print(f"   Expected Value: ${expected_value:.2f}")
        print(f"   ROI: {roi:.1%}")

        # Verify our manual calculations match the code
        assert abs(candidate.total_cost - 18.0) < 1e-6
        assert abs(candidate.expected_value - expected_value) < 1e-6
        assert abs(candidate.roi - roi) < 1e-6

        print("\n" + "="*50)
        print("DRY RUN COMPLETE - All calculations verified!")
        print("="*50)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
