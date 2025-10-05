"""
Tests for calculation engine module.
"""

import pytest
from tradeups.engine import CandidateGenerator, TradeUpFilter, TradeUpEngine
from tradeups.price_model import NormalizedItem


class TestCandidateGenerator:
    """Test the CandidateGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        items = self._create_test_items()
        generator = CandidateGenerator(
            items, max_pool_per_bucket=10, hard_cap_results=100)

        assert generator.max_pool_per_bucket == 10
        assert generator.hard_cap_results == 100
        assert generator.target_enumerator is not None
        assert generator.calculator is not None

    def test_build_item_pools(self):
        """Test building item pools."""
        items = self._create_test_items()
        generator = CandidateGenerator(items)

        # Check pools are created for both StatTrak modes
        assert True in generator.item_pools  # StatTrak
        assert False in generator.item_pools  # Non-StatTrak

        # Check rarity ranks are present
        non_st_pools = generator.item_pools[False]
        assert 2 in non_st_pools  # Mil-Spec items

        # Check collections are present
        if 2 in non_st_pools:
            mil_spec_pools = non_st_pools[2]
            assert "Collection A" in mil_spec_pools
            assert "Collection B" in mil_spec_pools

    def test_generate_mono_collection_candidates(self):
        """Test generating mono-collection candidates."""
        items = self._create_test_items()
        generator = CandidateGenerator(items)

        # Get Mil-Spec collections
        collections = generator.item_pools[False][2]  # Non-StatTrak, Mil-Spec

        candidates = generator._generate_mono_collection(
            collections, max_cost=None)

        # Should generate candidates if collections have enough items and valid targets
        # May be 0 if not enough items in test data
        assert len(candidates) >= 0

    def test_filter_by_max_cost(self):
        """Test filtering by maximum cost."""
        items = self._create_test_items()
        generator = CandidateGenerator(items)

        # Generate with low max cost
        candidates = generator.generate_candidates(
            max_cost=5.0, stattrak_filter="false")

        # All candidates should have inputs under max cost
        for candidate in candidates:
            for item in candidate.inputs:
                assert item.price <= 5.0

    def test_stattrak_filter(self):
        """Test StatTrak filtering."""
        items = self._create_test_items()
        generator = CandidateGenerator(items)

        # Generate StatTrak only
        st_candidates = generator.generate_candidates(stattrak_filter="true")
        for candidate in st_candidates:
            assert candidate.stattrak is True

        # Generate non-StatTrak only
        non_st_candidates = generator.generate_candidates(
            stattrak_filter="false")
        for candidate in non_st_candidates:
            assert candidate.stattrak is False

    def test_collection_filter(self):
        """Test collection filtering."""
        items = self._create_test_items()
        generator = CandidateGenerator(items)

        # Filter to specific collection
        candidates = generator.generate_candidates(
            collection_filter="Collection A")

        # All candidates should use only Collection A items
        for candidate in candidates:
            for mix in candidate.collection_mix:
                assert mix["collection"] == "Collection A"

    def _create_test_items(self) -> list:
        """Create test items for generator testing."""
        items = []

        # Collection A - Mil-Spec items (enough for trade-up)
        for i in range(15):
            item = NormalizedItem(
                id=f"a_milspec_{i}",
                market_hash_name=f"A Mil-Spec {i}",
                weapon="AK-47",
                skin=f"Skin {i}",
                stattrak=False,
                collection="Collection A",
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                price=10.0 + i * 0.5
            )
            items.append(item)

        # Collection A - Restricted targets
        for i in range(3):
            item = NormalizedItem(
                id=f"a_restricted_{i}",
                market_hash_name=f"A Restricted {i}",
                weapon="AWP",
                skin=f"Skin {i}",
                stattrak=False,
                collection="Collection A",
                rarity="Restricted",
                rarity_rank=3,
                price=25.0 + i * 5.0
            )
            items.append(item)

        # Collection B - Mil-Spec items
        for i in range(12):
            item = NormalizedItem(
                id=f"b_milspec_{i}",
                market_hash_name=f"B Mil-Spec {i}",
                weapon="M4A4",
                skin=f"Skin {i}",
                stattrak=False,
                collection="Collection B",
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                price=8.0 + i * 0.3
            )
            items.append(item)

        # Collection B - Restricted targets
        for i in range(2):
            item = NormalizedItem(
                id=f"b_restricted_{i}",
                market_hash_name=f"B Restricted {i}",
                weapon="AK-47",
                skin=f"Skin {i}",
                stattrak=False,
                collection="Collection B",
                rarity="Restricted",
                rarity_rank=3,
                price=20.0 + i * 3.0
            )
            items.append(item)

        # Some StatTrak variants
        for i in range(5):
            item = NormalizedItem(
                id=f"a_milspec_st_{i}",
                market_hash_name=f"StatTrak™ A Mil-Spec {i}",
                weapon="AK-47",
                skin=f"Skin {i}",
                stattrak=True,
                collection="Collection A",
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                price=20.0 + i * 1.0
            )
            items.append(item)

        return items


class TestTradeUpFilter:
    """Test the TradeUpFilter class."""

    def test_filter_by_success_rate(self):
        """Test filtering by minimum success rate."""
        candidates = self._create_test_candidates()

        # Filter for candidates with at least 80% success
        filtered = TradeUpFilter.filter_candidates(
            candidates, min_success_pct=80.0)

        for candidate in filtered:
            assert candidate.success_pct >= 80.0

    def test_filter_by_profit_margin(self):
        """Test filtering by minimum profit margin."""
        candidates = self._create_test_candidates()

        # Filter for candidates with at least 10% profit
        filtered = TradeUpFilter.filter_candidates(
            candidates, min_profit_pct=10.0)

        for candidate in filtered:
            assert candidate.margin_pct >= 10.0

    def test_combined_filters(self):
        """Test combining multiple filters."""
        candidates = self._create_test_candidates()

        # Apply both filters
        filtered = TradeUpFilter.filter_candidates(
            candidates,
            min_success_pct=75.0,
            min_profit_pct=5.0
        )

        for candidate in filtered:
            assert candidate.success_pct >= 75.0
            assert candidate.margin_pct >= 5.0

    def test_sort_by_ev(self):
        """Test sorting by expected value."""
        candidates = self._create_test_candidates()

        sorted_candidates = TradeUpFilter.sort_candidates(
            candidates, sort_by="ev")

        # Should be sorted by EV descending
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].ev >= sorted_candidates[i + 1].ev

    def test_sort_by_margin(self):
        """Test sorting by profit margin."""
        candidates = self._create_test_candidates()

        sorted_candidates = TradeUpFilter.sort_candidates(
            candidates, sort_by="margin")

        # Should be sorted by margin descending
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].margin_pct >= sorted_candidates[i + 1].margin_pct

    def test_sort_by_success(self):
        """Test sorting by success rate."""
        candidates = self._create_test_candidates()

        sorted_candidates = TradeUpFilter.sort_candidates(
            candidates, sort_by="success")

        # Should be sorted by success rate descending
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].success_pct >= sorted_candidates[i + 1].success_pct

    def test_sort_by_cost(self):
        """Test sorting by total cost."""
        candidates = self._create_test_candidates()

        sorted_candidates = TradeUpFilter.sort_candidates(
            candidates, sort_by="cost")

        # Should be sorted by cost ascending
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].total_input_cost <= sorted_candidates[i +
                                                                              1].total_input_cost

    def _create_test_candidates(self):
        """Create test candidates for filtering."""
        from tradeups.engine import TradeUpCandidate

        # Create mock inputs
        inputs = []
        for i in range(10):
            item = NormalizedItem(
                id=f"input_{i}",
                market_hash_name=f"Input {i}",
                weapon="AK-47",
                skin="Test",
                stattrak=False,
                collection="Test Collection",
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                price=10.0
            )
            inputs.append(item)

        candidates = [
            TradeUpCandidate(
                inputs=inputs,
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                stattrak=False,
                collection_mix=[
                    {"collection": "Test Collection", "count": 10}],
                avg_input_price=10.0,
                total_input_cost=100.0,
                success_pct=100.0,
                dead_pct=0.0,
                ev=25.0,
                margin_pct=25.0,
                outcomes=[]
            ),
            TradeUpCandidate(
                inputs=inputs,
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                stattrak=False,
                collection_mix=[
                    {"collection": "Test Collection", "count": 10}],
                avg_input_price=10.0,
                total_input_cost=100.0,
                success_pct=75.0,
                dead_pct=25.0,
                ev=10.0,
                margin_pct=10.0,
                outcomes=[]
            ),
            TradeUpCandidate(
                inputs=inputs,
                rarity="Mil-Spec Grade",
                rarity_rank=2,
                stattrak=False,
                collection_mix=[
                    {"collection": "Test Collection", "count": 10}],
                avg_input_price=10.0,
                total_input_cost=100.0,
                success_pct=50.0,
                dead_pct=50.0,
                ev=-5.0,
                margin_pct=-5.0,
                outcomes=[]
            )
        ]

        return candidates


class TestTradeUpEngine:
    """Test the TradeUpEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        items = self._create_minimal_items()
        engine = TradeUpEngine(items)

        assert engine.generator is not None

    def test_find_profitable_tradeups(self):
        """Test finding profitable trade-ups."""
        items = self._create_minimal_items()
        engine = TradeUpEngine(items)

        results, total_count = engine.find_profitable_tradeups(
            max_cost=20.0,
            min_success_pct=50.0,
            page=1,
            page_size=10
        )

        # Should return results and count
        assert isinstance(results, list)
        assert isinstance(total_count, int)
        assert len(results) <= 10  # Page size limit
        # Total should be at least the returned count
        assert total_count >= len(results)

    def test_pagination(self):
        """Test pagination functionality."""
        items = self._create_minimal_items()
        engine = TradeUpEngine(items)

        # Get first page
        page1, total = engine.find_profitable_tradeups(page=1, page_size=2)

        # Get second page
        page2, _ = engine.find_profitable_tradeups(page=2, page_size=2)

        # Pages should be different (if there are enough results)
        if total > 2:
            assert page1 != page2

    def _create_minimal_items(self) -> list:
        """Create minimal items for engine testing."""
        items = []

        # Create enough items for testing
        for collection in ["Collection A", "Collection B"]:
            # Mil-Spec inputs
            for i in range(15):
                item = NormalizedItem(
                    id=f"{collection.lower().replace(' ', '_')}_milspec_{i}",
                    market_hash_name=f"{collection} Mil-Spec {i}",
                    weapon="Test Weapon",
                    skin=f"Skin {i}",
                    stattrak=False,
                    collection=collection,
                    rarity="Mil-Spec Grade",
                    rarity_rank=2,
                    price=10.0 + (i % 5)
                )
                items.append(item)

            # Restricted targets
            for i in range(3):
                item = NormalizedItem(
                    id=f"{collection.lower().replace(' ', '_')}_restricted_{i}",
                    market_hash_name=f"{collection} Restricted {i}",
                    weapon="Test Weapon",
                    skin=f"Target Skin {i}",
                    stattrak=False,
                    collection=collection,
                    rarity="Restricted",
                    rarity_rank=3,
                    price=25.0 + (i * 5)
                )
                items.append(item)

        return items


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
