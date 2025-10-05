"""
Calculation engine for CS2 trade-ups.

Handles candidate generation, EV calculations, filtering, and pagination.
"""

import logging
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .price_model import NormalizedItem
from .rules import TradeUpCalculator, TargetEnumerator, build_target_enumerator

logger = logging.getLogger(__name__)


@dataclass
class TradeUpCandidate:
    """Represents a candidate trade-up recipe."""

    inputs: List[NormalizedItem]
    rarity: str
    rarity_rank: int
    stattrak: bool
    collection_mix: List[Dict[str, Any]]
    avg_input_price: float
    total_input_cost: float
    success_pct: float
    dead_pct: float
    ev: float
    margin_pct: float
    outcomes: List[Dict[str, Any]]

    @property
    def inputs_example(self) -> List[str]:
        """Get example input item names."""
        return [item.display_name for item in self.inputs]


class CandidateGenerator:
    """Generates trade-up candidates using various strategies."""

    def __init__(
        self,
        items: List[NormalizedItem],
        max_pool_per_bucket: int = 50,
        hard_cap_results: int = 500
    ):
        """
        Initialize the candidate generator.

        Args:
            items: List of all available items
            max_pool_per_bucket: Max items per rarity/collection bucket
            hard_cap_results: Maximum number of candidates to generate
        """
        self.items = items
        self.max_pool_per_bucket = max_pool_per_bucket
        self.hard_cap_results = hard_cap_results

        # Build lookup structures
        self._build_item_pools()

        # Initialize calculator
        self.target_enumerator = build_target_enumerator(items)
        self.calculator = TradeUpCalculator(self.target_enumerator)

    def _build_item_pools(self):
        """Build item pools organized by rarity, stattrak, and collection."""
        # Pool structure: stattrak -> rarity_rank -> collection -> [items]
        self.item_pools: Dict[bool, Dict[int, Dict[str, List[NormalizedItem]]]] = {
            True: {},   # StatTrak items
            False: {}   # Non-StatTrak items
        }

        for item in self.items:
            if not item.is_tradeable:
                continue

            stattrak = item.stattrak
            rarity_rank = item.rarity_rank
            collection = item.collection or "Unknown"

            # Initialize nested structure
            if rarity_rank not in self.item_pools[stattrak]:
                self.item_pools[stattrak][rarity_rank] = {}

            if collection not in self.item_pools[stattrak][rarity_rank]:
                self.item_pools[stattrak][rarity_rank][collection] = []

            self.item_pools[stattrak][rarity_rank][collection].append(item)

        # Sort items by price (cheapest first) and apply pool limits
        for stattrak_pools in self.item_pools.values():
            for rarity_pools in stattrak_pools.values():
                for collection, items in rarity_pools.items():
                    # Sort by price ascending
                    items.sort(key=lambda x: x.price)
                    # Limit pool size
                    if len(items) > self.max_pool_per_bucket:
                        rarity_pools[collection] = items[:self.max_pool_per_bucket]

        # Log pool statistics
        self._log_pool_stats()

    def _log_pool_stats(self):
        """Log statistics about item pools."""
        total_items = 0
        total_collections = 0

        for stattrak, stattrak_pools in self.item_pools.items():
            for rarity_rank, rarity_pools in stattrak_pools.items():
                for collection, items in rarity_pools.items():
                    total_items += len(items)
                    total_collections += 1

        logger.info(
            f"Built item pools: {total_collections} collection/rarity buckets, {total_items} total items")

    def generate_candidates(
        self,
        max_cost: Optional[float] = None,
        collection_filter: Optional[str] = None,
        stattrak_filter: str = "both"  # "true", "false", "both"
    ) -> List[TradeUpCandidate]:
        """
        Generate trade-up candidates.

        Args:
            max_cost: Maximum price per input item
            collection_filter: Filter to specific collection ("any" for all)
            stattrak_filter: StatTrak filter mode

        Returns:
            List of trade-up candidates
        """
        candidates = []
        generated_count = 0

        # Determine StatTrak modes to process
        stattrak_modes = self._get_stattrak_modes(stattrak_filter)

        for stattrak in stattrak_modes:
            if stattrak not in self.item_pools:
                continue

            for rarity_rank in sorted(self.item_pools[stattrak].keys()):
                # Check if this rarity can be traded up
                from .rules import TradeUpRules
                if not TradeUpRules.can_trade_up_rarity(rarity_rank):
                    continue

                rarity_pools = self.item_pools[stattrak][rarity_rank]

                # Filter collections if specified
                collections_to_process = self._filter_collections(
                    rarity_pools, collection_filter)

                # Generate candidates for this rarity/stattrak combination
                rarity_candidates = self._generate_for_rarity(
                    collections_to_process,
                    rarity_rank,
                    stattrak,
                    max_cost
                )

                candidates.extend(rarity_candidates)
                generated_count += len(rarity_candidates)

                # Check hard cap
                if generated_count >= self.hard_cap_results:
                    logger.info(
                        f"Hit hard cap of {self.hard_cap_results} candidates")
                    break

            if generated_count >= self.hard_cap_results:
                break

        # Truncate to hard cap
        if len(candidates) > self.hard_cap_results:
            candidates = candidates[:self.hard_cap_results]

        logger.info(f"Generated {len(candidates)} candidates")
        return candidates

    def _get_stattrak_modes(self, stattrak_filter: str) -> List[bool]:
        """Get list of StatTrak modes to process."""
        if stattrak_filter == "true":
            return [True]
        elif stattrak_filter == "false":
            return [False]
        else:  # "both"
            return [False, True]

    def _filter_collections(
        self,
        rarity_pools: Dict[str, List[NormalizedItem]],
        collection_filter: Optional[str]
    ) -> Dict[str, List[NormalizedItem]]:
        """Filter collections based on filter criteria."""
        if not collection_filter or collection_filter.lower() in ["any", "all"]:
            return rarity_pools

        # Filter to specific collection
        if collection_filter in rarity_pools:
            return {collection_filter: rarity_pools[collection_filter]}

        return {}

    def _generate_for_rarity(
        self,
        collections: Dict[str, List[NormalizedItem]],
        rarity_rank: int,
        stattrak: bool,
        max_cost: Optional[float]
    ) -> List[TradeUpCandidate]:
        """Generate candidates for a specific rarity."""
        candidates = []

        # Strategy 1: Mono-collection stacks (10 from same collection)
        mono_candidates = self._generate_mono_collection(collections, max_cost)
        candidates.extend(mono_candidates)

        # Strategy 2: Bi-collection mixes
        if len(collections) >= 2:
            bi_candidates = self._generate_bi_collection(collections, max_cost)
            candidates.extend(bi_candidates)

        # Convert to TradeUpCandidate objects
        trade_up_candidates = []
        for candidate_inputs in candidates:
            try:
                candidate = self._create_candidate(candidate_inputs)
                if candidate:
                    trade_up_candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to create candidate: {e}")
                continue

        return trade_up_candidates

    def _generate_mono_collection(
        self,
        collections: Dict[str, List[NormalizedItem]],
        max_cost: Optional[float]
    ) -> List[List[NormalizedItem]]:
        """Generate mono-collection candidates."""
        candidates = []

        for collection, items in collections.items():
            # Filter by max cost
            if max_cost:
                items = [item for item in items if item.price <= max_cost]

            if len(items) >= 10:
                # Check if this collection has valid targets
                if self.target_enumerator.has_valid_targets(collection, items[0].rarity_rank):
                    # Take 10 cheapest items
                    candidates.append(items[:10])

        return candidates

    def _generate_bi_collection(
        self,
        collections: Dict[str, List[NormalizedItem]],
        max_cost: Optional[float]
    ) -> List[List[NormalizedItem]]:
        """Generate bi-collection mix candidates."""
        candidates = []

        collection_names = list(collections.keys())

        # Try different split ratios
        split_ratios = [(7, 3), (6, 4), (5, 5)]

        for i, collection1 in enumerate(collection_names):
            for j, collection2 in enumerate(collection_names[i+1:], i+1):
                items1 = collections[collection1]
                items2 = collections[collection2]

                # Filter by max cost
                if max_cost:
                    items1 = [item for item in items1 if item.price <= max_cost]
                    items2 = [item for item in items2 if item.price <= max_cost]

                # Skip if either collection is empty after filtering
                if not items1 or not items2:
                    continue

                # Check if both collections have valid targets
                if not (self.target_enumerator.has_valid_targets(collection1, items1[0].rarity_rank) and
                        self.target_enumerator.has_valid_targets(collection2, items2[0].rarity_rank)):
                    continue

                for count1, count2 in split_ratios:
                    if len(items1) >= count1 and len(items2) >= count2:
                        candidate = items1[:count1] + items2[:count2]
                        candidates.append(candidate)

                    # Try reverse split
                    if count1 != count2 and len(items1) >= count2 and len(items2) >= count1:
                        candidate = items1[:count2] + items2[:count1]
                        candidates.append(candidate)

        return candidates

    def _create_candidate(self, inputs: List[NormalizedItem]) -> Optional[TradeUpCandidate]:
        """Create a TradeUpCandidate from input items."""
        if len(inputs) != 10:
            return None

        try:
            # Calculate outcomes and metrics
            outcomes, success_pct, dead_pct = self.calculator.calculate_outcomes(
                inputs)
            ev, total_cost, margin_pct = self.calculator.calculate_expected_value(
                inputs)

            # Calculate collection mix
            collection_counts = {}
            for item in inputs:
                collection = item.collection or "Unknown"
                collection_counts[collection] = collection_counts.get(
                    collection, 0) + 1

            collection_mix = [
                {"collection": collection, "count": count}
                for collection, count in collection_counts.items()
            ]

            # Calculate average input price
            avg_price = total_cost / len(inputs) if inputs else 0.0

            return TradeUpCandidate(
                inputs=inputs,
                rarity=inputs[0].rarity,
                rarity_rank=inputs[0].rarity_rank,
                stattrak=inputs[0].stattrak,
                collection_mix=collection_mix,
                avg_input_price=avg_price,
                total_input_cost=total_cost,
                success_pct=success_pct * 100,  # Convert to percentage
                dead_pct=dead_pct * 100,
                ev=ev,
                margin_pct=margin_pct,
                outcomes=outcomes
            )

        except Exception as e:
            logger.error(f"Failed to calculate candidate metrics: {e}")
            return None


class TradeUpFilter:
    """Filters trade-up candidates based on criteria."""

    @staticmethod
    def filter_candidates(
        candidates: List[TradeUpCandidate],
        min_success_pct: Optional[float] = None,
        min_profit_pct: Optional[float] = None
    ) -> List[TradeUpCandidate]:
        """
        Filter candidates based on success and profit criteria.

        Args:
            candidates: List of candidates to filter
            min_success_pct: Minimum success percentage
            min_profit_pct: Minimum profit margin percentage

        Returns:
            Filtered list of candidates
        """
        filtered = candidates

        if min_success_pct is not None:
            filtered = [
                c for c in filtered if c.success_pct >= min_success_pct]

        if min_profit_pct is not None:
            filtered = [c for c in filtered if c.margin_pct >= min_profit_pct]

        return filtered

    @staticmethod
    def sort_candidates(
        candidates: List[TradeUpCandidate],
        sort_by: str = "ev"
    ) -> List[TradeUpCandidate]:
        """
        Sort candidates by specified criteria.

        Args:
            candidates: List of candidates to sort
            sort_by: Sort criteria ("ev", "margin", "success", "cost")

        Returns:
            Sorted list of candidates
        """
        if sort_by == "ev":
            return sorted(candidates, key=lambda x: x.ev, reverse=True)
        elif sort_by == "margin":
            return sorted(candidates, key=lambda x: x.margin_pct, reverse=True)
        elif sort_by == "success":
            return sorted(candidates, key=lambda x: x.success_pct, reverse=True)
        elif sort_by == "cost":
            return sorted(candidates, key=lambda x: x.total_input_cost)
        else:
            # Default to EV
            return sorted(candidates, key=lambda x: x.ev, reverse=True)


class TradeUpEngine:
    """Main engine that coordinates candidate generation and filtering."""

    def __init__(
        self,
        items: List[NormalizedItem],
        max_pool_per_bucket: int = 50,
        hard_cap_results: int = 500
    ):
        """
        Initialize the trade-up engine.

        Args:
            items: List of all available items
            max_pool_per_bucket: Max items per rarity/collection bucket  
            hard_cap_results: Maximum number of results to return
        """
        self.generator = CandidateGenerator(
            items, max_pool_per_bucket, hard_cap_results)

    def find_profitable_tradeups(
        self,
        max_cost: Optional[float] = None,
        collection: Optional[str] = None,
        min_success_pct: Optional[float] = None,
        min_profit_pct: Optional[float] = None,
        stattrak: str = "both",
        sort_by: str = "ev",
        page: int = 1,
        page_size: int = 25
    ) -> Tuple[List[TradeUpCandidate], int]:
        """
        Find profitable trade-ups based on criteria.

        Args:
            max_cost: Maximum price per input item
            collection: Collection filter
            min_success_pct: Minimum success percentage
            min_profit_pct: Minimum profit margin percentage
            stattrak: StatTrak filter ("true", "false", "both")
            sort_by: Sort criteria
            page: Page number (1-based)
            page_size: Items per page

        Returns:
            Tuple of (candidates_page, total_count)
        """
        # Generate candidates
        candidates = self.generator.generate_candidates(
            max_cost=max_cost,
            collection_filter=collection,
            stattrak_filter=stattrak
        )

        # Apply filters
        candidates = TradeUpFilter.filter_candidates(
            candidates,
            min_success_pct=min_success_pct,
            min_profit_pct=min_profit_pct
        )

        # Sort candidates
        candidates = TradeUpFilter.sort_candidates(candidates, sort_by)

        # Apply pagination
        total_count = len(candidates)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_candidates = candidates[start_idx:end_idx]

        return page_candidates, total_count


if __name__ == "__main__":
    # Test the calculation engine
    logging.basicConfig(level=logging.INFO)

    # Create mock items for testing
    from .price_model import NormalizedItem

    mock_items = []
    collections = ["Mirage 2021", "Nuke 2021", "Dust2 2021"]
    rarities = [("Consumer Grade", 0), ("Industrial Grade", 1),
                ("Mil-Spec Grade", 2), ("Restricted", 3)]

    item_id = 0
    for collection in collections:
        for rarity_name, rarity_rank in rarities:
            for i in range(20):  # 20 items per collection/rarity
                item_id += 1
                price = 1.0 + (rarity_rank * 5) + (i * 0.1)  # Varying prices

                item = NormalizedItem(
                    id=f"item_{item_id}",
                    market_hash_name=f"Test Weapon | Skin {item_id} (Field-Tested)",
                    weapon="Test Weapon",
                    skin=f"Skin {item_id}",
                    stattrak=False,
                    collection=collection,
                    rarity=rarity_name,
                    rarity_rank=rarity_rank,
                    price=price
                )
                mock_items.append(item)

    print(f"Created {len(mock_items)} mock items")

    # Test engine
    engine = TradeUpEngine(mock_items)

    results, total = engine.find_profitable_tradeups(
        max_cost=10.0,
        min_success_pct=50.0,
        page_size=5
    )

    print(f"\nFound {total} total candidates, showing first {len(results)}:")
    for i, candidate in enumerate(results):
        print(f"  {i+1}. {candidate.rarity} | Success: {candidate.success_pct:.1f}% | "
              f"EV: ${candidate.ev:.2f} | Margin: {candidate.margin_pct:.1f}%")
