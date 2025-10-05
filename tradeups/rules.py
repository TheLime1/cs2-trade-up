"""
Trade-up rules engine for CS2 calculator.

Handles rarity progression, target enumeration, and trade-up validation rules.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from .price_model import NormalizedItem, RarityMapper

logger = logging.getLogger(__name__)


class TradeUpRules:
    """Defines and validates CS2 trade-up rules."""

    TRADE_UP_INPUT_COUNT = 10

    @classmethod
    def get_next_rarity_rank(cls, current_rank: int) -> Optional[int]:
        """
        Get the next higher rarity rank.

        Args:
            current_rank: Current rarity rank

        Returns:
            Next rarity rank or None if already at highest
        """
        max_rank = len(RarityMapper.RARITY_ORDER) - 1
        if current_rank >= max_rank:
            return None
        return current_rank + 1

    @classmethod
    def get_next_rarity_name(cls, current_rarity: str) -> Optional[str]:
        """
        Get the next higher rarity name.

        Args:
            current_rarity: Current rarity name

        Returns:
            Next rarity name or None if already at highest
        """
        current_rank = RarityMapper.get_rarity_rank(current_rarity)
        next_rank = cls.get_next_rarity_rank(current_rank)

        if next_rank is None:
            return None

        return RarityMapper.RARITY_ORDER[next_rank]

    @classmethod
    def validate_trade_up_inputs(cls, inputs: List[NormalizedItem]) -> Tuple[bool, str]:
        """
        Validate that inputs are valid for a trade-up.

        Args:
            inputs: List of input items

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(inputs) != cls.TRADE_UP_INPUT_COUNT:
            return False, f"Trade-up requires exactly {cls.TRADE_UP_INPUT_COUNT} inputs, got {len(inputs)}"

        # Check that all inputs have the same rarity
        if not inputs:
            return False, "No inputs provided"

        first_rarity_rank = inputs[0].rarity_rank
        for item in inputs:
            if item.rarity_rank != first_rarity_rank:
                return False, f"All inputs must have the same rarity. Found {item.rarity} and {inputs[0].rarity}"

        # Check StatTrak consistency
        first_stattrak = inputs[0].stattrak
        for item in inputs:
            if item.stattrak != first_stattrak:
                return False, "All inputs must be either StatTrak or non-StatTrak"

        # Check that all inputs are tradeable
        for item in inputs:
            if not item.is_tradeable:
                return False, f"Item '{item.display_name}' is not tradeable (price: {item.price}, collection: {item.collection})"

        return True, "Valid trade-up inputs"

    @classmethod
    def can_trade_up_rarity(cls, rarity_rank: int) -> bool:
        """Check if items of this rarity can be traded up."""
        next_rank = cls.get_next_rarity_rank(rarity_rank)
        return next_rank is not None


class TargetEnumerator:
    """Enumerates possible trade-up targets."""

    def __init__(self, items_by_collection: Dict[str, List[NormalizedItem]]):
        """
        Initialize with items grouped by collection.

        Args:
            items_by_collection: Dict mapping collection name to list of items
        """
        self.items_by_collection = items_by_collection

        # Build indices for faster lookups
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for performance."""
        # Index: collection -> rarity_rank -> items
        self.collection_rarity_items: Dict[str,
                                           Dict[int, List[NormalizedItem]]] = {}

        # Index: collection -> available rarity ranks
        self.collection_rarities: Dict[str, Set[int]] = {}

        for collection, items in self.items_by_collection.items():
            if collection not in self.collection_rarity_items:
                self.collection_rarity_items[collection] = {}
                self.collection_rarities[collection] = set()

            for item in items:
                rarity_rank = item.rarity_rank

                if rarity_rank not in self.collection_rarity_items[collection]:
                    self.collection_rarity_items[collection][rarity_rank] = []

                self.collection_rarity_items[collection][rarity_rank].append(
                    item)
                self.collection_rarities[collection].add(rarity_rank)

    def get_targets_for_collection(
        self,
        collection: str,
        input_rarity_rank: int,
        stattrak: bool
    ) -> List[NormalizedItem]:
        """
        Get all possible targets for a collection and input rarity.

        Args:
            collection: Collection name
            input_rarity_rank: Input items rarity rank
            stattrak: Whether inputs are StatTrak

        Returns:
            List of possible target items
        """
        targets = []

        if collection not in self.collection_rarity_items:
            logger.warning(
                f"Collection '{collection}' not found in items database")
            return targets

        # Get next rarity rank
        next_rarity_rank = TradeUpRules.get_next_rarity_rank(input_rarity_rank)
        if next_rarity_rank is None:
            return targets

        # Get items of next rarity in this collection
        collection_items = self.collection_rarity_items[collection]
        if next_rarity_rank not in collection_items:
            return targets

        # Filter by StatTrak requirement and tradeability
        for item in collection_items[next_rarity_rank]:
            if item.stattrak == stattrak and item.is_tradeable:
                targets.append(item)

        return targets

    def get_all_targets(
        self,
        input_collections: List[str],
        input_rarity_rank: int,
        stattrak: bool
    ) -> Dict[str, List[NormalizedItem]]:
        """
        Get all possible targets grouped by collection.

        Args:
            input_collections: List of input collections
            input_rarity_rank: Input items rarity rank  
            stattrak: Whether inputs are StatTrak

        Returns:
            Dict mapping collection to list of target items
        """
        all_targets = {}

        for collection in input_collections:
            targets = self.get_targets_for_collection(
                collection, input_rarity_rank, stattrak)
            if targets:
                all_targets[collection] = targets

        return all_targets

    def has_valid_targets(self, collection: str, input_rarity_rank: int) -> bool:
        """
        Check if a collection has any valid targets for trade-up.

        Args:
            collection: Collection name
            input_rarity_rank: Input items rarity rank

        Returns:
            True if collection has valid targets
        """
        if collection not in self.collection_rarities:
            return False

        next_rarity_rank = TradeUpRules.get_next_rarity_rank(input_rarity_rank)
        if next_rarity_rank is None:
            return False

        return next_rarity_rank in self.collection_rarities[collection]


class TradeUpCalculator:
    """Calculates trade-up probabilities and outcomes."""

    def __init__(self, target_enumerator: TargetEnumerator):
        """
        Initialize with a target enumerator.

        Args:
            target_enumerator: TargetEnumerator instance
        """
        self.target_enumerator = target_enumerator

    def calculate_probabilities(
        self,
        inputs: List[NormalizedItem]
    ) -> Dict[str, float]:
        """
        Calculate collection selection probabilities.

        Args:
            inputs: List of input items (must be exactly 10)

        Returns:
            Dict mapping collection to probability
        """
        # Validate inputs
        is_valid, error = TradeUpRules.validate_trade_up_inputs(inputs)
        if not is_valid:
            raise ValueError(f"Invalid trade-up inputs: {error}")

        # Count items by collection
        collection_counts = {}
        for item in inputs:
            if item.collection:
                collection_counts[item.collection] = collection_counts.get(
                    item.collection, 0) + 1

        # Convert counts to probabilities
        total = sum(collection_counts.values())
        if total == 0:
            return {}

        probabilities = {}
        for collection, count in collection_counts.items():
            probabilities[collection] = count / total

        return probabilities

    def calculate_outcomes(
        self,
        inputs: List[NormalizedItem]
    ) -> Tuple[List[Dict], float, float]:
        """
        Calculate detailed trade-up outcomes.

        Args:
            inputs: List of input items (must be exactly 10)

        Returns:
            Tuple of (outcomes_list, success_probability, dead_probability)
        """
        # Validate inputs
        is_valid, error = TradeUpRules.validate_trade_up_inputs(inputs)
        if not is_valid:
            raise ValueError(f"Invalid trade-up inputs: {error}")

        # Get collection probabilities
        collection_probs = self.calculate_probabilities(inputs)

        # Get all possible targets
        input_rarity_rank = inputs[0].rarity_rank
        stattrak = inputs[0].stattrak

        all_targets = self.target_enumerator.get_all_targets(
            list(collection_probs.keys()),
            input_rarity_rank,
            stattrak
        )

        outcomes = []
        success_probability = 0.0

        for collection, collection_prob in collection_probs.items():
            if collection in all_targets:
                targets = all_targets[collection]

                if targets:
                    # Uniform distribution among targets in this collection
                    target_prob = collection_prob / len(targets)
                    success_probability += collection_prob

                    for target in targets:
                        outcomes.append({
                            'item': target,
                            'collection': collection,
                            'probability': target_prob,
                            'name': target.display_name,
                            'price': target.price
                        })

        dead_probability = 1.0 - success_probability

        return outcomes, success_probability, dead_probability

    def calculate_expected_value(
        self,
        inputs: List[NormalizedItem],
        fee_rate: float = 0.15
    ) -> Tuple[float, float, float]:
        """
        Calculate expected value and related metrics.

        Args:
            inputs: List of input items
            fee_rate: Market fee rate (default 15%)

        Returns:
            Tuple of (expected_value, total_input_cost, profit_margin_pct)
        """
        outcomes, success_prob, dead_prob = self.calculate_outcomes(inputs)

        # Calculate total input cost
        total_input_cost = sum(item.price for item in inputs)

        # Calculate expected value
        expected_value = 0.0
        for outcome in outcomes:
            net_price = outcome['price'] * (1 - fee_rate)
            expected_value += outcome['probability'] * net_price

        # Subtract input cost to get net expected value
        net_expected_value = expected_value - total_input_cost

        # Calculate profit margin percentage
        if total_input_cost > 0:
            profit_margin_pct = (net_expected_value / total_input_cost) * 100
        else:
            profit_margin_pct = 0.0

        return net_expected_value, total_input_cost, profit_margin_pct


def build_target_enumerator(items: List[NormalizedItem]) -> TargetEnumerator:
    """
    Build a TargetEnumerator from a list of items.

    Args:
        items: List of normalized items

    Returns:
        TargetEnumerator instance
    """
    items_by_collection = {}

    for item in items:
        if item.collection and item.is_tradeable:
            if item.collection not in items_by_collection:
                items_by_collection[item.collection] = []
            items_by_collection[item.collection].append(item)

    logger.info(
        f"Built target enumerator with {len(items_by_collection)} collections")

    return TargetEnumerator(items_by_collection)


if __name__ == "__main__":
    # Test the rules engine
    logging.basicConfig(level=logging.INFO)

    # Test rarity progression
    print("Testing rarity progression:")
    for i, rarity in enumerate(RarityMapper.RARITY_ORDER):
        next_rank = TradeUpRules.get_next_rarity_rank(i)
        if next_rank is not None:
            next_rarity = RarityMapper.RARITY_ORDER[next_rank]
            print(f"  {rarity} (rank {i}) -> {next_rarity} (rank {next_rank})")
        else:
            print(f"  {rarity} (rank {i}) -> Cannot trade up (highest rarity)")

    # Test trade-up validation
    print("\nTesting trade-up validation:")

    # Create mock items for testing
    from .price_model import NormalizedItem

    mock_items = [
        NormalizedItem(
            id=f"test_{i}",
            market_hash_name=f"Test Item {i}",
            weapon="AK-47",
            skin="Test",
            stattrak=False,
            collection="Test Collection",
            rarity="Mil-Spec Grade",
            rarity_rank=2,
            price=10.0
        ) for i in range(10)
    ]

    is_valid, message = TradeUpRules.validate_trade_up_inputs(mock_items)
    print(f"  10 matching items: {is_valid} - {message}")

    # Test with wrong count
    is_valid, message = TradeUpRules.validate_trade_up_inputs(mock_items[:5])
    print(f"  5 items: {is_valid} - {message}")

    # Test with mismatched rarity
    mock_items[0].rarity_rank = 1
    is_valid, message = TradeUpRules.validate_trade_up_inputs(mock_items)
    print(f"  Mismatched rarity: {is_valid} - {message}")
