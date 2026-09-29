import unittest
from dynamic_trading_engine.state import PortfolioState
from dynamic_trading_engine.optimization import equal_weight
from dynamic_trading_engine.policies import target_orders
from dynamic_trading_engine.execution import fill_market

class TradingTests(unittest.TestCase):
    def test_target_and_fill(self):
        state = PortfolioState(100, {})
        weights = equal_weight(["A", "B"])
        orders = target_orders(state, {"A": 10, "B": 20}, weights)
        self.assertEqual([order.units for order in orders], [5, 2.5])
        self.assertGreater(fill_market(orders[0], 10, 0.02).price, 10)
