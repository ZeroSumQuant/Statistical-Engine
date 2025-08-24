import unittest
import pandas as pd
import numpy as np
from market_stats_engine import (
    cluster_pivots,
    test_zone_significance,
    ZoneConfig,
    InstrumentConfig,
)

class TestMarketStatsEngine(unittest.TestCase):

    def test_cluster_pivots(self):
        config = ZoneConfig(cluster_width_points=5.0, min_touches_for_significance=2)
        pivots = [
            {'price': 100, 'center_time': pd.Timestamp('2023-01-01 10:00')},
            {'price': 101, 'center_time': pd.Timestamp('2023-01-01 10:05')},
            {'price': 104, 'center_time': pd.Timestamp('2023-01-01 10:10')},
            {'price': 110, 'center_time': pd.Timestamp('2023-01-01 10:15')},
            {'price': 112, 'center_time': pd.Timestamp('2023-01-01 10:20')},
        ]
        clusters = cluster_pivots(pivots, config)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 3)
        self.assertEqual(len(clusters[1]), 2)
        self.assertEqual(clusters[0][0]['price'], 100)
        self.assertEqual(clusters[1][0]['price'], 110)

    def test_zone_significance(self):
        # Test case 1: Significant result
        touches = [100.5, 100.6, 99.4, 99.5]
        level = 100
        width = 1.0
        local_prices = pd.Series([90, 110])
        is_sig, p_val = test_zone_significance(touches, level, width, local_prices, alpha=0.05)
        self.assertTrue(is_sig)
        self.assertLess(p_val, 0.05)

        # Test case 2: Not significant
        touches = [95, 105, 96, 106]
        is_sig, p_val = test_zone_significance(touches, level, width, local_prices, alpha=0.05)
        self.assertFalse(is_sig)
        self.assertGreaterEqual(p_val, 0.05)

        # Test case 3: Not enough touches
        touches = [100.5]
        is_sig, p_val = test_zone_significance(touches, level, width, local_prices, alpha=0.05)
        self.assertFalse(is_sig)
        self.assertEqual(p_val, 1.0)


if __name__ == '__main__':
    unittest.main()
