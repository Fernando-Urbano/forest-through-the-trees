from implement_experiment_functions import *

for max_depth in [8, 11, 14]:
    for min_sample in [5]:
        # Fama-French Carhart
        test_multiple_periods(
            months_testing_period=36,
            characteristics=['hml', 'smb', 'mom'],
            factors=['mkt', 'hml', 'smb', 'mom'],
            sorting_quantiles=3,
            ap_tree_max_depth=max_depth,
            ap_tree_min_samples_leaf=min_sample
        )
        # CAPM
        test_multiple_periods(
            months_testing_period=36,
            characteristics=['hml', 'smb', 'mom'],
            factors=['mkt'],
            sorting_quantiles=3,
            ap_tree_max_depth=max_depth,
            ap_tree_min_samples_leaf=min_sample
        )
        test_multiple_periods(
            months_testing_period=36,
            characteristics=['hml', 'smb'],
            factors=['mkt'],
            sorting_quantiles=4,
            ap_tree_max_depth=max_depth,
            ap_tree_min_samples_leaf=min_sample
        )
        # Fama-French
        test_multiple_periods(
            months_testing_period=36,
            characteristics=['hml', 'smb'],
            factors=['mkt', 'hml', 'smb'],
            sorting_quantiles=4,
            ap_tree_max_depth=max_depth,
            ap_tree_min_samples_leaf=min_sample
        )
        test_multiple_periods(
            months_testing_period=36,
            characteristics=['hml', 'smb', 'mom'],
            factors=['mkt', 'hml', 'smb'],
            sorting_quantiles=3,
            ap_tree_max_depth=max_depth,
            ap_tree_min_samples_leaf=min_sample
        )