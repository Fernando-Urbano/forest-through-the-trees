from data.characteristics_data import stock_characteristics, stock_returns
from data.factors_data import factors_returns
from portfolio_construction.triple_sorting_portfolios import (
    create_sorting,
    create_sorting_groups_returns,
    create_long_short_portfolios
)
from portfolio_construction.ap_tree_portfolios import create_ap_tree_sorting
from portfolio_construction.ridge_portfolios import (
    calculate_ts_cv,
    calculate_mv_portfolios_forcing_positive_weights,
    calculate_portfolio_statistics,
    calculate_mv_portfolio
)
from portfolio_construction.factors_statistics import (
    factor_regression,
    cross_section_factor_beta_regression
)
import math

import datetime
import pandas as pd
import numpy as np
import statistics
import sys
from dateutil.relativedelta import relativedelta
from sklearn.tree import DecisionTreeRegressor
from dateutil import rrule

def write_update(update, testing_start_date=None, testing_end_date=None):
    if testing_start_date and testing_end_date:
        update = testing_start_date.strftime("%Y-%m-%d") + " - " + testing_end_date.strftime("%Y-%m-%d")  + " - " + update
    sys.stdout.write(("\r" + datetime.datetime.now().strftime("%H:%M:%S") + ": " + update).ljust(100))

def test_period(
        stock_characteristics,
        stock_returns,
        factors_returns,
        training_start_date,
        training_end_date,
        validation_start_date,
        validation_end_date,
        testing_start_date,
        testing_end_date,
        characteristics,
        factors,
        sorting_quantiles,
        ap_tree_max_depth,
        ap_tree_min_samples_leaf
    ):
    for char in ['mom', 'hml', 'smb']:
        if char not in characteristics:
            stock_characteristics = stock_characteristics.drop(char, axis=1)
    for fac in ['mkt', 'mom', 'hml', 'smb']:
        if fac not in factors:
            fac = fac + "_minus_rf" if fac == 'mkt' else fac
            factors_returns = factors_returns.drop(fac, axis=1)
    
    # Characteristics
    stock_characteristics_training_final_date = stock_characteristics.loc[lambda df: df.date == training_end_date]
    stock_characteristics_validation_final_date = stock_characteristics.loc[lambda df: df.date == validation_end_date]
    # Stock returns
    stock_returns_training = stock_returns.loc[lambda df: (df.index >= training_start_date) & (df.index <= training_end_date)]
    stock_returns_validation = stock_returns.loc[lambda df: (df.index >= validation_start_date) & (df.index <= validation_end_date)]
    stock_returns_in_sample = pd.concat([stock_returns_training, stock_returns_validation])
    stock_returns_testing = stock_returns.loc[lambda df: (df.index >= testing_start_date) & (df.index <= testing_end_date)]
    write_update("Completed database imports.", testing_start_date, testing_end_date)

    # Triple sorting portfolios
    stock_sorting_groups = create_sorting(
        stock_characteristics_validation_final_date,
        characteristics=characteristics,
        num_quantiles=sorting_quantiles
    )
    sorting_groups_returns_in_sample = create_sorting_groups_returns(stock_returns_in_sample, stock_sorting_groups)
    sorting_groups_returns_testing = create_sorting_groups_returns(stock_returns_testing, stock_sorting_groups)
    long_short_portfolios_returns_testing = create_long_short_portfolios(sorting_groups_returns_testing, max_quantile=sorting_quantiles)
    write_update("Completed creation of Triple Sorting portfolios.", testing_start_date, testing_end_date)

    # Triple sorting sharpe-ratio out-of-sample
    best_lambda_cv = calculate_ts_cv(sorting_groups_returns_in_sample, folds_by_end_date=training_end_date, optimize_calculation=False)
    w = calculate_mv_portfolio(sorting_groups_returns_in_sample, best_lambda_cv["ts_cv_fold0"])
    triple_sorting_optimized_portfolio_returns_testing = calculate_portfolio_statistics(
        w, sorting_groups_returns_testing, # .loc[lambda df: df.index < df.index.min() + relativedelta(months=12)],
        return_ts=True
    )
    triple_sorting_optimized_portfolio_sharpe_testing = (
        np.mean(triple_sorting_optimized_portfolio_returns_testing) / np.std(triple_sorting_optimized_portfolio_returns_testing)
    )

    # AP-Trees portfolios
    # Training sample
    characteristics_table = (
        stock_characteristics_training_final_date
        .rename({"rt_m12": "expected_return"}, axis=1)
        .drop(["date", "rt_m1"], axis=1)
        .set_index("stock_id")
    )
    X = characteristics_table.drop('expected_return', axis=1)
    y = characteristics_table['expected_return']
    tree = DecisionTreeRegressor(max_depth=ap_tree_max_depth, min_samples_leaf=ap_tree_min_samples_leaf)
    tree.fit(X, y)
    ap_tree_groups_returns_training = create_ap_tree_sorting(tree, characteristics_table)
    portfolios_best_lambdas = {}
    for portfolio_name, portfolio_stocks in ap_tree_groups_returns_training.items():
        portfolio_stock_returns = (
            pd.concat([stock_returns_training, stock_returns_validation])
            .loc[:, portfolio_stocks]
        )
        portfolio_best_lambda = calculate_ts_cv(
            portfolio_stock_returns,
            return_all_lambdas=False,
            folds_by_end_date=training_end_date, 
            optimize_calculation=False
        )
        portfolios_best_lambdas[portfolio_name] = portfolio_best_lambda["ts_cv_fold0"]
    best_global_lambda = statistics.mean(list(portfolios_best_lambdas.values()))
    write_update("Completed cross-validation to find best lambda.", testing_start_date, testing_end_date)

    # Training + validation sample
    characteristics_table = (
        stock_characteristics_validation_final_date
        .rename({"rt_m12": "expected_return"}, axis=1)
        .drop(["date", "rt_m1"], axis=1)
        .set_index("stock_id")
    )
    X = characteristics_table.drop('expected_return', axis=1)
    y = characteristics_table['expected_return']
    tree = DecisionTreeRegressor(max_depth=ap_tree_max_depth, min_samples_leaf=ap_tree_min_samples_leaf)
    tree.fit(X, y)
    ap_tree_groups_returns_training_validation = create_ap_tree_sorting(tree, characteristics_table)
    ap_tree_portfolios_returns_in_sample = pd.DataFrame({})
    ap_tree_portfolios_returns_testing = pd.DataFrame({})
    for portfolio_name, portfolio_stocks in ap_tree_groups_returns_training_validation.items():
        portfolio_stock_returns_in_sample = stock_returns_in_sample.loc[:, portfolio_stocks]
        portfolio_stock_returns_testing = stock_returns_testing.loc[:, portfolio_stocks]
        mv_optimal_weights_in_sample = calculate_mv_portfolios_forcing_positive_weights(
            portfolio_stock_returns_in_sample,
            lambda_reg=best_global_lambda,
        )
        portfolio_returns_in_sample = calculate_portfolio_statistics(
            mv_optimal_weights_in_sample, portfolio_stock_returns_in_sample,
            return_ts=True
        )
        portfolio_returns_testing = calculate_portfolio_statistics(
            mv_optimal_weights_in_sample, portfolio_stock_returns_testing,
            return_ts=True
        )
        ap_tree_portfolios_returns_in_sample = (
            ap_tree_portfolios_returns_in_sample
            .merge(
                pd.DataFrame(portfolio_returns_in_sample).set_axis(["ap_tree_port_" + portfolio_name], axis=1),
                left_index=True, right_index=True, how='outer'
            )
        )
        ap_tree_portfolios_returns_testing = (
            ap_tree_portfolios_returns_testing
            .merge(
                pd.DataFrame(portfolio_returns_testing).set_axis(["ap_tree_port_" + portfolio_name], axis=1),
                left_index=True, right_index=True, how='outer'
            )
        )
    write_update("Completed creation of AP-Tree portfolios", testing_start_date, testing_end_date)

    # AP-Tree sharpe-ratio out-of-sample
    best_lambda_cv = calculate_ts_cv(ap_tree_portfolios_returns_in_sample, folds_by_end_date=training_end_date, optimize_calculation=False)
    w = calculate_mv_portfolios_forcing_positive_weights(ap_tree_portfolios_returns_in_sample, best_lambda_cv["ts_cv_fold0"])
    ap_tree_optimized_portfolio_returns_testing = calculate_portfolio_statistics(
        w, ap_tree_portfolios_returns_testing, # .loc[lambda df: df.index < df.index.min() + relativedelta(months=12)],
        return_ts=True
    )
    ap_tree_optimized_portfolio_sharpe_testing = (
        np.mean(ap_tree_optimized_portfolio_returns_testing) / np.std(ap_tree_optimized_portfolio_returns_testing)
    )

    # Time-series regression
    ap_tree_time_series_factor_regressions = factor_regression(ap_tree_portfolios_returns_testing, factors_returns)
    triple_sorting_time_series_factor_regressions = factor_regression(long_short_portfolios_returns_testing, factors_returns)
    write_update("Completed time-series regression", testing_start_date, testing_end_date)

    # Cross-section regression
    sml_ap_tree_lines = cross_section_factor_beta_regression(
        ap_tree_time_series_factor_regressions,
        ap_tree_portfolios_returns_testing
    )
    sml_triple_sorting_lines = cross_section_factor_beta_regression(
        triple_sorting_time_series_factor_regressions, 
        long_short_portfolios_returns_testing
    )
    sml_ap_tree_vs_triple_sorting = (
        pd.concat([
            sml_ap_tree_lines.assign(portfolio="AP-Tree Portfolio").assign(sharpe=ap_tree_optimized_portfolio_sharpe_testing),
            sml_triple_sorting_lines.assign(portfolio="Triple Sorting Portfolio").assign(sharpe=triple_sorting_optimized_portfolio_sharpe_testing),
        ])
        .assign(start_date=testing_start_date, end_date=testing_end_date)
    )
    write_update("Completed cross-section regression", testing_start_date, testing_end_date)

    return sml_ap_tree_vs_triple_sorting

def test_multiple_periods(
    months_testing_period,
    characteristics,
    factors,
    sorting_quantiles,
    ap_tree_max_depth,
    ap_tree_min_samples_leaf,
    first_testing_start_date = datetime.date(2009, 1, 1),
    last_testing_end_date = datetime.date(2018, 1, 1),
):
    testing_start_dates = list(
        rrule.rrule(
            freq=rrule.MONTHLY,
            interval=6,
            dtstart=first_testing_start_date,
            until=last_testing_end_date - relativedelta(months=months_testing_period)
        )
    )
    testing_start_dates = [d.date() for d in list(testing_start_dates)]
    testing_end_dates = [d + relativedelta(months=months_testing_period) for d in testing_start_dates]
    all_sml_ap_tree_vs_triple_sorting = pd.DataFrame({})
    for testing_start_date, testing_end_date in zip(testing_start_dates, testing_end_dates):
        new_sml_ap_tree_vs_triple_sorting = test_period(
            stock_characteristics,
            stock_returns,
            factors_returns,
            training_start_date=stock_returns.index.min(),
            training_end_date=testing_start_date - relativedelta(years=2, months=1),
            validation_start_date=testing_start_date - relativedelta(years=2),
            validation_end_date=testing_start_date - relativedelta(months=1),
            testing_start_date=testing_start_date,
            testing_end_date=testing_end_date,
            characteristics=characteristics,
            factors=factors,
            sorting_quantiles=sorting_quantiles,
            ap_tree_max_depth=ap_tree_max_depth,
            ap_tree_min_samples_leaf=ap_tree_min_samples_leaf
        )
        all_sml_ap_tree_vs_triple_sorting = pd.concat([
            all_sml_ap_tree_vs_triple_sorting,
            new_sml_ap_tree_vs_triple_sorting
        ])
        write_update(f"Completed {testing_start_date.strftime('%Y-%m-%d')} - {testing_end_date.strftime('%Y-%m-%d')}")
    file_name = [
        "sml",
        str(months_testing_period) + "_months_testing_period",
        "_".join(characteristics) + "_characteristics",
        "_".join(factors) + "_factors",
        str(sorting_quantiles) + "_sorting_quantiles",
        str(ap_tree_max_depth) + "_ap_tree_max_depth",
        str(ap_tree_min_samples_leaf) + "_ap_tree_min_samples_leaf"
    ]
    all_sml_ap_tree_vs_triple_sorting.to_csv("results/data/" + "_".join(file_name) + ".csv", index=False)