import re
import pandas as pd
import numpy as np
import itertools

from data.characteristics_data import stock_characteristics


def add_quantile_group(df, col_name, num_quantiles):
    ordered_df = df.sort_values(col_name, ascending=True)
    separated_dfs = np.array_split(ordered_df, num_quantiles)
    for i, sep_df in enumerate(separated_dfs):
        sep_df[col_name] = i + 1
    return pd.concat(separated_dfs)


def create_sorting(stock_characteristics, characteristics, num_quantiles=3):
    if "stock_id" in list(stock_characteristics.columns):
        stock_characteristics = stock_characteristics.set_index("stock_id")
    for char in characteristics:
        stock_characteristics = add_quantile_group(stock_characteristics, char, num_quantiles)
    group_combinations = {}
    for char in characteristics:
        group_combinations[char] = list(range(1, num_quantiles + 1))
    char_combinations = list(itertools.product(*group_combinations.values()))
    char_combinations = [{k: v[i] for i, k in enumerate(group_combinations.keys())} for v in char_combinations]
    stock_sorting_groups = {}
    for char_comb in char_combinations:
        stock_sorting_group = stock_characteristics.loc[lambda df: (df[list(char_comb)] == pd.Series(char_comb)).all(axis=1)]
        char_comb = "_".join([f'{k}{str(v)}' for k, v in char_comb.items()])
        stock_sorting_groups[char_comb] = list(stock_sorting_group.index)
    return stock_sorting_groups


def create_sorting_groups_returns(returns_table, stock_sorting_groups):
    sorting_groups_returns = pd.DataFrame({})
    for sorting_group, stock_ids in stock_sorting_groups.items():
        group_returns = returns_table.loc[:, stock_ids]
        group_returns = group_returns.apply(lambda x: x / len(group_returns.columns))
        group_returns = pd.DataFrame(group_returns.sum(axis=1)).set_axis([sorting_group], axis=1)
        sorting_groups_returns = sorting_groups_returns.merge(group_returns, left_index=True, right_index=True, how='outer')
    return pd.DataFrame(sorting_groups_returns)


def create_long_short_portfolios(sorting_groups_returns, min_quantile=1, max_quantile=3):
    groups = list(sorting_groups_returns.columns)
    quantiles_inversion = {
        str(k): str(v) for k, v in zip(
            range(min_quantile, max_quantile + 1), reversed(range(min_quantile, max_quantile + 1))
        ) if k != v
    }
    long_short_portfolios = pd.DataFrame({})
    for group in groups:
        inverse_group = group.translate(str.maketrans(quantiles_inversion))
        if inverse_group == group:
            continue
        new_ls_portfolio = (
            sorting_groups_returns
            .loc[:, [group, inverse_group]]
            .set_axis(["long", "short"], axis=1)
            .assign(long_short=lambda df: df.long * 2 - df.short)
            .drop(["long", "short"], axis=1)
            .rename({"long_short": group + "_minus_" + inverse_group}, axis=1)
        )
        long_short_portfolios = (
            long_short_portfolios
            .merge(new_ls_portfolio, left_index=True, right_index=True, how='outer')
        )
    return long_short_portfolios