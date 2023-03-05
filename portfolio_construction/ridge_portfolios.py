import numpy as np
import datetime
import pandas as pd
import random
import string
import sys

def calculate_mv_portfolio(returns_matrix, lambda_reg):
    """
    Calculate minimum variance portfolio
    """

    # Compute the covariance matrix of the returns matrix
    cov_matrix = np.cov(returns_matrix, rowvar=False)

    # Add the regularization parameter to the diagonal of the covariance matrix
    reg_matrix = cov_matrix + lambda_reg * np.identity(cov_matrix.shape[0])

    # Calculate the weights for the minimum variance portfolio
    e = np.ones(cov_matrix.shape[0])
    w_unnormalized = np.linalg.inv(reg_matrix) @ e
    w = w_unnormalized / np.sum(w_unnormalized)

    return w


def calculate_mv_portfolios_forcing_positive_weights(returns_matrix, lambda_reg, delta=.2, minimum_delta=.001, record_w_history=False):
    w_history = []
    w = calculate_mv_portfolio(returns_matrix, lambda_reg)
    w_history.append(w)
    if all(w >= 0):
        return w
    while delta > minimum_delta:
        if any(w < 0):
            while any(w < 0):
                lambda_reg += delta
                w = calculate_mv_portfolio(returns_matrix, lambda_reg)
                w_history.append(w)
        else:
            while all(w >= 0):
                lambda_reg -= delta
                w = calculate_mv_portfolio(returns_matrix, lambda_reg)
                w_history.append(w)
        delta /= 2
        lambda_reg -= delta
    if record_w_history:
        return w_history
    else:
        return w


def calculate_portfolio_statistics(w, returns_matrix, return_ts=False):
    """
    Calculate portfolio's average return and variance
    """
    compound = False
    if compound:
        # Calculate the portfolio return as the weighted sum of the compounded returns
        portfolio_compounded_return = np.prod(returns_matrix @ w + 1)
        portfolio_compounded_return **= (1 / len(returns_matrix))
    else:
        portfolio_compounded_return = np.mean(returns_matrix @ w) + 1

    # Calculate the average return and variance of the portfolio
    avg_return = portfolio_compounded_return - 1
    portfolio_return = returns_matrix @ w
    variance = np.var(portfolio_return)

    if return_ts:
        return portfolio_return
    else:
        return avg_return, variance


def test_lambdas(
    in_sample_returns,
    out_of_sample_returns,
    lambda_values=list(np.arange(0, 0.601, 0.0002)),
    return_all_lambdas=False,
    sharpe_criterion=True,
    force_positive_weights=False,
    optimize_calculation=False,
    delta=.2,
    minimum_delta=.00002
):
    """
    Test best lambda regularization with out-of-sample returns
    """
    best_lambda = .5
    best_variance, best_sharpe = float('inf'), float('-inf')
    all_lambdas = {}

    if optimize_calculation:
        raise NotImplementedError("'optimize_calculation' not implemented.")
        w = calculate_mv_portfolio(in_sample_returns, best_lambda)
        avg_return, variance = calculate_portfolio_statistics(w, out_of_sample_returns)
        best_sharpe = avg_return / variance
        addition = True
        while delta > minimum_delta:
            addition = False if addition else True
            while True:
                if not addition and best_lambda - delta < 0:
                    break
                lambda_value = best_lambda + delta if addition else best_lambda - delta
                w = calculate_mv_portfolio(in_sample_returns, lambda_value)
                avg_return, variance = calculate_portfolio_statistics(w, out_of_sample_returns)
                sharpe = avg_return / variance
                if sharpe >= best_sharpe:
                    best_sharpe = sharpe
                    best_lambda = lambda_value
                else:
                    break
            delta /= 2

    else:
        for lambda_value in lambda_values:
            w = calculate_mv_portfolio(in_sample_returns, lambda_value)
            avg_return, variance = calculate_portfolio_statistics(w, out_of_sample_returns)
            if sharpe_criterion:
                sharpe = avg_return / variance
                all_lambdas[lambda_value] = sharpe
                if sharpe > best_sharpe:
                    if force_positive_weights and any(w < 0):
                        continue
                    best_sharpe = sharpe
                    best_lambda = lambda_value
            else:
                all_lambdas[lambda_value] = variance
                if variance < best_variance:
                    if force_positive_weights and any(w < 0):
                        continue
                    best_variance = variance
                    best_lambda = lambda_value

    if return_all_lambdas and not optimize_calculation:
        return all_lambdas
    else:
        return best_lambda
    

def calculate_ts_cv(
        returns_table,
        n_folds=5,
        folds_by_end_date=None, 
        lambda_values=None,
        return_all_lambdas=None,
        optimize_calculation=False
    ):
    """
    Calculate time-series cross-validation to find the best lambda.
    """
    cv_dict = {}
    start = 0
    if folds_by_end_date is None:
        for i in range(n_folds):
            n_obs = returns_table.shape[0]
            fold_size = n_obs // n_folds
            extra_obs = n_obs % n_folds
            extra = 1 if extra_obs > 0 else 0
            extra_obs -= 1
            end = start + fold_size + extra
            cv_dict[f'returns_matrix_cv_fold{i}'] = returns_table[start:end]
            start = end + 1
    else:
        if not isinstance(folds_by_end_date, list):
            folds_by_end_date = [folds_by_end_date]
        folds_by_end_date = [datetime.datetime.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d for d in folds_by_end_date]
        if max(returns_table.index) not in folds_by_end_date:
            folds_by_end_date += [max(returns_table.index)]
        remaining_returns = returns_table
        i = 0
        for end_date in folds_by_end_date:
            cv_dict[f'returns_matrix_cv_fold{i}'] = remaining_returns.loc[lambda df: df.index <= end_date].values
            remaining_returns = remaining_returns.loc[lambda df: df.index > end_date]
            i += 1

    cv_best_lambdas = {}

    for i in range(len(cv_dict) - 1):
        in_sample_returns = cv_dict[f'returns_matrix_cv_fold{i}']
        out_of_sample_returns = cv_dict[f'returns_matrix_cv_fold{i + 1}']
        test_lambda_parameters = {
            'in_sample_returns': in_sample_returns,
            'out_of_sample_returns': out_of_sample_returns,
        }
        if lambda_values is not None:
            test_lambda_parameters["lambda_values"] = lambda_values
        if return_all_lambdas is not None:
            test_lambda_parameters["return_all_lambdas"] = return_all_lambdas
        cv_best_lambdas[f'ts_cv_fold{i}'] = test_lambdas(**test_lambda_parameters, optimize_calculation=optimize_calculation)

    return cv_best_lambdas