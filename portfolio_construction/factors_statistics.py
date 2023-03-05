import numpy as np
import pandas as pd
import statsmodels.api as sm

def factor_regression(returns_table, factors_returns, return_pvalues=False):
    if "date" in returns_table.columns:
        returns_table.set_index("date", inplace=True)
    if "date" in factors_returns.columns:
        factors_returns.set_index("date", inplace=True)
    if "rf" in factors_returns.columns:
        factors_returns.drop("rf", axis=1, inplace=True)
    common_dates = [d for d in returns_table.index if d in factors_returns.index]
    factors_returns = factors_returns.loc[lambda df: df.index.isin(common_dates)]
    returns_table = returns_table.loc[lambda df: df.index.isin(common_dates)]
    regression_parameters = {}
    regression_pvalues = {}
    portfolios = list(returns_table.columns)
    for portfolio in portfolios:
        X = factors_returns
        X = sm.add_constant(X)
        y = returns_table[portfolio]
        model = sm.OLS(y, X).fit()
        dir(model)
        regression_parameters[portfolio] = model.params.to_dict()
        regression_pvalues[portfolio] = model.pvalues.to_dict()
    if return_pvalues:
        return regression_parameters, regression_pvalues
    else:
        return regression_parameters
    
def cross_section_factor_beta_regression(factor_betas, returns_table):
    factor_betas = pd.DataFrame(factor_betas).transpose()
    annual_returns_table = (
        returns_table
        .apply(lambda x: ((x + 1).cumprod()) ** (12 / (len(x) - 1)) - 1)
        .loc[lambda df: df.index == df.index.max()]
        .transpose()
        .set_axis(["annual_return"], axis=1)
    )
    betas_vs_returns_table = factor_betas.merge(annual_returns_table, left_index=True, right_index=True)
    X = betas_vs_returns_table.drop(["const", "annual_return"], axis=1)
    X = sm.add_constant(X)
    y = betas_vs_returns_table["annual_return"]
    model = sm.OLS(y, X).fit()
    model_results = (
        pd.concat([model.params, model.pvalues, model.tvalues], axis=1)
        .set_axis(["value", "pvalue", "tvalue"], axis=1)
        .rename({"const": "alpha"}, axis=0)
        .rename(lambda x: x.replace(" ", ""), axis=0)
        .reset_index()
        .rename({"index": "parameter"}, axis=1)
        .assign(
            start_date=returns_table.index.min(),
            end_date=returns_table.index.max(),
            rsquared=model.rsquared
        )
    )
    return model_results