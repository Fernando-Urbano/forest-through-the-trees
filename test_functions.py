# %%
from portfolio_construction.ridge_portfolios import *
from data.factors_data import *
from data.characteristics_data import *
from portfolio_construction.triple_sorting_portfolios import *

# %% ridge_portfolios.py
# Set the number of series and observations
num_series = 15
num_obs = 15000

# Set the mean and standard deviation for each series
mean = np.zeros(num_series)
std_dev = np.ones(num_series)

# Set the correlation matrix for the series
corr_matrix = np.random.uniform(-1, 1, size=(num_series, num_series))
corr_matrix = np.triu(corr_matrix, k=1) + np.triu(corr_matrix, k=1).T + np.eye(num_series)

# Generate the returns for each series
returns = np.random.multivariate_normal(mean, corr_matrix, size=num_obs)

returns_df = (
    pd.DataFrame(returns)
    .assign(date=lambda df: [datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x) for x in range(int(len(df.index)))])
    .set_index("date")
    .rename(columns=lambda x: ''.join(random.choices(string.ascii_lowercase, k=4)).upper())
)

# Get the best lambda by cross-validation
best_lambda_by_cv = pd.DataFrame(
    calculate_ts_cv(
        returns_df, return_all_lambdas=True,
        folds_by_end_date=["2050-01-01", "2055-01-01", "2060-01-01"]
    )
)

sns.set(style='whitegrid', palette='Spectral')
x = best_lambda_by_cv.index
ts_cvs = best_lambda_by_cv.rename(columns=lambda x: x.replace("ts_cv_fold", "TS CV Fold "))
fig, ax = plt.subplots(figsize=(10, 5))
for col in list(ts_cvs.columns):
    ax.plot(x, ts_cvs[col], linewidth = 2.5, label=col)
ax.set_xlabel('Lambda')
ax.set_ylabel('Sharpe')
ax.set_title(
    'Sharpe Out-of-Sample of Time Series Cross Validations by Lambda'
)
ax.text(
    1, -0.2,
    "",
    verticalalignment='bottom',
    horizontalalignment='right',
    transform=ax.transAxes
)
ax.legend()
plt.show()

# %% ridge_portfolios.py
# Calculate weights of the portfolio and force weights to be positive by raising lambda little by little
w_history = calculate_mv_portfolios_forcing_positive_weights(returns_df, 0, delta=5, minimum_delta=.01, record_w_history=True)
w_history = pd.DataFrame(w_history)

sns.set(style='whitegrid', palette='rocket')
x = w_history.index
fig, ax = plt.subplots(figsize=(10, 5))
for col in list(w_history.columns):
    ax.plot(x, w_history[col], linewidth = 2.5)
ax.set_xlabel('N of Iterations')
ax.set_ylabel('Weights')
ax.set_title(
    'Evolution of Weights with Lambda Increments/Decrements'
)
ax.text(
    1, -0.2,
    "",
    verticalalignment='bottom',
    horizontalalignment='right',
    transform=ax.transAxes
)


# %% ridge_portfolios.py
weights_per_lambda = {}
# How do weights change based on the value of the lambda
for chosen_lambda in list(np.arange(0, 4, 0.0025)):
    weights_per_lambda[chosen_lambda] = calculate_mv_portfolio(returns_df, chosen_lambda)

weights_per_lambda = pd.DataFrame(weights_per_lambda).transpose()

sns.set(style='whitegrid', palette='rocket')
x = weights_per_lambda.index
fig, ax = plt.subplots(figsize=(10, 5))
for col in list(weights_per_lambda.columns):
    ax.plot(x, weights_per_lambda[col], linewidth = 2.5)
ax.axhline(y=0, linestyle='--', color='black')
ax.set_xlabel('Weights')
ax.set_ylabel('Lambda')
ax.set_title(
    'All the Weights of a Portfolio by RIDGE Lambda'
)
ax.text(
    1, -0.2,
    "",
    verticalalignment='bottom',
    horizontalalignment='right',
    transform=ax.transAxes
)

# %% factors_data.py
returns_plot(factors_returns, start_date="2015-01-01")
returns_plot(factors_returns, start_date="1990-01-01")

# %% triple_sorting_portfolios.py
stock_sorting_groups = create_sorting(stock_characteristics, characteristics=["mom", "hml", "smb"])
sorting_groups_returns = create_sorting_groups_returns(stock_returns, stock_sorting_groups)
stock_characteristics = stock_characteristics.loc[lambda df: df.date == max(df.date)]
