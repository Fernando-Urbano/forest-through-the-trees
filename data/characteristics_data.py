import pandas as pd
import os
import datetime
from data.factors_data import factors_returns

risk_free_rate = (
    factors_returns
    .loc[:, ['rf']]
    .reset_index()
    .sort_values('date')
    .rename({"rf": "rf_m1"}, axis=1)
    .assign(rf_m12=lambda df: (1 + df['rf_m1']).rolling(12).apply(lambda x: x.prod()) - 1)
)

all_characteristics = pd.read_csv("data/stock_characteristics_and_returns.csv")

def transform_to_date(date_str):
    if int(date_str[6:]) < 25:
        return datetime.date(2000 + int(date_str[6:]), int(date_str[:2]), 1)
    else:
        return datetime.date(1900 + int(date_str[6:]), int(date_str[:2]), 1)

chosen_characteristics = (
    all_characteristics
    .loc[:, ["stock_id", "date", "Mom_11M_Usd", "Pb", "Mkt_Cap_12M_Usd", "R1M_Usd", "R12M_Usd"]]
    .rename(
        {
            "Mom_11M_Usd": "mom_ranking", "Pb": "hml_ranking", "Mkt_Cap_12M_Usd": "smb_ranking",
            "R1M_Usd": "rt_m1", "R12M_Usd": "rt_m12"
        },
        axis=1
    )
    .assign(date=lambda df: df.date.map(lambda x: transform_to_date(x)))
    .sort_values(["stock_id", "date"])
    .assign(
        rt_m1=lambda df: df.groupby("stock_id")["rt_m1"].transform(lambda x: x.shift(1)),
        rt_m12=lambda df: df.groupby("stock_id")["rt_m12"].transform(lambda x: x.shift(12)),
    )
    .merge(risk_free_rate, on='date', how='left')
    .assign(
        rt_m1=lambda df: (1 + df.rt_m1) / (1 + df.rf_m1) - 1,
        rt_m12=lambda df: (1 + df.rt_m12) / (1 + df.rf_m12) - 1,
    )
    .drop(["rf_m1", "rf_m12"], axis=1)
)

start_date = datetime.date(2001, 1, 1)
end_date = datetime.date(2018, 1, 1)

stock_returns = (
    chosen_characteristics
    .pivot(index="date", columns="stock_id", values="rt_m1")
    .loc[lambda df: df.index >= start_date]
    .loc[lambda df: df.index <= end_date]
    .dropna(axis=1)
)
chosen_stock_ids = stock_returns.columns

def standardize_ranking(ranking):
    return (ranking - min(ranking)) / (max(ranking) - min(ranking))

stock_characteristics = (
    chosen_characteristics
    .loc[lambda df: df.date >= start_date]
    .loc[lambda df: df.date <= end_date]
    .loc[lambda df: df.stock_id.isin(chosen_stock_ids)]
    .assign(**{
        k: lambda df, col=k: (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
        for k in ["hml_ranking", "mom_ranking", "smb_ranking"]
    })
    .rename(columns=lambda x: x.replace("_ranking", ""))
)