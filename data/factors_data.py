import datetime
import re
import pandas as pd
import polars as pl
import io
import requests
from zipfile import ZipFile
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

ken_french_website_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
zip_files = {'fama_french_factors': 'F-F_Research_Data_Factors_CSV.zip', 'momentum_factor': 'F-F_Momentum_Factor_CSV.zip'}

current_dir = os.getcwd()  # get the current working directory
os.chdir(os.path.join(current_dir, "data"))  # change the directory to the "data" folder

if os.path.exists("fama_french_factors.csv"):
    fama_french_factors = pd.read_csv("fama_french_factors.csv", skiprows=3)
else:
    url = ken_french_website_url + zip_files['fama_french_factors']
    response = requests.get(url)
    with ZipFile(io.BytesIO(response.content)) as zip_ref:
        csv_file = zip_ref.extract(zip_ref.namelist()[0])
        fama_french_factors = pd.read_csv(csv_file, skiprows=3)
        os.rename(csv_file, "fama_french_factors.csv")

if os.path.exists("momentum_factor.csv"):
    momentum_factor = pd.read_csv("momentum_factor.csv", skiprows=13)
else:
    url = ken_french_website_url + zip_files['momentum_factor']
    response = requests.get(url)
    with ZipFile(io.BytesIO(response.content)) as zip_ref:
        csv_file = zip_ref.extract(zip_ref.namelist()[0])
        momentum_factor = pd.read_csv(csv_file, skiprows=13)
        os.rename(csv_file, "momentum_factor.csv")

os.chdir(current_dir)

factors_returns = (
    fama_french_factors
    .merge(momentum_factor, on='Unnamed: 0', how='outer')
    .rename({"Unnamed: 0": "date"}, axis=1)
    .rename(columns=lambda x: x.lower().replace("-", "_minus_"))
    .loc[lambda df: df.date.map(lambda x: bool(re.search("^[0-9]{6}$", str(x))))]
    .assign(date=lambda df: df.date.map(lambda x: datetime.date(int(x[:4]), int(x[4:]), 1)))
    .set_index("date")
    .apply(lambda x: x.astype(float) / 100)
    .rename(columns=lambda x: x.replace(" ", ''))
)

def returns_plot(returns_df, start_date=datetime.date(1900, 1, 1)):
    start_date = (
        datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(start_date, str) else start_date
    )
    ffc_accum = (
        returns_df
        .loc[lambda df: df.index >= start_date]
        .apply(lambda x: (1 + x).cumprod() - 1)
        .rename(columns=lambda x: x.replace("_minus_", " - ").upper())
        .dropna(how='any')
    )
    plt.figure(figsize=(10, 6))
    sns.set(style='whitegrid', palette='Spectral')
    ax = sns.lineplot(data=ffc_accum, dashes=False, linewidth=1.5, alpha=0.9)
    ax.set_title(
        'Accumulated Returns for Fama-French-Carhart Factors'
        + '\n' + f'From {min(ffc_accum.index).strftime("%m/%Y")} to {max(ffc_accum.index).strftime("%m/%Y")}'
    )
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: '{:.0%}'.format(x))
    )
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Accumulated Returns')
    plt.show()
