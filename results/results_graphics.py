# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

def plot_cross_sections(df, name, file_name, ylabel, x_axis_label):
    sns.set(style='whitegrid', palette='Spectral')
    ap_tree = df['ap_tree']
    triple_sorting = df['triple_sorting']
    x = df.index
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, ap_tree, label='Rolling AP-Tree', color="#d53e4f", linewidth = 2.5)
    ax.scatter(x, ap_tree, color="#d53e4f", linewidth = 2.5)
    ax.plot(x, triple_sorting, label='Rolling Triple Sorting', color="#66c2a5", linewidth = 2.5)
    ax.scatter(x, triple_sorting, color="#66c2a5", linewidth = 2.5)
    ax.set_xlabel('Cross-section')
    ax.set_ylabel(ylabel)
    ax.set_title(
        f'{ylabel.replace("Monthly ", "")} of Portfolios Out-of-Sample'
        + '\n' + 'Ordered by AP-Tree Sharpe ratio out-of-sample'
    )
    ax.text(
        1, -0.2,
        "",
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes
    )
    months_testing_period = re.search(r"sml_([0-9]+)", file_name).group(1)
    characteristics = re.search(r"(?<=period_).*?(?=_characteristics)", file_name).group(0).upper().replace("_", ", ")
    factors = re.search(r"(?<=characteristics_).*?(?=_factors)", file_name).group(0).upper().replace("_", ", ")
    sorting_quantiles = re.search(r"(?<=factors_).*?(?=_sorting_quantiles)", file_name).group(0)
    ap_tree_max_depth = re.search(r"(?<=sorting_quantiles_).*?(?=_ap_tree_max_depth)", file_name).group(0)
    ap_tree_min_samples_leaf = re.search(r"(?<=ap_tree_max_depth_).*?(?=_ap_tree_min_samples_leaf)", file_name).group(0)
    graphic_info = [
        "MTP: " +  months_testing_period,
        "CHAR: " + characteristics,
        "FAC: " + factors,
        "SQ: " + sorting_quantiles,
        "MXD: " + ap_tree_max_depth,
        "MNS: " + ap_tree_min_samples_leaf
    ]
    ax.text(
        0, -0.2,
        "; ".join(graphic_info),
        verticalalignment='bottom',
        horizontalalignment='left',
        transform=ax.transAxes
    )
    ax.axhline(y=0, linestyle='--', color='black', linewidth = 1)
    num_ticks = len(x_axis_label)
    tick_positions = np.linspace(0, len(x) - 1, num_ticks)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_axis_label)
    ax.legend()
    plt.tight_layout()
    plot_name = file_name.replace(".csv", "_") + name + ".png"
    plt.savefig("results/graphics/" + plot_name)

files_names = [f for f in os.listdir("results/data") if re.match(r"^sml.*\.csv$", f)]

for file_name in files_names:
    all_sml_ap_tree_vs_triple_sorting = pd.read_csv("results/data/" + file_name)
    sharpe_ap_tree_triple_sorting = (
        all_sml_ap_tree_vs_triple_sorting
        .loc[lambda df: df.parameter == "alpha"]
        .loc[:, ["sharpe", "portfolio", "end_date"]]
        .rename({"sharpe": "value"}, axis=1)
        .pivot(columns="portfolio", values="value", index="end_date")
        .set_axis(["ap_tree", "triple_sorting"], axis=1)
        .reset_index()
        .sort_values("ap_tree")
        .reset_index()
        .rename({"index": "number"}, axis=1)
    )
    alpha_tvalue_ap_tree_triple_sorting = (
        all_sml_ap_tree_vs_triple_sorting
        .loc[lambda df: df.parameter == "alpha"]
        .loc[:, ["tvalue", "portfolio", "end_date"]]
        .rename({"tvalue": "value"}, axis=1)
        .pivot(columns="portfolio", values="value", index="end_date")
        .set_axis(["ap_tree", "triple_sorting"], axis=1)
        .reset_index()
        .merge(sharpe_ap_tree_triple_sorting.loc[:, ['end_date']], on='end_date', how='right', sort=False)
    )
    alpha_value_ap_tree_triple_sorting = (
        all_sml_ap_tree_vs_triple_sorting
        .loc[lambda df: df.parameter == "alpha"]
        .loc[:, ["value", "portfolio", "end_date"]]
        .rename({"value": "value"}, axis=1)
        .pivot(columns="portfolio", values="value", index="end_date")
        .set_axis(["ap_tree", "triple_sorting"], axis=1)
        .reset_index()
        .merge(sharpe_ap_tree_triple_sorting.loc[:, ['end_date']], on='end_date', how='right', sort=False)
    )
    rsquared_ap_tree_triple_sorting = (
        all_sml_ap_tree_vs_triple_sorting
        .loc[lambda df: df.parameter == "alpha"]
        .loc[:, ["rsquared", "portfolio", "end_date"]]
        .rename({"rsquared": "value"}, axis=1)
        .pivot(columns="portfolio", values="value", index="end_date")
        .set_axis(["ap_tree", "triple_sorting"], axis=1)
        .reset_index()
        .merge(sharpe_ap_tree_triple_sorting.loc[:, ['end_date']], on='end_date', how='right', sort=False)
    )
    plot_cross_sections(
        sharpe_ap_tree_triple_sorting,
        "sharpe", file_name, "Monthly Sharpe Ratio",
        sharpe_ap_tree_triple_sorting.number
    )
    plot_cross_sections(
        alpha_tvalue_ap_tree_triple_sorting,
        "alpha_tvalue", file_name, "Alpha T-Value",
        sharpe_ap_tree_triple_sorting.number
    )
    plot_cross_sections(
        alpha_value_ap_tree_triple_sorting,
        "alpha_value", file_name, "Alpha",
        sharpe_ap_tree_triple_sorting.number
    )
    plot_cross_sections(
        rsquared_ap_tree_triple_sorting,
        "rsquared", file_name, "R-Squared",
        sharpe_ap_tree_triple_sorting.number
    )
