# %%
import numpy as np
import itertools
import pandas as pd
import itertools
import random
import string
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

def export_dict(clf, feature_names=None):
    tree = clf.tree_
    if feature_names is None:
        feature_names = range(clf.max_features_)
    n_outputs = clf.n_outputs_
    
    # Build tree nodes
    tree_nodes = []
    for i in range(tree.node_count):
        if (tree.children_left[i] == tree.children_right[i]):
            if n_outputs == 1:
                tree_nodes.append(tree.value[i][0])
            else:
                tree_nodes.append(tree.value[i])
        else:
            tree_nodes.append({
                "feature": feature_names[tree.feature[i]],
                "value": tree.threshold[i],
                "left": tree.children_left[i],
                "right": tree.children_right[i],
            })
    
    # Link tree nodes
    for node in tree_nodes:
        if isinstance(node, dict):
            node["left"] = tree_nodes[node["left"]]
        if isinstance(node, dict):
            node["right"] = tree_nodes[node["right"]]
    
    # Return root node
    return tree_nodes[0]


def get_breakpoints(returns_table, max_depth):
    X = returns_table.drop("expected_return", axis=1)
    y = returns_table["expected_return"]
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X, y)

    breakpoints = export_dict(tree, feature_names=X.columns.tolist())
            
    return breakpoints


def create_combined_expand_grid(returns_table, expected_returns_column=None, num_combinations=100, **kwargs):
    """
    Takes a returns_table as input parameter and creates a combined "expand_grid" 
    with every element of every column. The number of combinations to generate 
    can be specified with the num_combinations parameter (default is 100).
    """
    column_order = list(returns_table.columns)
    expanded_variables = [c for c in list(returns_table.columns) if c not in kwargs.keys() and c != expected_returns_column]
    
    # Initialize an empty list to store the expanded grids for each column
    expanded_grids = []

    # Loop through each column in the returns table
    for column in expanded_variables:
        col_min = returns_table[column].min()
        col_max = returns_table[column].max()
        expanded_grid = np.linspace(col_min, col_max, num=num_combinations)

        # Add the column's expanded grid to the list of expanded grids
        expanded_grids.append(expanded_grid)

    # Use itertools to create a combined expand_grid from the expanded grids for each column
    combined_expand_grid = list(itertools.product(*expanded_grids))

    # Convert the combined expand_grid into a pandas DataFrame
    combined_expand_grid_df = pd.DataFrame(combined_expand_grid, columns=expanded_variables)
    if kwargs is not None:
        combined_expand_grid_df = combined_expand_grid_df.assign(**kwargs).loc[:, column_order]

    # Return the DataFrame
    return combined_expand_grid_df


def attribute_node(tree, returns_table, expected_returns_column=None, num_combinations=100, **kwargs):
    X = returns_table if expected_returns_column is None else returns_table.drop(expected_returns_column, axis=1)

    expand_grid_table = create_combined_expand_grid(X, num_combinations=num_combinations, **kwargs)
    expand_grid_table = expand_grid_table.assign(node=tree.apply(expand_grid_table))

    return expand_grid_table


def create_heatmap(expand_grid_nodes):
    fixed_features = (
        expand_grid_nodes
        .loc[:, expand_grid_nodes.nunique() == 1]
        .iloc[0, :]
        .to_dict()
    )
    fixed_features_subtitle = [f"{k} ({round(v, 3)})" for k, v in fixed_features.items()]
    expand_grid_nodes = (
        expand_grid_nodes
        .loc[:, expand_grid_nodes.nunique() != 1]
    )

    heatmap_data = expand_grid_nodes.pivot_table(
        index=expand_grid_nodes.columns[0], columns=expand_grid_nodes.columns[1], values='node'
    )
    ax = sns.heatmap(heatmap_data, cmap='rocket')
    plt.xticks(np.arange(len(np.arange(0, 1.1, 0.1))) * 10, np.arange(0, 1.1, 0.1), fontsize=10)
    plt.yticks(np.arange(len(np.arange(0, 1.1, 0.1))) * 10, np.arange(0, 1.1, 0.1), fontsize=10)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.text(
        x=0.5, y=1.1,
        s=f'AP-Tree Groups by {expand_grid_nodes.columns[0]} and {expand_grid_nodes.columns[1]}',
        fontsize=16, weight='bold', ha='center', va='bottom',
        transform=ax.transAxes
    )
    if fixed_features_subtitle != []:
        fixed_features_subtitle = ",".join(fixed_features_subtitle) + "."
        ax.text(
            x=0.5, y=1.05,
            s=f'Fixed Features: {fixed_features_subtitle}',
            fontsize=12, alpha=.8, ha='center', va='bottom', transform=ax.transAxes
        )

    plt.show()


def create_ap_tree_sorting(tree, stock_characteristics):
    if "stock_id" in list(stock_characteristics.columns):
        stock_characteristics = stock_characteristics.set_index("stock_id")
    stock_ap_tree_groups = {}
    stock_characteristics = (
        stock_characteristics
        .assign(group=tree.apply(stock_characteristics.loc[:, list(tree.feature_names_in_)]))
    )
    for group in list(set(stock_characteristics.group)):
        stock_ap_tree_groups[str(group)] = list(
            stock_characteristics\
            .loc[lambda df: df.group == group]
            .index
        )
    return stock_ap_tree_groups

    
# # %%
# return_table = pd.DataFrame({
#     'stock': ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'FB'],
#     'expected_return': [0.05, 0.08, 0.07, 0.06, 0.09],
#     'SMB': [0.7, 0.8, 0.6, 0.9, 0.5],
#     'WML': [0.3, 0.4, 0.2, 0.6, 0.1],
#     'GBK': [0.5, 0.6, 0.4, 0.8, 0.2]
# })

# return_table = return_table.set_index('stock')
# tree_dict = get_breakpoints(return_table, max_depth=2)

# # %%
# return_table = (
#     pd.DataFrame({
#         'stock': [''.join(random.choices(string.ascii_lowercase, k=4)) for _ in range(200)],
#         'expected_return': np.random.normal(0.02, 1, 200),
#         'SMB': np.random.normal(0.5, 1, 200),
#         # 'mom': np.random.normal(0.5, 1, 200),
#         'HML': np.random.normal(0.5, 1, 200)
#     })
#     .set_index('stock')
#     .apply(lambda x: x.map(lambda y: (y - x.min()) / (x.max() - x.min())))
#     .assign(expected_return=lambda df: df.expected_return / 100)
# )

# X = return_table.drop('expected_return', axis=1)
# y = return_table['expected_return']

# tree = DecisionTreeRegressor(max_depth=4)
# tree.fit(X, y)

# expand_grid_nodes = attribute_node(
#     tree=tree,
#     returns_table=return_table,
#     expected_returns_column='expected_return',
#     num_combinations=100,
#     # **{"GBK": 0.9}
# )

# create_heatmap(expand_grid_nodes)
# %%
