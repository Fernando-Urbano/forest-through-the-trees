# Forest Through the Trees: Building Cross-Sections of Stock Returns Replication
Through this code, I object to implement a replication of the paper "Forest Through the Trees: Building Cross-Sections of Stock Returns".
The code is divided by its goals:
1. portfolio constructions: has the functions to calculate RIDGE optimization, sorting, AP-Trees, etc.
2. data: has the functions and databases to bring and organize characteristics, returns, and factors.
3. results: has the results generated by the calculation.

The whole code can be reproduce in four simple steps:
1. Assure the existence of the csv file "stock_characteristics_and_returns.csv" inside the "data" directory.
2. Run "main.py" file to get the results for multiple time-frames, factor models, characteristics, amount of sorting quantiles, and depth of the AP-Tree.
3. Run "results/results_graphics.py" to get the graphics of each experiment about (i) alpha t-value, (ii) Sharpe out-of-sample, (iii) R-squared.
4. Run "results/join_graphics.py" to join the graphic results.

Detailed explanation of the reproduction can be found in:
