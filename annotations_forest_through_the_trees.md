# FOREST THROUGH THE TREES: BUILDING CROSS-SECTIONS OF STOCK RETURNS
# Introduction
What explains expected returns:
1. the returns of test assets in cross-section
2. the model that succeeds in pricing them

If the test assets do not span the underlying Stochastic Discount Factor (SDF: price of the asset is computed by the cash flow discounted by the stochastic factor) there is no reason to expect that a well-fitting model is anywhere close to the truth.

## Problems in the conventional way of sorting factor based portfolios
The convencional sorting-based portfolios drastically fail to span the SDF and hence lead to the wrong conclusions when used to evaluate or construct asset pricing models. Those sorting-based portfolios (i) suffer from the curse of dimensionality and neglect characteristic interactions.

Solution: asset pricing trees addresses every problem; complex interactions, curse of dimensionality, repackaging and duplication, thus providing a new way of building better cross-sections of portfolios.

## AP-Tree approach as a solution and the method towards better portfolios
AP-Trees group stocks in different managed portfolios that reflect information in a given set of characteristics.

The method is composed of two key elements:
1) the construction of conditional tree portfolios, and;
2) the prunoing of the oveall portfolio set based on the SDF spanning requirements.

An AP-Tree divides the in many groups. A full AP-Tree is the combination of all possible orders and splits.

With AP-Trees, we use the return as the explained variable and the factors as the features.

Important: each portfolio (tree node) can always be traced back to economic fundamentals.

After that, we use prunning: we consider the whole set of potential managed portfolios offered by AP-Trees and develop a new approach to reduce them to a small number of interpretable test assets. In other words, assets are combined together in higher-level nodes, making the original portfolios redundant. With that, we map the pruning into a robust SDF recovery within a mean-variance framework.

We group individual stocks into the same managed portfolios, if they have the same conditional contribution to the SDF.

Tree-based construction enforces a stable and balanced composition by default, making them fully comparable and often more diversified than triple sorts of the same depth.

## Advantages of AP-Trees
Compared with decile-sorted portfolios (100 assets) we double both the Sharpe ratio achieved out-of-sample and the alphas associated with the SDF spanned by these portfolios relative to traditional models.

This demonstrate the importance of taking into account the joint distribution of multiple characteristics. Furthermore, it shows that the superior performance of AP-Trees is due to their ability to efficiently capture interactions among characteristics.

When using prunning, it is shown that the inclusiion of mean or variance shrinkage in addition to sparsity shrinkage improves almost all out-of-sample results.

# Test Assets, Sorting, and Trees
The value of a stock is represented by M and is constructed by a valid Stochastic Discount Factor. The same can be viewed as the acumulated returns of this same asset which are functions of the assets characteristics. The asset characteristics are represented by a line of C. C is a matrix with N x K size, being N the number of stocks and K the number of characteristics observed.

## The importance of SDF spanning
The SDF spanning requirements implies that we should find set of managed portfolios such that their combination has the highest Sharpe ratio, which is equivalent to minimizing the distance between the true SDF and a candidate one. It is necessary that the R is represented by the corrected portfolios: if this is not true, a model that explains it well could still be far from the actual pricing kernel.

Optimal portfolios spanning the SDF, projected on a certain group of characteristics, should possess the following properties:

a) they should be able to reflect the impact of multiple characteristics at the same time.
b) they should span the SDF, that is, achieve the highest possible Sharpe Ratio out-of-sample when combined together.
b) they should allow for a flexible nonlinear dependence and interactions.
d) they should be a relatively small number of well-diversified managed portfolios feasible for investors.
e) they should provide an interpretable link to fundamentals.

# Conditional and Unconditional Sorting
## Conditional sorting
Conditional sorts (tree) can model interactions among many characteristics without suffering from the curse of dimensionality. Furthermore, the tecnique works particurlarly well when the underlying characteristics have a complex joint distribution: trees group together securities that have similar economic fundamentals and are directly interpretable.

## Unconditional sorting
On the other hand, unconditonal sorting (sorting with decils, for instance), does not allow us to do many combination of factors, because this would lead to small amount of stocks per portfolio. Furthermore, there is no guarantee that the underlying test assets span the SDF regardless of the size of the cross-section.

Important: the portfolios are constructed for each "t" with the information of "t-1".

## Empirical perceptions
As we see empirically, stocks have a complicated joint relationship that makes longer trees better when trying to sort.

Empirically, the paper has seen that there is a "negative correlation" (in part, mechanical) between size and book-to-market, with a clear clustering around the north-west and south-east corners. Therefore, only two sorts might not make a good "clusterization".

Furthermore, the paper points that the impact of characteristics on expected returns are highly nonlinear and full of interaction effects. For instance, the value effect is higher on the smallest decils, on other hand, the impact of accrual is almost flat for large stocks and have an inverted U-shape pattern for medium and small stocks.

Since characteristics are generally dependent and have a nontrivial joint impact on expected returns, the order of the variables used to build a tree and generate conditional sorts matter.

## Complexity in characateristic's relationships
The amount of interactions between factors and their order of relationship with returns of the stocks is proportional to the depth of the tree.

In conclusion, due to the complicated relationship between returns and factors, it is better to use the AP-Tree (conditonal sorting).

# Recursive Portfolios and Split Choice
## Use of final and intermediate nodes
Important point: AP-Tree is a unique feature that makes them a particularly appealing choice for test assets because we can use the final and intermediate nodes to test. While the final nodes provide more granular information, the intermediate nodes are more general and will have more stocks to make the portfolio.

Moreover, the paper proposes a combinatory approach: a final node is used if there is enough information gain, otherwise, the intermediary node is used.

AP-Prunning: The optimal selection among both final and intermediate nodes of the trees, allows a creation of a cross-section manageable in size with endogenous choice of the types of the splits and their depth.

## Advantage of intermediary nodes vs. final nodes
Tree portfolios at higher nodes are more diversified, naturally leading to a smaller variance of their mean estimation, and so forth, while more splits allow us to capture a more complex structure in teh returns at the cost of using investment strategies with higher variance.

## When to use final and intermediary nodes
Final result of AP-Prunning: In the end, the paper's authors decide to only use the children node if the children adds asset pricing information that outweights the increase in variance.

In other words, the sparsity operates directly on assigning individuals stocks into groups.

Furthermore, we rely on shrinkage to deliver robust and reliable portfolio selection.

Instead of using the conventional approach for minimum variance portfolio, they use the weights derived from the robust method, with elastic net penalty. 

## Definition of the process of pruning
Divide the original sample of data into three non-intersecting subsamples for use in training, validation, and testing of the model.
1. Use the training set to tune the paramet
ers (mu, lambda 1, and lambda 2 - two lambdas are used because of elastic net).
2. Use the validation data to estimate the Sharpe ratio of the optimal portfolio as a function of tuning parameters (mu, lambda 1, lambda 2). Select parameters that maximize Sharpe ratios.
3. A previously unused testing subsample of data is used to trace the fully out-of-sample performance of the SDF and the properties of the individual portfolios in the chosen cross-section.

## Shrinkage of the mean
Furthermore, they use a shrinkage to the mean. The shrinkage to the mean is rarely found to be zero and often has a significant effect on the chosen portfolio spanning the SDF.
That is because estimated expected returns have severe measurement errors, it is likely that extremely high or low rates of returns are actualy overestimated/underestimated simply due to chance and, hence, it left unchanged, would bias the SDF recovery. Therefore, mean shrinkage provides robustness against the overall Sharpe ratio uncertainty, variance shrinkage governs robustness against variance, while reflecting uncertainty on the mean estimation.

Individual, lasso, ridge, and shrinkage to the mean have been successfully motivated with both robust control and a Bayesian approach to portfolio optimization with a specific choice of uncertainty sets, leading to substantial empirical gains in a number of applications.

Attention: tree-based asset returns are long-only portfolios, which are easy to trace back to the fundamentals and interpret.

# Empirical Design: Data
Use of standard data from CRSP/Compustat to construct portfolios based on firm-specific characteristics from jan/64 to dec/16, and the one-month treasury bill rate as a proxy for the risk-free rate, yielding 53 years to monthly observations.

Use as well 10 firm-specific characteristics from the Kenneth French Data Library, based accounting and market data. Market-based measures are updated monthly, while all the accounting variables are updated at the end of June.

AP-Tree does not require every stock to have every single characteristic: if one industry lacks data, it is not problematic.

## What are the firm-specific characteristics?
- AC (Accrual);
- BEME (Book-to-Market ratio);
- IdioVol (Idiosyncratic volatility)
- Investment
- LME (Size)
- LT_Rev (Long-term reversal)
- Lturnover (Turnover)
- OP (Operating profitability)
- r12_2 (Momentum)
- ST_Rev (Short-term reversal)

## Experiment construction
1964-1983: training sample, 20 years
1984-1993: validation-sample, 10 years
1994-2016: testing-sample, 23 years

# Empirical Design: Estimation and Hyperparameter Tuning
## Tree construction, size and its comparison to the Fama French portfolios
To get results closer to the literature, we start with a triplet of stock-specific characteristics: size and two other variables, which implies a set of 36 cross-sections based on three variables each.
Later, they consider an AP-Tree with depth four and its closest analogue in the literature: 32 and 64 triple sorts.
Finally, portfolios at the level-four nodes based on a sinle characteristic are excluded, because they could count as an unfair advantage to the AP-Trees. With that, there are 40 AP-Tree portfolios in the baseline empirical application, which makes the dimension of pruned trees comparable to the Fama-French triple-sorted 32 and 64 portfolios.

## Training sample estimation
Estimate portfolios for both final and intermediate nodes that are used to construct efficient portfolio frontier with elastic net regularization.
For different levels of shrinkage, we select optimal test assets and form the corresponding SDF spanned by them.
The same procedure is applied to Fama French triple sorts (as an element of comparison).

## Validation sample estimation
The 10 years ahead are used to select the best tuning parameters. We track the performance of the portfolio frontier (formed in the training sample), based on the validation data, and select optimal portfolio shrinkage to maximize its Sharpe ratio. Lasso penalty is used to set the number of non-zero weights to the target number of portfolios.

## Testing sample
Used to compare the performance of different basis assets, achieved fully out-of-sample.

# Empirical Design: Baseline Evaluation Metrics
Use many different models to find the results:
- FF3 (market, size, and value)
- FF5 (FF3 + investment, profitability)
- XSF (cross-section-specific model with market and three long-short portfolios, corresponding to the three characteristics used in the cross-section)
- FF11 (an 11-factor model, consisting of the market factor and all 10 long-short portfolios, based on the full set of 10 characteristics)

For each use, the paper's authors report:
- SR: out-of-sample Sharpe ratio of the SDF constructed with AP-Trees, TS (32 portfolios in the FF format), and TS (64 portfolios in the FF format)
- alpha: SDF pricing error, and its t-statistics, obtained from the linear regression against different factor models.
- alpha_i: pricing error for individual basis assets, relative to a set of candidate factors.
- XS-R^2: relative magnitude of pricing errors in a given cross-section

# AP-Trees vs. Triple Sorts: 36 Cross-Sections of Expected Returns
This part shows the out-of-sample SR of th AP-Trees, triple sorts, and the cross-section-specific long-short portfolios.
Cross-sections are sorted based on the SR achievable with 40 AP-Tree portfolios.

## AP-Tree sharpe out-of-sample results comparison
AP-Trees deliver considerably higher out-of-sample Sharpe ratios compared to triple sorts .or conventional long-short factors.
The worst performance is seen in the long-short factors, with an out-of-sample Sharpe with around a third of the value of the AP-Trees.
The worst performance of the long-short factors is expected, since it cannot account for the important cross-sectional information. Triple-sorting could reflect their impact at least to a partial extent, but doing so with 32 or 64 triple-sorted portfolios does not have high impact.

## Regressing the results by the factors used
Important: it seems that, after doing the portfolios, the paper regress the portfolios by the factors used to understand if it can still generate alpha. What is seem is that the alpha is larger for AP-Trees and the Adjusted R^2 is smaller, when compared to the triple-sorting and the long-short factors.

# AP-Trees vs. Triple Sorts: How Many Portfolios?
Increasing the number of portfolios may not always be beneficial. Larger number of portfolios could span better the SDF, but it could also lead to unnecessary repackaging of the same data without providing any new information. Therefore, how many portfolios is the optimal number?

We use lasso penalty as a way of building cross-sections of different sizes, and find that almost all of our empirical results carry through using only a quarter of the original basis assets. For most cases, using just a quarter of the original cross-section is enough to retain roughly 90% of the original Sharpe ratio and its alpha relative to the FF5F model.

Furthermore, the optimal number of portfolios depends on the complexity of the conditional SDF, projected on these characteristics, and could be chosen optimally based on the validation sample or using a full cross-validation.

Last, the depth of the tree is ultimatelly dependent of the best mean-variance-efficient (MVE) optimization.
The paper concludes that intermediate nodes of the tree constitute a substantial fraction of the chosen basis assets and substantially contribute to the overall projection of the pricing kernel.

# Nonlinearities and Characteristic Interactions
Nonlinearity can come from:
- a nonlinear impact of a particular characteristic on asset returns/volatility.
- complicated general interaction effects among characteristics.
The AP-Trees approach does not require exact parametrization of these effects; that is, combining conditional sorts with pruning allows us to capture general forms of nonlinearities without having to model them explicitly.

The lack of interactions between characteristics lead to out-of-sample Sharpe generally half as large as those of the general AP-Trees that include interaction nodes.

# Zooming into the Cross-Section
The results shows that the use of intermediary nodes do not provide worst result when comparing to final nodes. In other words, comparing 10 portfolios with 40, we see similar out-of-sample sharpe. On the other hand, triple sorting with 32 and 64 perform way worst than both AP-Trees. The results shows that the conditional tree-based splits are the "real deal".

Furthermore, the paper states that out of the 10 selected portfolios with AP-Trees, only 5 are members of final nodes.

It seems like the triple sort is done by using 4 quartiles of each characteristic: leading to 64 different portfolios. In the case study, the excessive granular nature of the triple sorted portfolios can also mask the true fit of the leading asset pricing models: in this case, we are treating an unique portfolio that is perfectly (or something like that) spanned as two different portfolios. With less stocks, the R^2 is bigger for the OLS estimation, while the GLS estimation has empirically lower R^2.

# Time Variation
To study time variation in the weights, we fix the portfolio selection based on the training and validating sample and estimate the optimal ronust combination of the basis assets on a rolling window for 20 years.

Time variation seems to be really important for both triple sort and AP-Trees.
For both the AP-Tree and the triple sort, the out-of-sample Sharpe is improved with time varying weights.

AP-Trees clearly dominate triple sort in every portfolio. Furthermore, both AP-Trees and triple-sort benefit from shrinkage to the mean, which substantially stabilizes empirical performance.

The sample of data is divided into three equal time-series blocks. Each block is used as a training period, and the model is evaluated on the remaining two.

# Evaluating Asset Pricing Models
This sections illustrate two points.

## Optimal SDF
The test assets are as important as the tests done: the optimal cross-section should span the SDF projected on stocks via characteristics, and allow for an easy diagnostic of model performance. Traditional sorts (specially deciles and simple long-short anomalies) do not span the underlying SDF, therefore, even capturing alpha, the model might be misspecified.

## Metric choice
In the authors view, the SDF alpha is the best wau to measure, since it aggregates all the information from the test assets and measures its out-of-sample feasible to investors performance.

# Pruning AP-Trees and Portfolio Selection
The pruning is a novel approach to select a small set of AP-Tree nodes with the most non-redundant pricing information for spanning the SDF. There are many conventional ways to prune a tree are not good in this particular case: the key issue lies in the fact that finding an optimal set of portfolios that span the SDF and maximize Sharpe ratio is a global problem and cannot be handled by local decision criteria.

The authors solve the problem by considering simultaneously both the final and intermediate nodes and applying the mean variance optimization to both.

As mentioned before, the minimum variance portfolio weights are found with elastic net penalty.

The training data is used to test different lambda parameters. The validation set chooses the best lambda. The testing subsample is used to trace the fully out-of-sample performance of the SDF and the properties of the individual portfolios in the chosen cross-section.

# DISSECTING ANOMALIES (FAMA FRENCH ARTICLE ANNOTATIONS FOR DEEPER UNDERSTANDING)
There are two approaches which are common when trying to identify anomalies: (i) sorts of returns on anomaly variables, and (ii) regressions.

## Sorts
The main advantage of sorts in a simple picture of how average returns vary across the spectrum of an anomaly variable.

## Equal-weight decil portfolios
A common approach is to form equal-weight (EW) decile portfolios by sorting stocks on the variable of interest. Generally the focus is largely applied to detailed results for extreme deciles. The first problem with that is the fact that portfolios can be dominated by tiny companies, microcaps, which account for 60% of the number of stocks, but only account for 3% of the market cap of the NYSE. Furthermore, smallcaps tend to account for more anomalies, therefore, tipically staying in more extreme sort portfolios.
Therefore, a solution 

## Value-weight hdege portfolio
Steady of giving the same weight to all the stocks, the weights of the portfolio are directly related to the relative size of the stocks.
On the other hand, this can lead to portfolios largely dominated by a few big stocks, resulting again in an unrepresentative picture of the importance of the anomaly.

## Separating stocks in quantile by size
To solve the problems, the sorting is done using quantiles for the size of the stock, being the sizes until percentil 20, 50 and larger than 50.

## Advantages of regression
While regression provides a clear marginal effect, the sorting does not come with the same possibility. On the other hand, regressions make all stocks weight the same. Therefore, when using regression, an alternative is separate the slope for microcaps, small stocks, and big stocks.

# WHAT IS A STOCHASTIC DISCOUNT FACTOR?
Net present value is made by discounting the cash flows of the future.
The discounted cash flows of the future brings the value to the present time by discounting it by a factor that takes into account the:
- time value of money (risk free)
- risk premium for non-diversifiable risk

The process of the SDF is generally abreaviate with capital M.

