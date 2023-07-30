# Forest Through the Trees: Building Cross-Sections of Stock Returns Replication
## Code Material
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

## Abstract
This is replication is made to reinforce my interest in factor investing. The replication aims to create test assets using AP-Tree and compare the results of the test assets generated with the unconditional sorting. To compare the results, the test assets for both methods are used in a cross-section of returns to verify a capacity of a factor model to capture systematic risk and used to produce an out-of-sample Sharpe ratio, in order to attest the capacity to span the SDF. The test assets used are a crucial aspect when evaluating if a factor model captures all systematic risk and should be able to correctly span the SDF by representing all economic risk, and, therefore, have the highest out-of-sample Sharpe when combined efficiently. The results found in the original paper show that AP-Tree test assets considerably outperform unconditional sorting test assets in its capacity to span the SDF, due to its higher Sharpe ratio, and often lead to bigger and more significantly different from zero alpha in the cross-sections of returns, disproving the capacity of some factor models to price all systematic risk. The results found in this replication agree with results found on the original paper, but do not lead to an outperformance of the AP-Tree when compared to unconditional sorting test assets as big as the one found in the original work.

## Disclaimers
Two major simplifications were made: (i) use of RIDGE steady of elastic net and mean shrinkage to produce the optimal mean-variance allocation, which permits the utilization of a close-end formula, (ii) use only of final nodes in the AP-Tree as test assets. Simplification number (ii) probably can be regarded as the major reason for difference in results: the new algorithm captures well the complex joint distribution of characteristics and avoids the curse of dimensionality by only dividing stocks into different groups when it helps to explain the excess of returns, nevertheless, only using final nodes leads to investment strategies with higher variance which do not necessarily reflect every time economic fundamentals. Simplification number (i) probably also declined the performance of the AP-Tree test assets, since the paper specifies that mean shrinkage often had a significant effect on the chosen portfolios spanning the SDF.

## 1. INTRODUCTION
### 1.1. THE QUESTION TO ANSWER: WHAT TEST ASSETS SHOULD BE USED TO EVALUATE A FACTOR MODEL?
The idea of the SDF is the basis of the whole factor investing industry, but it is not in the mind of every factor investor…

Nowadays, a zoo factor is present, and investors constantly find a risk-premium for what they happen to consider systematic risk. Nevertheless, finding a “risk-premium” or testing a new or old factor model is part of a scientific method, which requires finding a set of assets that can span the SDF, conditional on a given information set.

According to the idea behind the SDF, the price at any point in time of any asset can be calculated by the future cash flows discounted by a stochastic factor which accounts for the value of money in time (risk-free rate) and the non-diversifiable risk (systematic risk). Therefore, the difference between prices in two periods of time, in other words, the expected return, is the SDF, and can only be explained by the systematic risk and the risk-free rate. Thus, the excess of return can only be explained by systematic risk, a.k.a. exposition to factors. According to this view, being exposed to the right factors (maybe even at the right time) is the crucial aspect, which makes factor investing a continuous search for the model that succeeds in pricing the test assets; in order words, explain all the excess of return by the exposure of systematic risks, meaning that in a cross-section of betas (the exposure of an asset to a particular factor) explaining the excess of returns of an asset, the intercept (named “alpha”) should be non-significantly different from zero. Finding the factor model that can price all systematic risk is an arduous task, nevertheless, it is not the only. Before testing factor models, investors must decide which test assets to use. Employing test assets non-representative of the overall economy leads to not being able to correctly span the SDF.

When the SDF is not correctly spanned, a well-fitting factor model does not represent the truth since it does not account for all economic risk. In other words, the decision behind which test assets to use to verify the efficacy of a factor model to absorb the systematic risk is as crucial as deciding what factors should be considered systematic risks. For instance, if the first part is done wrongly, one “incomplete” factor model might not generate alpha, meaning that you would be prone to think that this “incomplete” model is able to capture all systematic risk, while a good set of test assets would clearly disprove the affirmative.

The idea behind spanning the SDF gained importance with the creation of multi-factor models. For instance, to create the FF3 model, Eugene Fama and Kenneth French realized that the market factor did not capture all systematic risks, since listed companies with smaller market caps systematically outperformed listed companies with bigger ones, and listed companies with bigger book-to-market ratios systematically outperformed listed companies with smaller ones. Therefore, to scientifically evaluate the performance of the CAPM, they proposed the construction of long-short portfolios of stocks based on the characteristics previously described. They regressed the portfolios by the market index, and finally constructed a cross-section with the market betas of those portfolios, in which the market attempts to explain the excess of return. The result was clear: the expected returns of those portfolios could not be fully explained by the market on a considerable set of cross-sections. That is, portfolios that accounted for more economic risk were able to disprove the affirmative that the market contained all systematic risks. The result of Fama and French was largely due to the capacity to look at a different set of assets and generate better test assets than the ones considered before.

Nevertheless, their test assets were not perfect…

### 1.2. CONDITIONAL VS. UNCONDITIONAL SORTING 
In this paper replication, we explain and reproduce conventional unconditional sorting, the same method used by Fama and French, and the new conditional sorting method: AP-Tree. During the explanation of the AP-Tree we point out the problems of the old method and address the simplifications done in comparison to the complete article.

#### 1.2.1. TIME SERIES AND CROSS-SECTIONS: HOW TO TEST THE PRICING CAPACITY OF A FACTOR MODEL? 
Before entering in conditional and unconditional sorting, we jump to the end of the experiment, which is the test of a factor model, so later we can focus only on the main team. The final part of the experiment for a given pool of test assets is composed of:

- i) Regress the excess return of each one of the test assets by the returns of the factors in the out-of-sample (time after the portfolio creation); 
- ii) Capture the betas of the factors for each one of the test assets and use each of the vectors of factor betas as a feature, where the number of observations for each is the number of tests assets;
- iii) Use those betas as features in a cross-section regression explaining the accumulated excess of return of the test assets for the period;
- iv) Check the presence of an intercept, which is called alpha, and if it is significantly different from zero. 

If the alpha of the cross-section regression is non-significantly different from zero, one of two options will be true: (i) the factor model captures all systematic risk for this period, or (ii) the test assets used did not span the SDF well and do not represent all economic non-diversifiable risk, creating misleading results.

After applying unconditional and conditional sorting to find the test assets, we do the four steps described above.

#### 1.2.2. DESIRABLE CHARACTERISTICS OF PORTFOLIOS SPANNING THE SDF
As seen before, the non-optimal spanning of the SDF leads to unreliable results. That is why correctly spanning the SDF, according to the paper, should:

- i) reflect the impact of multiple characteristics at the same time. 
- ii) achieve the highest possible Sharpe Ratio out-of-sample when combined together.
- iii) allow for a flexible nonlinear dependence and interactions. 
- iv) be a relatively small number of well-diversified managed portfolios feasible for investors.
- v) provide an interpretable link to fundamentals.

#### 1.2.3. UNCONDITIONAL SORTING 
The unconditional method, used by Fama and French, uses quantiles of characteristics to divide stocks into groups and later create portfolios. For instance, in “triple sorting”, the quantiles are divided for each of three different characteristics, and in “double sorting” the quantiles are divided for each of two different characteristics. After, the creation of portfolios is done by staying long on one group and short on the group on the opposite side.

In our replication, we create groups of stocks using terciles when using quantiles for three characteristics: Momentum, Book/Market, and Size, and create quartiles when using two characteristics: Book/Market, and Size. For instance, in the double sorting, to create the portfolios, we:

- Stay long on Q4 B/M, Q4 Size and short on Q1 B/M, Q1 Size; 
- Stay long on Q2 B/M, Q3 Size and short on Q3 B/M, Q2 Size; 
- Stay long on Q1 B/M, Q3 Size and short on Q4 B/M, Q2 Size; 
- Etc… 

The stocks in the long, and short parts of the portfolio are equally weighted. With this calculation, we end up with 32 portfolios for double sorting and 26 portfolios for triple sorting (for triple sorting, the quantile Q2 B/M, Q2 Size, and Q2 Momentum has no opposite quantile, and, therefore, it is not used). After constructing the portfolios, which are now test assets, we apply the method described in 1.2.1.

#### 1.2.4. CONDITIONAL SORTING: AP-TREE
The previous method used could disprove a simpler factor model capacity of pricing systematic risk, nevertheless, it is antiquated in its capacity to better span the SDF. Unconditional sorting fails in two major aspects regarding the previous task:

- i) Curse of dimensionality: in unconditional sorting, more characteristics necessary lead to more portfolios while keeping the number of quantiles stable. For instance, sorting by 13 different characteristics with quintiles would lead to 1.220.703.125 (513) different groups of stocks. The technique becomes impossible when dealing with a larger number of dimensions and ends up making unnecessary divisions that provide no interpretable link to fundamentals.
- ii) Complex joint distribution: unconditional sorting cannot account for nonlinear specific joint distribution, since it has predetermined divisions, failing to separate stocks with different economic fundamentals and failing to join ones with similar.

The method proposed by the paper suggests the assets’ separation into portfolios based on the nodes of a Regression Tree. By the way of explanation, instead of separating portfolios by just looking at the characteristics, the paper proposes using the past return of the assets as an explained variable, while the characteristics (like Book/Market index, Size, and Momentum) are used as features for explaining it. In the end, the nodes have assets that were used to model the returns. The separation in nodes instead of percentiles gives us a division conditional to relevant differences in characteristics, solving the complex joint distribution problem, and enables the use of more characteristics when trying to explain returns without suffering from the curse of dimensionality, since a characteristic will only be used if it is relevant to explain, solving the “overfitting” problem seen when separating by quantiles with a large amount of characteristics, since the depth of the tree can control the degree of the fitting. Particularly for the joint distribution problem, AP-Tree test assets outperform unconditional sorting to a much higher degree in samples where the joint distribution of characteristics is more complex. In the end, each node is selected to become a portfolio to be used as one of the test assets.

The paper’s authors carefully select the nodes to use, since the groups should always be traced back to economic fundamentals. Due to this particular necessity, the method proposes the use of both intermediary and final nodes, adding further separation only when it traces different economic fundamentals in each leaf. By reducing the number of nodes, meaning groups, and grouping them if they have the same conditional contribution to the SDF, the test assets become more interpretable. Tree-based construction enforces a stable composition, making them fully comparable and often more diversified than triple sorts of the same depth.

As an element of simplification, in the paper reproduction, we use only final nodes and control the tree by its depth and the minimum amount of samples in each leaf. We expect better results than sorting of the same depth, but not as good as the AP-Tree using intermediary and final nodes.

After creating the nodes with groups of stocks, each of those becomes a portfolio with weights defined by mean variance optimization. Due to the small determinant of the matrix of var-cov of asset returns, the original solution to the problem of mean-variance is often unstable: small changes in input lead to big changes in output, which are the weights. The use of regularization comes in hand to avoid its instability: the paper proposes the creation of mean-variance portfolios using Elastic-Net (RIDGE and Lasso), and mean shrinkage. To optimize the hyperparameter, the sample is divided into training and validation, using validation to find the optimal lambdas for RIDGE, LASSO, and mean shrinkage. Therefore, first the AP-Tree is done only on the training set, leading to nodes to be used as portfolios. Those portfolios are tested in the validation set with different levels of regularization to find the best fit. Later the AP-Tree is done joining the training and validation set and used to construct portfolios using the mean-variance approach with the best lambda found in validation.

As an element of simplification, we use only RIDGE regularization. According to the method, the portfolios should be long-only, and therefore, RIDGE is not capable of assuring that. As a result, after generating the optimal lambdas, when producing portfolios using the training and validation sets together, if a node produces a portfolio that when optimized by the mean-variance approach generates weights that are negative, we marginally increase lambda until the weights generated are all positive. Differently from LASSO, RIDGE has a close-end formula easier to implement, which is composed of multiplying lambda by an identity matrix and adding to the matrix of var-cov.

Finally, we have the test assets generated by AP-Tree and use those to execute the method pointed out in section 1.2.1., in the same way, it was done with the test assets generated by sorting.

## 2.	EXPERIMENT DESIGN AND METHOD
According to the information given before, the experiment for each (i) number of characteristics, (ii) factor model, (iii) depth of the tree, (iv) number of minimum sample size per leaf, and (v) testing sample, we:

### 2.1. UNCONDITIONAL SORTING
1. Take the pool of stocks and take the cross-section of accumulated excess of returns in 12 months at the end of the training sample, and their characteristics on the same date.
2. For each of the characteristics, rank the stocks from lowest to highest.
3. Divide the stocks in quantiles (meaning terciles if using three characteristics, and quartiles if using two characteristics).
4. Construct equal-weighted portfolios for each division.
5. Construct test assets with the equal-weighted portfolios of quantiles by going long on one division and short on the opposite division (for instance, one test asset is done by going long on Q4 B/M, Q4 Size, and short on Q1 B/M, Q1 Size).
6. Regress the assets by the chosen factor model in the testing sample in a time-series. For instance, when using a Fama French model, the following regression is done for each asset, where $tilde{r}_i$ is a vector of excess of returns of test asset, $i$, $r_MKT$ is a vector of returns of the market factor (which represents the return of the market minus the return of the risk-free rate), $r_HML$ is a vector of returns of the HML factor (High book-to-market ratio stocks minus low book-to-market ratio stocks), and $r_SMB$ is a vector of returns of the SMB factor (small caps minus big caps):
