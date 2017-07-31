## Multiple Linear Regression

* Multiple linear regression: a linear of function of several predictors (or covaraibles)

  * Formula ![mlr_formula](./img/mlr_formula.png)
  * Assumptions: errors are **i.i.d.** with zero expectation and an unknown variance; n > p; X is of full rank.
  * Look for good estimates of $\beta$ in terms of least squares![mlr_param](./img/mlr_param.png)
  * Variance $\sigma^2$ can be estimated using ![mlr_var](./img/mlr_var.png)
  * Assumptions for the linear model so that least square estimator and test and confidence intervals are approximately valid:
    1. LR equation is correct;
    2. All predictors can be perfectly observed;
    3. Homoscedasticity for errors (var is constant);
    4. Errors are uncorrelated;
    5. Errors are jointly normally distritubed.
  * ZSD: when a matrix $P$ is sym.; idem-potent $P^2 = P$; and has trace equal to dim, then it's a orthogonal projection
  * Simple least squares regressions on single predictor variables yield the multiple regression least squares solution, only if the predictor variables are orthogonal.

* Tests and confidence regions

  * Use *ANOVA* decomposition to test whether there is any effect from predictor variables (instead of individual test), where $\hat Y$ is the mean of Y: more specifically, use F-test to compare mean square of regression and error respectively![anova](./img/anova.png)
  * ![anova_full](./img/anova_full.png)
  * **Coefficient of determination**: goodness of fit of the linear model for explaining the data![R](./img/R.png)

* Analysis of residuals and checking of model assumptions

  * The Tukey-Anscombe plot: residual against fitted values
    * Linear: log transformation for y
    * Non-linear: square root transformation for y if quadratic
    * Systematic: either transform the response variable or use weighted regression
  * The normal plot: a special case of QQ plot (quantile-quantile), empirical quantile of residuals versus theoretical quantile of normal distribution
  * Residuals versus observation number to detect serial correlation

* Generalized least squares: errors are correlated with known covariance matrix; using square root of covariance matrix to solve the problem

* Model selection

  * Mallows $C_p$ statistic where SSE is residual sum of squares and $\hat \sigma ^2$ error variance estimate: ![mallows](./img/mallows.png)
  * Forward selection
    1. Start with the smallest model
    2. Include the predictor that reduces most the residual sum of squares
    3. Until all predictors are selected or a large number of predictors
    4. Choose the model in the sequence with the smallest $C_p$ statistic

  ​		

  ## Nonparametric Density Estimation

  *  Kernel estimator
    * Naive estimator![naive_kernel](./img/naive_kernel.png)
    * Typical kernel: gaussian, Epanechnikov (optimal w.r.t. mean squared error)

* Hyperparameter: bandwidth, which can be either global or local

  * IMSE (integrated mean squared error) ![imse](./img/imse.png)
  * ![mise](./img/mise.png)
  * Local and global optimal bandwidth (see p.p 20 - 22)



## Nonparametric Regression

* Nonparametric regression model, where noise is i.i.d. and centered, m is called nonparametric regression function satisfying $E[Y|X=x]$ ![nonparametric_regression](./img/nonparametric_regression.png)
* The kernel regression estimator
  * Nadaraya-Watson kernel estimator: plug univariate and bivariate kernel density into the conditional expectation ![Nadaraya_Watson](./img/Nadaraya_Watson.png)
  * Role of bandwidth
    * Similar to its role in nonparametric density estimation, we can deduce optimal local bandwidth
    * `library(lokern); lofit <- lokerns(cars$ speed, cars$ dist)`
* The **Hat Matrix S**: linear operator to map true Ys to fitted values
  * Degrees of freedom = tr(S)
* Local polynomial nonparametric regression estimator (**TBU**)
  * Extend to locally polynomial function from locally constant Nadaraya-Watson kernel estimator
* Smoothing splines and penalized regression
  * Penalized sum of squares, where large $\lambda$ gives a smooth function![penalized_sum_of_squares](./img/penalized_sum_of_squares.png)
  * The solution is a natural cubic spline with knots at the observed points, **see derivation**
  * Adaption to density of the predictor variables

## Cross-Validation

* Generalization performance
  * measures the predictive power of a learning method on new, out-sample data
  * generalization error: expected value of loss w.r.t training and test data, see (4.1)
* Leave-one-out CV
  * (4.2) is an estimate of generalization error
  * expensive due to retraining for every sample
* Leave-d-out CV
  * basic version: consider all subsets of d data points as test data
  * improvement using randomization: draw B random test subsets uniformly without replacement
    * implementation: a random test subset can be constructed by **sampling without replacement**, i.e., draw d times randomly without replacement from {1,2,…,n}
    * d = [10%*n], B = 50 - 500
    * can also be applied to LOO CV
    * possible drawback is that C_k may coincide
* K-fold CV
  * note: random partition is important to prevent missing information
* Properties of different CV- schemes
  * LOO: approx. unbiased for true generalization error; slight bias + high variance (due to high covariance caused by similar training set)
  * Leave-d-out: higher bias than LOO + less variance (even though we average over more highly correlated summands)
  * K-fold: higher bias + unclear variance
* Computational shortcut for some linear fitting operators
  * LOO score
    * when prediction on the training data points can be linearly represented by targets
    * compute the CV score by fitting the original estimator **once** on the **full** dataset (4.5)
    * use GCV (generalized cross-validation) to approximate the (4.5)



## Bootstrap

* Objective: to obtain the full distribution of the estimator since in general we can only otherwise derive aymptotic distribution whereas in other cases we even lack mathematical tools

* We use **empirical distribution** to generate **bootstrap samples** based on which we compute estimators, the process of which repeats many times to get an approximate distribution

* **Bootstrap distribution**: the distribution which arises when resampling with empirical distribution and applying the function g on such a bootstrap sample

* Bootstrap consistence

  * see (5.3)
  * consistency of the bootstrap (usually) implies consistent variance and bias estimation

* Bootstrap confidence interval

  * from "real world" data to Bootstrap world: add * to $\hat\theta$ to replace that with $\hat \theta ^*$, add hat to $\theta$ to replace it with $\hat \theta$

  * two-sided confidence interval:

    ![bootstrap-ci](./img/bootstrap-ci.png)

* Bootstrap estimate of the generalization error

  * 1) generate training samples 2) compute estimator 3) evaluate error averaged on all data 4) repeat 1-3 and average results
  * drawback: in step 3), evaluate on data already used to compute estimator!

* Out-of-bootstrap sample for estimation of the generalization error

  * use only out-of-bootstrap sample to estimate error!
  * the size of out-of-bootstrap sample is approximately 0.368*n

* Double bootstrap

  * much more expensive yet powerful



  ​		
  ​	


​			
​		
​	