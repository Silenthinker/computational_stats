## Multiple Linear Regression



## Cross-Validation##

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

    ![bootstrap-ci](/Users/Junlin/Documents/ETH/FS2017/Computational statistics/img/bootstrap-ci.png)

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