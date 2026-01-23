## Eric's Feature Store questions

1. how does a DS define a new feature like "total transactions in home state over the last week, for a particular user"? How do spine queries fit into this?

2. do all features need to be defined in the feature platform monorepo? Or can they be defined in training code? But if they are defined in training code, how
how are they versioned
how does the inference endpoint get that same transform logic to run (or in other words when does it get calculated?)

3. Consider the box-cox transform. It's a common preprocessing method for time series data. It's niche in that you would not really see it in SQL. There probably is no DS library over narwhals  to calculate it. Spark ML Lib probably has it. At best, a PARTICULAR SQL ENGINE might support this, but by no means every engine. How would you express niche transforms like this in the feature platform?

4. When are features calculated (for real time)? Are batch jobs caching them in redis? And then does the client query them out of redis.

5. When do historical features (backfill) get computed? How/when do the online features get calculated?

6. Article: https://www.ibm.com/think/tutorials/sktime-multivariate-time-series-forecasting Has an excellent paragraph defining exogenous vs endogenous variables.
    - endogenous variables are a struggle for feature stores because you cannot pre-compute them. Would this be handled by sending parameters to the feature store?


Insights from Betsy:

1. at least for mlforecast, but often in timeseries, you don't save models for the purpose of using them later. You retrain the model EVERY TIME you do a batch inference.

2. you should not inference using unfinished time intervals. Common mistake.

3. Another rookie mistake: future data leakage.

4. Cross validation and evaluations (at least for `mlforecast`) are COUPLED to the library. Or at least: they have have their own eval and cv functions. And example from Eric: XGboost has a feature importance function--not every framework or model family can easily give you that. Takeaway: it is difficult to create a model agnostic:
  - cross validation framework
  - evaluation framework

5. Evaluating is harder when you have LOTS of time series, e.g. 10,000's of products. You can plot a sample, but you cannot look at all of them. When you only have a few time series, you can look at all of them.

## Note on process for a new data science project or feature

1. What will your output look like?
  - what are your primary keys? AKA what uniquely identifies a row on which you would make an inference?
    - example: for a time series, it might be `series-id`-`date-day` (series id and date grain), e.g. `pickup-zone-id`-`date-day`
    - example: for fraud detection, it might be `transaction-id` (since transaction has a many to one relationship with a user). We'd be inferring whether a particular transaction is fraudulent.
    - example: for a product recommendation system, it might be `user-id`-`product-id` (since user has a many to many relationship with product). We'd be inferring whether a particular user would like a particular product.
    - example: for a customer churn prediction, it might be `customer-id` (since customer has a one to one relationship with a user). We'd be inferring whether a particular customer will churn.
  - Primary keys are important because at inference time, we will need to look up the features for whichever keys we want to run inference on.
    - used for caching and looking up pre-computed features
    - makes it clear how to parameterize a batch inference job (a set or range of primary keys)
    - defines the labels. Labels look like the final output. Predictions should be able to be `UNION` (sql) directly onto the labels table.
      - Labels can be joined to features (think of them as dimensions of the label) via the primary keys.

## Deployment

1. You might deploy a model artifact. Package it in training, deploy to prod.
  - But VERY often, you do not. You deploy a set of metadata:
    - hyperparameters
    - feature view query
    - training code
    Imagine committing all of these to a branch:
     - `hyperparameters.yaml`
     - `feature-view-query.sql` or YAML file
     - `train.py` or a Metaflow flow, etc.
    Models would have been trained in an ad-hoc fashion in non-prod. We throw those away (or save them purely for the sake of archiving history and debugging).

    Then we retrain the model using those same data (feature view) and code (train.py) in prod.
  - In highly regulated industries like finance--deciding who to approve a loan for--models often get saved in a registry so that they can undergo an standard, rigid audit process--the code for which is not under the control of the modeler. Even in this scenario, models from non-prod should probably not be used--instead, the model should be retrained in prod and THEN saved for audit and THEN deployed to prod.
2. 