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