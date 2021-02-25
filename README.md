# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & Classification

## Contents:
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Data Dictionary](#Data-Dictionary)
- [Conclusions](#Conclusions)
- [Limitations and Recommendations](#Limitations-and-Recommendations)


## Problem Statement

1.1 Objective :
The main objective is to estimate or predict the total sales of Walmart retail goods at stores in various locations for the next 28-days. The prediction is based on close to 5 years of historical daily total sales data. This project intended to evaluate and compare the different forecasting methods on the given data. This estimation certainly helps different companies to increase their revenues. SMAPE (symmetric mean absolute percentage error) is used to quantify the accuracy of all forecasts and to compare different forecasting algorithms. A lower SMAPE means higher accuracy.

1.2 Data: 
The total sales of various products sold in the USA, organized in the form of grouped time series.


## Executive Summary

### Hypothesis
Based on the data given some of the factors that may affect sales are:
Day- Customers shopping time and spending mostly depends on the weekend. Many customers may like to shop only at weekends.

Special Events/Holidays: Depending on the events and holidays customers purchasing behavior may change. For holidays like Easter, food sales may go up and for sporting events like Superbowl finals Household item sales may go up.
 
### Exploratory Data Analysis
#### Time series analysis
The time series for all years is plotted to observe the seasonality trend over the years.
It can be seen that sales are very less on some days like Christmas
#### Yearly Sales Trend
It can be seen that total sales are increasing every year. This trend is due to the introduction of new products every year at Walmart. Also, the trend pattern for increase or decrease is almost similar for every year. 
#####  Monthly Sales Trend
It can be seen that total sales are increasing every year. This trend is due to the introduction of new products every year at Walmart. Also, the trend pattern for increase or decrease is almost similar for every year.
It can be observed that the sales were increasing every year and are at a peak in March. After March, there is a decrease in sales till May and plummeted in June recording the lowest sales every year. After, June there is a gradual increase in sales for two months, before dropping further until November.
##### Weekly Sales Trend
As expected the total sales are more during Saturday and Sunday when compared to normal weekdays.
##### Sales trend on Holiday and Special Events:
The sales were highest on SuperBowl sporting events. On the day of the National holidays, sales were low. And sales were consistent on the day of the religious festivals.

### Preprocessing,
**Converting days to dates to analyze the data in a better way**
**Train/Test Split**
Since we need to forecast for 28 days, with 5 years of data. All the data with dates less than or equal to April 24th,2016 is considered as training data. Prediction is made for the 28 days following April 24th,2016.
#### **Modeling**
specify (or build) a model, then fit it to the training data, and finally call predict to generate forecasts for the given forecasting horizon.
#### Steps Involved

Model Training/Cross-Validation
Prediction
SMAPE Error for Comparison of Models



Why SMAPE?
The sMape error rate or symmetrical mean absolute percent error is listed as one of the significant, but uncommon forecast error measurements.
#### Naive model
Predicting the last value.
#### Seasonal Naive model
Predicting the last season.
#### Exponential Smoothing
The previous time steps are exponentially weighted and added up to generate the forecast. The weights decay as we move further backwards in time. 
#### Auto ETS
Exponential Smoothing State Space Model. The methodology is fully automatic. The only required argument for ets is the time series.
#### Auto Arima
ARIMA stands for Auto Regressive Integrated Moving Average. While exponential smoothing models were based on a description of trend and seasonality in data, ARIMA models aim to describe the correlations in the time series. Auto Arima automatically select the best ARIMA model. 
#### Prophet
Prophet is an opensource time series forecasting project by Facebook. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, including holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. It is also supposed to be more robust to missing data and shifts in trend compared to other models.
#### Random Forest Regression:
The Random Forest model is a bagging technique, which uses bootstrap sampling without replacement of every sample to train and it reduces variance while training and prevents from overfitting.

On hyperparameter tuning, we found out max_depth=26, n_estimators=31.

#### Lgbm Regression.
Lgbm Regression is a boosting technique to reduce bias while training the model. It has faster training speed and higher efficiency. It replaces continuous values to discrete bins which result in lower memory usage.
On hyperparameter tuning, we found out learning_rate = 0.071, num_leaves = 12 and min_data_in_leaf = 147.

#### Tuning
In the ReducedRegressionForecaster, both the window_length and strategy arguments are hyper-parameters which we may want to optimise.
SlidingWindowSplitter
We fit the forecaster on the initial window, and then use temporal cross-validation to find the optimal parameter
GridSearchCV, we can tune regressors imported from scikit-learn, in addition to tuning window_length. For example RandomForestRegressor.

#### Ensembling Models
Ensemble methods help improve machine learning results by combining multiple models. Using ensemble methods allows us to produce better predictions compared to a single model. 

#### **Modeling**
1. Worked with 2 classification models - Multinomial Naive Bayes and Logistic Regression, coupled with Count Vectorizer and TD-IDF Vectorizer
2. Employed Pipeline and Grid Search for all combinations of Vectorizers and Model
    - To tune hyperparameters for both vectorizers and models
    - Identify best score and best parameters for each of the model combinations
    - Fit the train data 
    - Score both train and test data to see model performance


### Model Evaluation
From the above graph, we can see that the two smoothing methods: moving average and exponential smoothing are the best-scoring models. Holt linear is not far behind. The remaining models: naive approach, ARIMA, and Prophet are the worst-scoring models. I believe that the accuracy of ARIMA and Prophet can be boosted significantly by tuning the hyperparameters. 




## Data Dictionary

Data dictionary for the final set of features.

| Feature           | Type  | Description                                                                                    |
|-------------------|-------|------------------------------------------------------------------------------------------------|
| subreddit       | object | label or name of the subreddit forum                                                                    |
|title     | object   | title text of the user post  
|selftext       | object   | raw text of the user post                    | 
|name        | object| unique Id of the parent post                               |   
| num_comments    | int | number of cmments per post                                   
| author      | object  | author of the post|



## Conclusions
1) Most sales have a linearly trended sine wave shape, reminiscent of the macroeconomic business cycle.
2) Several non-ML models can be used to forecast time series data. Moving average and exponential smoothing are very good models.
3) ARIMA and Prophet's performance can be boosted with more hyperparamter tuning


## Limitations and Recommendations
1) Some models's performance can be boosted with more hyperparameter tyning.
2) Models can be further evaluated by adding more data.

