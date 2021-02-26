# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Sales Forecasting

# Contents:
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Preprocessing](#Preprocessing)
- [Modeling](#Modeling)
- [Tuning](#Tuning)
- [Model Evaluation](#Model-Evaluation)
- [Conclusions](#Conclusions)


# Problem Statement

1.1 Objective :
The main objective is to estimate or predict the total sales of Walmart retail goods at stores in various locations for the next 28-days. The prediction is based on 5 years of historical daily total sales data. This project intended to evaluate and compare a number of statistical forecasting algorithms on the given data. SMAPE (symmetric mean absolute percentage error) is used to quantify the accuracy of all forecasts and to compare different forecasting algorithms. A lower SMAPE means higher accuracy.


1.2 Data: 
The total daily sales of various products sold in the USA, organized in the form of time series.


# Executive Summary

## Background
Sales forecasting is essentially involves predicting future sales/profit, based on the sales that your dealership has on order. The purpose of this is to give you an insight into your margins so that you are able to manage your business more efficiently.  This estimation certainly helps different companies to increase their revenues. 

## Hypothesis
Based on the data given some of the factors that may affect sales are:

Day: Customers shopping time and spending mostly depends on the weekend. Many customers may like to shop only at weekends.

Special Events/Holidays: Depending on the events and holidays customers purchasing behavior may change. For holidays like Easter, food sales may go up and for sporting events like Superbowl finals Household item sales may go up.
 
## Exploratory Data Analysis
### Time series analysis
The time series for all years is plotted to observe the seasonality trend over the years.
It can be seen that sales are very less on some days like Christmas
#### Yearly Sales Trend
It can be seen that total sales are increasing every year. This trend could be due to the introduction of new products every year at Walmart. Also, the trend pattern for increase or decrease is almost similar for every year. 
####  Monthly Sales Trend
Since 2012, August rates the highest sales about 1.2M products July either ranks 2nd or 3rd on total sales. At 2016, the sales numbers have increased almost 20%.
#### Weekly Sales Trend
As expected the total sales are more during Saturday and Sunday when compared to normal weekdays.
#### Sales trend on Holiday and Special Events:
The sales happened during sporting event times are slightly more and that happened on National event days are little lower.

## Preprocessing
### Train/Test Split
Since we need to forecast for 28 days, with 5 years of data. All the data with dates less than or equal to April 24th,2016 is considered as training data. Prediction is made for the 28 days following April 24th,2016.

## Modeling
A number of statistical forecasting algorithms from SKtime are used to predict the sales for the given forecasting horizon.  
### Steps Involved
1) Model Training/Cross-Validation
2) Prediction
3) SMAPE Error for Comparison of Models
The sMape error rate or symmetrical mean absolute percent error is listed as one of the significant forecast error measurements.

The Various Model that are used are stated below:
### Naive model
Naive model is the baseline model which predict the last value over the whole forecasting horizon. The SMAPE loss for this model is 0.173581.
### Seasonal Naive model
Seasonal Naive model predicts the last season over the whole whole forecasting horizon. 
### Exponential Smoothing
The previous time steps are exponentially weighted and added up to generate the forecast. The weights decay as we move further backwards in time. 
### Auto ETS
Exponential Smoothing State Space Model. The methodology is fully automatic. The only required argument for ets is the time series.
### Auto Arima
ARIMA stands for Auto Regressive Integrated Moving Average. While exponential smoothing models were based on a description of trend and seasonality in data, ARIMA models aim to describe the correlations in the time series. Auto Arima automatically select the best ARIMA model. 
### BATS
BATS is for Exponential smoothing state space model with Box-Cox transformation, ARMA errors, Trend and Seasonal components. Box Cox transformation is a transformation of a non-normal dependent variables into a normal shape. BATS fits the best performing model. SMAPE loss for BATS is 0.070344.
### TBATS
TBATS stands for Exponential smoothing state space model with Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components. Each seasonality is modeled by a trigonometric representation based on Fourier series. SMAPE loss is 0.072434.

### ThetaForecaster
ThetaForecaster is equivalent to simple exponential smoothing(SES) with drift. SMAPE loss for Theta Forecaster is 0.080372.
### Prophet
Prophet is an opensource time series forecasting project by Facebook. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, including holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet worked best with the current data with the lowest SMAPE loss among all the other models. SMAPE loss for Prophet is 0.031872.
### Random Forest Regression:
The Random Forest model is a bagging technique, which uses bootstrap sampling without replacement of every sample to train and it reduces variance while training and prevents from overfitting. SMAPE loss is 0.060069.
### XGBoost regression
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. SMAPE loss for XGBoost Regressor is 0.067767.
### LGBM Regression.
Light Gradient Boosting Regression is a boosting technique to reduce bias while training the model. It has faster training speed and higher efficiency.  SMAPE loss for LGBM Regressor is 0.060069.
### Ensembler
Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. SMAPE loss for the Ensembler is 0.063478
### Ensembling Models
Ensemble methods help improve machine learning results by combining multiple models. Using ensemble methods allows us to produce better predictions compared to a single model. 

## Tuning
### ReducedRegressionForecaster
ReducedRegressionForecaster was used when SKlearn models are to be used for timeseries analysis. Reduced regression is simply reducing the task of time-series forecasting (extrapolation) to simple tasks of regression and combining the unique solutions to each regression (interpolation) task into a solution for the original problem, using a sliding window.


## Model Comparison

<img src = "Model Comparison.png"/>
From the above graph, we can see that Prophet is the best-scoring mode. Auto ARIMA is the worst model, followed by Naive Method. All the rest of the models have rather close to similiar score in between the above 3 models. SMAPE loss for Auto ARIMA could be improved by doing gridsearch in the various parameters.

# Conclusions
This project predicts the aggregated total number of sales in the USA-based Walmart store for the next 28 days time period. Prediction is made based on historical sales data for 5 years. A number of statistical forecasting algorithms are used for the prediction and and a number of models are compared based on the SMAPE loss score it has. Various algorithms from SKtime library is considered instead of the traditional forecasting algorithms and it is found that the SKtime models are very easy to use for prediction. Various models that are used are below:

1) Naive Approach
2) Seasonal Naive
3) Exponential Smoothing
4) Auto ETS
5) Auto ARIMA
6) BATS
7) TBATS
8) ThetaForecaster
9) RandomForest
10) Prophet
11) XGboost
12) Light Gradient Boost
13) Ensemble

SAMPE losses of each of the above models are calculated. Low SMAPE loss score has better accuracy and the models are compared based on thir SMAPE loss score. Among all the models, prophet performed best on the given data with the SMAPE loss of 0.031872. Auto ARIMA performed worse among the other models with a loss of 0.571929.

A sales forecast helps every business make better business decisions. It helps in overall business planning, budgeting, and risk management. Sales forecasting allows companies to efficiently allocate resources for future growth and manage its cash flow. Thus an accurate sales forecasting is very important for every business to suceed.


