# Time series forecasting using recurrent neural networks
-- August 2020

## Motivation
To efficiently forecast time series values, a whole range of methods and models are available. During my time as an Insight Fellow, I've utilized a three component linear additive model (Facebook's prophet: https://facebook.github.io/prophet/) as well as an ARIMA approach (https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html) to generate forecasts on a one-dimensional time series. The project results can be found here: https://github.com/StofAle/Forecasting_Cocktail_Trends  (currently private due to NDA constraints). While these approaches were sufficient for the given project scope, I was curious to see how a neural network approach would do when forecasting time series data. The answer to this question will be explored here. More specifically, I'll setup a simple recurrent neural network (RNN) as well as a long-short-term memory network (LSTM) to initially learn a continuous (clean) time series and generate forecasts. In this initial stage, a few of the model hyper-parameters are tuned and the network architecture fixed. In a second step, random noise is added to the time series in two different ways. The performance of the simple RNN is compared to that of the LSTM network for a range of scenarios. With the  forecasting horizon fixed, the optimal amount of historical information used as training data is tuned. Finally, I'll use the tuned simple RNN and LSMT model with the information on how much historical data to use and attempt to forecast two financial time series for a leveraged exchange traded funds (ETF) on Brazilian stocks (www.etf.com/EWZ and https://www.etf.com/BRZU) .

#### <br>

The analysis is presented in 5 separate notebooks:

* **Part 1: initial RNN and LSTM trial runs w/ clean data**
* **Part 2: find optimal configuration in epochs and number of units for simple RNN and LSTM model**
* **Part 3: re-evaluating optimal configurations from part 2 on noisy data**
* **Part 4: choosing length of historical data**
* **Part 5: using real world financial time series data**

Helper functions are implemented in the Kind_of_Blue class, given in the associated .py file in the \src folder.


The project structure is as follows:


```
├── README.md           <- The README 
│
├── images              <- Images from exploratory data analysis as well as model forecasts
│
├── notebooks           <- Jupyter notebooks containing the analysis
│
    ├── 01_So_What.ipynb
    ├── 02_Freddie_Freeloader.ipynb
    ├── 03_Blue_in_Green.ipynb
    ├── 04_All_Blues.ipynb
    ├── 05_Flamenco_Sketches.ipynb
│   
│
├── src                 <- Source code for use in this project
│   ├── Kind_of_Blue.py <- ToolBox containing custom functionalities used throughout this analysis

```




## Data
<br> 
The goal is to predict one week of daily data points, i.e. seven points into the future (disrespecting any daycount or business day conventions), given five years worth of historical data. For the synthetic dataset considered in the first four parts, this implies 1820 observations.


## Approach
<br> 
In the first two experiments (**Part 1 & 2**), the training data is a continuous function, a sine wave to be precise. In **Part 3 & 4** random noise with a given variance is added to the input data. Finally, observed time series for two exchange traded funds prices are used in **Part 5**.


## Results
<br> 

### Clean synthetic data
In a first step, the neural networks are specified by having two dense layers with 128 units and a 20% dropout layer between them. The activation function is a rectified linear unit. With the training data having one-year worth of daily data, the following pictures from two sample results illustrate the differences in the quality of the fit from the simple RNN and LSTM:

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_LSTM_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_RNN_1.png' width=600px>

Using this setup, the RNN has trouble getting the slope of the data points right. 

The above picture is of course an isolated view on the performance of the two models. Looking at the overall training and validation error as measured through the MSE, the RNN and LSTM models are comparable. 

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_RNN_mse.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_LSTM_mse.png' width=600px>

Choosing the mean-square error as a performance indicator, the minimum error as a function of the number of epochs and network units is tuned. Visualizing the magnitude of the MSE as the size of the 'dots' in the plots below, the performance of the two networks as a function of the number of units and number of epochs is evaluated.

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_RNN.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_LSTM.png' width=600px>

From the above plots, the following qualitative statements can be made: 

For the RNN model, the MSE is not monotonic in either the number of units or the number of epochs. This seems rather peculiar: Increasing the capacity of the network by increasing the nunmber of units does not necessarily lead to a smaller error bar in the forecasts for all number of epochs. Having 10 or 50 epochs and increasing the number of units, the error initially goes down and then up. For 30 epochs, the error goes initially up with an increase in the number of units and then down. 

For the LSTM model, the error is found to be roughly invariant under changes to the number of epochs beyond 10 epochs. It decreases with the number of units.

### Noisy synthetic data
Adding noise to the clean data will impact the model performance. By how much? This question is answered by 1) adding noise as a separate input next to the clean data and 2) distorting the clean data by super-imposing it with random noise. In both cases, three scenarios are considered: The random noise is Gaussian with zero mean and standard deviations of 0.1, 1.0 and 10.0. For the first case, the following two inputs are fed into the models. In the second case, the clean input is distorted with varying noise level.

Both network configurations have 128 units and are run on 10 epochs. 

#### clean data with added noise
In this scenario, the following three input data set are fed into the models:
##### standard deviation = 0.1
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_addedNoise_1.png' width=600px>

##### standard deviation = 1.0
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_addedNoise_2.png' width=600px>

##### standard deviation = 10.0
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_addedNoise_3.png' width=600px>
The performance of both models is not impacted by the magnitude of the noise. This is visible in the plots below by noting that the size of the dots stay the same as the standard deviation for the Gaussian noise increases. What is rather peculiar is dependency of the time it takes to train the model with the stardard deviation. For both models, the training time varies by roughly 10% when the standard deviation is scaled from 0.1 to 10.0. While it increases with increasing noise level for the simple RNN, the opposite behavior is observed for the LSTM model: the training time decreases with an increase in noise.
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_RNN_addedNoise_training.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_addedNoise_training.png' width=600px>

#### distored data
In this scenario, the following three input data set are fed into the models:

##### standard deviation = 0.1
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_distorted_1.png' width=600px>

##### standard deviation = 1.0
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_distorted_2.png' width=600px>

##### standard deviation = 10.0
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_distorted_3.png' width=600px>
Similar to the scenario run above, the model performance for various levels of noise is evaluated. For the simple RNN, the results plotted below show a rather weak dependency of the model performance on the noise level; the size of the dots stay roughly constant with increasing standard deviation. For the LSTM model, a dramatical increase in MSE is observed when scaling the standard deviation from 0.1 to 1.0. For a standard deviation of 0.1, the model performance is comparable to that in the previous scenario, where the noise was added as a separate feature. Beyond a value of 1.0, the magnitude of the error is comparable to that of the simple RNN.
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_RNN_distorted_training.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_distorted_training.png' width=600px>

### How much history is needed?

In order to optimize for the amount of historical data to supply to the networks in order have a minimum MSE given the (fixed) forecasting horizon, both models are evaluated under changing length of historical data. Both the clean (sine-wave) data set as well as the noisy (distorted) data set are considered.

The past history lengths is chosen as multiples of 1, 10, 20 and 52 of the the to-be-predicted future length, which is 7 data points. 

#### clean data
For the simple RNN as well as the LSTM model, the performance seems to be roughly invariant under the amount of historical data supplied. 
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/04_RNN_clean.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/04_LSTM_clean.png' width=600px>

#### distorted data
With the standard deviation set to 0.1, the model performance on the distorted data does show a non-trivial dependency on the amount of historical data supplied. With increasing historical information, the model performance of the simple RNN initially improves but progressively gets worst for length beyond ten times the forecasting horizon. For the LSTM, the model performance seems to improve with increasing amounts of historical data. Note the missing results in the plot for the LSTM model output for length at 10 and 20 times the future horizon - here the implementation simply did not put out a number.S  

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/04_RNN_distorted.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/04_LSTM_distorted.png' width=600px>


### ETF stock data
The insights gained from the analysis on the clean and noisy data sets are applied to two time series on exchange traded fund stock data. Data on two ETFs with Brazilian stock underlyings are forecasted; their tickers are EWZ and BRZU. The time historical time windows as well as the forecasting horizon is taken to be the same as in the previous analysis. Similarly, the configuration of the two neural networks is the one that has been optimized in the previous analyses.
A visual inspection on the EWZ time series shows a somewhat seasonal behavior in the time period 2015-2020, with two peaks: one around the turn of the year (Nov-Dec) and one around summer (May-June). (This has motivated to start the series of experiments with a sine-wave and subsequently adding noise to it). The time series for the (leveraged) BRZU ETF shows a similar seasonal behavior as the EWZ. Since its leveraged, it does have a lot more variance though when compared to the non-leveraged EWZ.
A scatter plot as well as seasonal decomposition and autocorrelation plots of the Close prices are shown below. 

#### EWZ
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_close.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_seasonal.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_seasonal.png' width=600px>

#### BRZU
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_close.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_seasonal.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_seasonal.png' width=600px>

Running the simple RNN as well as the LSTM model in the previously optimized configuration (with a seven day forecasting horizon and thirty five days worth of historical data), the model performance as measured through the mean-square error is comparable for both models:

#### EWZ
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_RNN_training.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_LSTM_training.png' width=600px>

#### BRZU
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_RNN_training.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_LSTM_training.png' width=600px>


This analysis is concluded by showing a few forecasts for both data sets. The model output for the EWZ data seems to suffer from a high bias, with the forecasted data points having little variance. A similar situation is visible for the BRZU data set. Especially the LSTM model configuration does systematically seem to overestimate the level of the data while not giving enough variance to the forecasted data points for them appear realistic compared to historical values. These and a few other issues will be the topic of future work... 

#### EWZ
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_RNN_pred_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_RNN_pred_2.png' width=600px>

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_LSTM_pred_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_EWZ_LSTM_pred_2.png' width=600px>

#### BRZU
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_RNN_pred_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_RNN_pred_2.png' width=600px>

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_LSTM_pred_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/05_BRZU_LSTM_pred_2.png' width=600px>
 
