# Time series forecasting using recurrent neural networks
-- July 2020

## Motivation
To efficiently forecast time series values, a whole range of models and modules are available. During my time as an Insight Fellow, I've utilized a three component linear additive model (Facebook's prophet: https://facebook.github.io/prophet/) as well as an ARIMA approach (https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html) to generate forecasts on a one-dimensional time series. The results of that project can be found here: https://github.com/StofAle/Forecasting_Cocktail_Trends  (currently private due to NDA constraints). While these approaches were sufficient for the task, I was still curious to see how a more general modelling approach using a neural network approach would do when forecasting time series data. The answer to this question will be explored here. More specifically, I'll setup a simple recurrent neural network (RNN) as well as a long-short-term memory network (LSTM) to initially learn a continuous time series and generate forecasts. This initial stage is used to fine tune the model parameters and fix the network architecture. In a second step, I'll add noise to the time series in two different ways and compare the error metrics of the forecasts for a range of scenarios. With a fixed forecasting horizon, an optimal length for the training data is explored. Finally, I'll apply the tuned configurations to two financial time series (www.etf.com/EWZ and https://www.etf.com/BRZU).


#### <br>

The analysis is presented in 5 separate notebooks:

* **Part 1: initial RNN and LSTM trial runs w/ clean data**
* **Part 2: find optimal configuration in epochs and number of units for simple RNN and LSTM model**
* **Part 3: re-evaluating optimal configurations from part 2 on noisy data**
* **Part 4: choosing length of historical data**
* **Part 5: using real world financial time series data**

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
<br> update
The goal is to predict one week of daily data points, i.e. seven points into the future (disrespecting any daycount or business day conventions), given five years worth of historical data. For the synthetic dataset considered in the first four parts, this implies 1820 observations.
In the first two experiments (**Part 1 & 2**), the training data is a continuous function, a sine wave to be precise. In **Part 3 & 4** random noise with a given variance is added to the input data. Finally, observed time series for two exchange traded funds prices are used in **Part 5**.


## Approach

Initially, the neural network is specified by having two dense layers with 128 units and a 20% dropout layer between them. The activation function is a recitified linear unit. With the training data having one-year length, the following pictures from two sample results illustrate the differences in the quality of the fit from the simple RNN and LSTM:

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_LSTM_1.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_RNN_1.png' width=600px>

Using this setup, the RNN has trouble getting the slope of the data points right. 

The above picture is of course an isolated view on the performance of the two models. Looking at the overall training and validation error as measured through the MSE, the RNN and LSTM models are comparable. 

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_RNN_mse.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/01_LSTM_mse.png' width=600px>

Choosing the mean-square error as a performance indicator, the minimum error as a function of the number of epochs and network units is tuned. Visualizing the magnitude of the MSE as the size of the 'dots' in the plots below, the performance of the two networks as a function of the number of units and number of epochs is evaluated.

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_RNN.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_LSTM.png' width=600px>

update: takeaways from plots

update: resume here

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_RNN.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/02_LSTM.png' width=600px>

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_std_0.5.jpg' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_std_1.0.jpg' width=600px>

<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_training.png' width=600px>
<img src='https://github.com/StofAle/Introductory_TimeSeriesForecasting_using_RNN/blob/master/images/03_LSTM_training.png' width=600px>


## Results





 
