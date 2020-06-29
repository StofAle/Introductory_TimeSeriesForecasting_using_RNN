# Time series forecasting using recurrent neural networks
-- July 2020

## Motivation
#### 1. 


#### <br>

The analysis is presented in 5 separate notebooks:

* **Part 1: initial RNN trial runs w/ clean data**
* **Part 2: systematic comparison btw simple RNN and LSTM model**
* **Part 3: re-evaluating optimal configurations from part 2 on noisy data**
* **Part 4: using real world financial time series data**
* **Part 5: MORE data = better results?**

The project structure is as follows:


```
├── README.md          <- The README 
│
├── images             <- Images from exploratory data analysis as well as model forecasts
│
├── notebooks          <- Jupyter notebooks containing the analysis
│
├── src                <- Source code for use in this project
│   ├── vp_cocktail.py <- ToolBox containing custom functionalities used throughout this analysis

```




# Data
<br> update
* 1. sine wave
* 2. sine wave and noise next to it
* 3. sine wave and noise ontop of it
* 4. iShares MSCI Brazil Index
* 5. iShares MSCI Brazil Index + BRZU - Direxion Daily Brazil Bull 3X Shares

<img src='https://github.com/StofAle/Forecasting_Cocktail_Trends/blob/master/images/02_hist_ts.png' width=600px>


# Approach

#### Model comparison
<br> LSTM vs RNN

<img src='https://github.com/StofAle/Forecasting_Cocktail_Trends/blob/master/images/05_forecast_gimlet.png' width=600px>


# Results

### Backtesting
<br> update

<img src='https://github.com/StofAle/Forecasting_Cocktail_Trends/blob/master/images/07_rmse.png' width=600px>


 
