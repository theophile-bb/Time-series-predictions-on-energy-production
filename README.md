# Time series predictions on energy production

This project focuses on comparing the performances of many time series models in order to predict future values.
In a context of climate and energy crisis, the goal was to try and forecast the energy production values for the 5 to 10 coming years to have a glimbse on the evolution of these energies.

![image](https://github.com/user-attachments/assets/fa61bfb7-77e7-4400-9fd1-ec19291ee48d)

The project is divided into 2 parts :

• Comparison of various time series forecast models to test their efficiency.

• Web Application: a gradio webapp designed as a tool and using a LSTM neural network to predict the energy poduction values in the chsosen number of years (in TWh) .

## Repository Structure
• *Energy production forecast.ipynb*: Comparison of popular time series models with descriptions about their specificities.

• *Energy forecast webapp.ipynb*: Gradio webapp to predict 1 to 10 future obervations with a LSTM and based on a 1-step-ahed forecast method.
(Input : select a country, select an energy type, sequence length, number of obervations to forecast. Output : list of values, plot of the observations)

## Dataset
Dataset : https://github.com/owid/energy-data?tab=readme-ov-file

The dataset is very complete, especially regarding the long running energies such as oil, gas or coal.
It contains a lot of observations related to the energy production and consumption in different countries and different years.

## Methodology
For this project I worked with oil, gas and coal production data for 2 reasons :
1. These are the most used energy sources globally...
2. ... there are therefore more observations available for the study.

The project compiles many time series forecasting models. The models introduced are :

- ARIMA
- ARIMAX
- Prophet (Facebook)
- LSTM
- Chronos (Amazon)

Each model has a dedicated section in the notebook with the functionnal code and explainations on how it works.

## How to use the Webapp ?

1. **Select a Country** from the dropdown list.
2. **Choose an Energy Type** (Oil, Gas, or Coal).
3. **Set the Sequence Length** (how many past years to consider for prediction).
4. **Select Future Steps** (how many years ahead to predict).
5. Click **"Generate Forecast"** to view predictions and a graph of historical + forecasted values.

📊 The output includes:
- A **list of predicted values** for the selected future years.
- A **graph** displaying both historical data and future predictions.
