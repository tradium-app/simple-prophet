# %% Import Libraries
from fbprophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
import matplotlib.pyplot as plt

plt.style.use('./plot.mplstyle')

# %% read data
df = pd.read_csv("./data/SPY.csv")
df['ds'] = pd.to_datetime(df['startEpochTime'], unit='s')
df['y'] = df['openPrice'].astype('float')

df1 = df[['ds', 'y']]

# %% build model
model = Prophet()
model.fit(df1)


# %% Predict
future = model.make_future_dataframe(periods=30)
future.tail()

forecast = model.predict(future)

fig1 = model.plot(forecast)

model.plot_components(forecast)
