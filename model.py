
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# Load the weather data
weather = pd.read_csv('mumbai_weather.csv', index_col="DATE")
weather = weather.ffill()

weather.fillna(method='bfill', inplace=True)
weather.index = pd.to_datetime(weather.index)
weather.columns = weather.columns.str.lower()

# Train a Ridge regression model
reg = Ridge(alpha=0.5)
predictors = ["tmax", "tmin"]
target =["prcp","tavg"]
reg.fit(weather[predictors], weather[target])

pickle.dump(reg, open("model.pkl", 'wb'))

predictions = reg.predict(weather[predictors])
mae = mean_absolute_error(weather[target], predictions)
print("MAE:", mae)