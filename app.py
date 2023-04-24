import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template

# Load the weather data
weather = pd.read_csv('mumbai_weather.csv', index_col="DATE")
weather = weather.ffill()
weather.fillna(method='bfill', inplace=True)
weather.index = pd.to_datetime(weather.index)
weather.columns = weather.columns.str.lower()

# Train a linear regression model
reg = LinearRegression()
predictors = ["tmax", "tmin", "prcp"]
target = ["tavg"]
reg.fit(weather[predictors], weather[target])

# Save the model to a file
filename = "model.pkl"
pickle.dump(reg, open(filename, 'wb'))

# Create the Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    tmax = float(request.form['tmax'])
    tmin = float(request.form['tmin'])
    prcp = float(request.form['prcp'])
    
    # Make a prediction using the model
    prediction = reg.predict([[tmax, tmin, prcp]])
    
    # Return the prediction as a response
    return render_template('index.html', prediction="The predicted average temperature is {:.2f} °C".format(prediction[0][0]))

if __name__ == '__main__':
    app.run(debug=True)