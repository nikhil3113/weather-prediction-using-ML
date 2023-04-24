import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Load the weather data
weather = pd.read_csv('mumbai_weather.csv', index_col="DATE")
weather = weather.ffill()
weather.fillna(method='bfill', inplace=True)
weather.index = pd.to_datetime(weather.index)
weather.columns = weather.columns.str.lower()

model = pickle.load(open("model.pkl", "rb"))

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
    tmax = float(request.form['tmax_f'])
    tmin = float(request.form['tmin_f'])
    # prcp = float(request.form['prcp'])

    tmax_c = (tmax - 32) * (5/9)
    tmin_c = (tmin - 32) * (5/9)

    if tmax < tmin:
        return render_template('index.html', error="Error: Maximum Tempertature must be greater than Minimum temperature")
    
    # Make a prediction using the model
    prediction = model.predict([[tmax, tmin]])
    prediction_c = (prediction[0][1] - 32) * (5/9)
    prcp_pred = prediction[0][0]
    prcp_pred = max(prcp_pred, 0)



    if prediction_c > 30 and prcp_pred <1:
        weather = 'HOT AND DRY'
        emoji = '127774'
        image = 'hot_and_dry.png'
    elif prediction_c > 30 and prcp_pred >1:
        weather = "HOT AND HUMID"
        emoji = '127765'
        image = 'hot_and_humid.png'
    elif 20 < prediction_c <30 and prcp_pred <1:
        weather = "PLEASANT"
        emoji = '127780'
        image = 'pleasant.png'
    elif prcp_pred > 5:
        weather = "RAINY "
        emoji = '127784'
        image = 'rainy.png'
    else:
        weather = "COOL"
        emoji = '10052'
        image = 'cool.png'
    # Return the prediction as a response
    return render_template('index.html', prediction="{:.2f} °C ({:.2f} °F)".format(prediction_c,prediction[0][1]), prcp = "{:.2f}".format(prcp_pred),weather=weather, emoji=emoji, image=image)


@app.route('/graph')
def graph():

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=weather.index, y=weather['tavg'], mode='lines', name='tavg'))
    fig.add_trace(go.Scatter(x=weather.index, y=weather['prcp'], mode='markers', name='prcp'))
    fig.update_layout(title='tavg and prcp Graph', xaxis_title='Date', yaxis_title='tavg/prcp')
    graph = fig.to_html(full_html=False)
    return render_template('graph.html', graph=graph)





if __name__ == '__main__':
    app.run(debug=True)
