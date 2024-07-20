from flask import Flask, render_template, request
import pickle
import pandas as pd
import xgboost

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template('index.html')

@app.route('/y_predict', methods=['POST'])
def prediction():
    data = {
        "clonesize": float(request.form["clonesize"]),
        "honeybee": float(request.form["honeybee"]),
        "bumbles": float(request.form["bumbles"]),
        "andrena": float(request.form["andrena"]),
        "osmia": float(request.form["osmia"]),
        "MaxOfUpperTRange": float(request.form["MaxOfUpperTRange"]),
        "MinOfUpperTRange": float(request.form["MinOfUpperTRange"]),
        "AverageOfUpperTRange": float(request.form["AverageOfUpperTRange"]),
        "MaxOfLowerTRange": float(request.form["MaxOfLowerTRange"]),
        "MinOfLowerTRange": float(request.form["MinOfLowerTRange"]),
        "AverageOfLowerTRange": float(request.form["AverageOfLowerTRange"]),
        "RainingDays": float(request.form["RainingDays"]),
        "AverageRainingDays": float(request.form["AverageRainingDays"]),
    }

    names=['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
       'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
       'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
       'RainingDays', 'AverageRainingDays']
    
    data_df = pd.DataFrame([data])
    data_df=pd.DataFrame(data_df, columns=names)
    data_scaled = scaler.transform(data_df)
    data_pred=pd.DataFrame(data_scaled, columns=names)
    prediction = model.predict(data_pred)[0]

    return render_template('index.html', prediction_text=f"Predicted Yield: {prediction:.2f} kg/ha")

if __name__ == "__main__":
    app.run(debug=False)
