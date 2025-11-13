from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd

app=Flask(__name__)


Gradient_Boosting_Regressor = pickle.load(open('models/GBR_model.pkl', 'rb'))
Light_Gradient_Boosting_Regressor = pickle.load(open('models/LGBR_model.pkl', 'rb'))
Extreme_Gradient_Boosting_Regressor = pickle.load(open('models/XGBoost_model.pkl', 'rb'))
Random_Forest_Regressor = pickle.load(open('models/RandomForest_model.pkl', 'rb'))
le=pickle.load(open('models/LabelEncoder.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # your existing POST code stays here
        
        Age = float(request.form['age'])
        Height = float(request.form['height'])
        Weight = float(request.form['weight'])
        Duration = float(request.form['duration'])
        Body_Temp = float(request.form['body_temp'])
        Heart_Rate = float(request.form['heart_rate'])
        Gender = request.form['gender']
        Gender_encoded = le.transform([Gender])[0]

        df = pd.DataFrame([[Gender_encoded, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]],
                          columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        
        pred = (Gradient_Boosting_Regressor.predict(df)[0] +
                Light_Gradient_Boosting_Regressor.predict(df)[0] +
                Extreme_Gradient_Boosting_Regressor.predict(df)[0] +
                Random_Forest_Regressor.predict(df)[0]) / 4
        pred=float(round(pred,2))


        return render_template('pred.html',pred=pred)

    else:
        return render_template('index.html')

    
if __name__=='__main__':
    app.run(debug=True)
