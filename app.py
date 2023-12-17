from flask import Flask, request, jsonify
from flask import render_template,redirect,url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["GET","POST"])
def predict():
    
    if request.method == "POST":

        location = request.form['city']
        day = request.form['dateInput']
        month = day[-2:].upper()
        df = pd.read_csv('database.csv')
        for i in range(len(df)):
            if location == df['District'][i]:
                prediction1 = df['Elevation'][i]
                #print(prediction1)
                break
        
        for i in range(len(df)):
            if month == '01' and location == df['District'][i]:
                prediction2 = df['JAN'][i]
                break
            elif month == '02' and location == df['District'][i]:
                prediction2 = df['FEB'][i]
                break
            elif month == '03' and location == df['District'][i]:
                prediction2 = df['MAR'][i]
                break
            elif month == '04' and location == df['District'][i]:
                prediction2 = df['APR'][i]
                break
            elif month == '05' and location == df['District'][i]:
                prediction2 = df['MAY'][i]
                break
            elif month == '06' and location == df['District'][i]:
                prediction2 = df['JUN'][i]
                break
            elif month == '07' and location == df['District'][i]:
                prediction2 = df['JUL'][i]
                break
            elif month == '08' and location == df['District'][i]:
                prediction2 = df['AUG'][i]
                break
            elif month == '09' and location == df['District'][i]:
                prediction2 = df['SEP'][i]
                break
            elif month == '10' and location == df['District'][i]:
                prediction2 = df['OCT'][i]
                break
            elif month == '11' and location == df['District'][i]:
                prediction2 = df['NOV'][i]
                break
            elif month == '12' and location == df['District'][i]:
                prediction2 = df['DEC'][i]
                break
        prediction3 = get_range(prediction2)
        
        ff = pd.read_csv('kmeans_flood_clusters.csv')
        df1 = ff.copy()
        df1 = df1.drop("Unnamed: 0",axis=1)
        X = df1[['JUL', 'Elevation']]
        y = df1['flood']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=23)
        # Standardize features (important for SVMs)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Apply regularization using SVM with the 'C' parameter
        # Adjust the value of C based on your regularization needs
        C_value = 1.0  # You can experiment with different values
        model = SVC(C=C_value, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Example new input
        new_input = [[prediction2, prediction1]]

        # Standardize the new input using the same scaler
        new_input_scaled = scaler.transform(new_input)

        # Predict the flood category
        predicted_flood_category = model.predict(new_input_scaled)
                
        
        #model = SVC(random_state=42)
        #model.fit(X_train, y_train)
        #new_input = [[prediction2, prediction1]]
        #predicted_flood_category = model.predict(new_input)
        #print(prediction2,prediction1,predicted_flood_category)
        if predicted_flood_category == 0:
            floodp = "No Flood"
            pfc = 0
        elif predicted_flood_category == 1:
            floodp = "High Flood"
            pfc = 1
        elif predicted_flood_category == 2:
            floodp = "Moderate Flood"
            pfc = 2
        elif predicted_flood_category == 3:
            floodp = "Very High Flood"
            pfc = 3
        elif predicted_flood_category == 4:
            floodp = "Low Flood"
            pfc = 4
        df2 = pd.read_csv('whole_db.csv')
        for i in range(len(df2)):
            if location == df2['District'][i]:
                prediction4 = df2['Landuse'][i]
                break
        for i in range(len(df2)):
            if location == df2['District'][i]:
                prediction5 = df2['Population'][i]
                #print(prediction5)
                break
        for i in range(len(df2)):
            if location == df2['District'][i]:
                prediction6 = df2['Area'][i]
                break
        if prediction4 == "Mix":
            prediction4 = "Agricultural & Industrial"
            p4 = 1
        else:
            p4 = 0


        ff1 = pd.read_csv('kmeans_destruction_clusters.csv')
        df3 = ff1.copy()
        df3 = df3.drop("Unnamed: 0",axis=1)
        X = df3[['Population', 'Area','flood','Landuse']]
        y = df3['destruction']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
        # Standardize features (important for SVMs)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Apply regularization using SVM with the 'C' parameter
        # Adjust the value of C based on your regularization needs
        C_value = 1.0  # You can experiment with different values
        model = SVC(C=C_value, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Example new input
        new_input = [[prediction5, prediction6,pfc,p4]]

        # Standardize the new input using the same scaler
        new_input_scaled = scaler.transform(new_input)

        # Predict the flood category
        predicted_desc_category = model.predict(new_input_scaled)
        #model = SVC(random_state=42)
        #model.fit(X_train, y_train)
        #new_input = [[prediction5, prediction6,pfc,p4]]
        #predicted_desc_category = model.predict(new_input)
        if predicted_desc_category == 0:
            descp = "Low Destruction"
        elif predicted_desc_category == 1 :
            descp = "High Destruction"
        elif predicted_desc_category == 2:
            descp = "Medium Destruction"
#####################################################################################
        
       
        if floodp == "Low Flood" and descp == "High Destruction":
            descp = "Low Destruction" 

        elif floodp == "Low Flood" and descp == "Medium Destruction":
            descp = "Low Destruction" 

        elif floodp == "No Flood" :
            descp = "No Destruction" 

        elif floodp == "Moderate Flood" and descp == "High Destruction":
            descp = "Low Destruction" 
        
         
        
        return redirect(url_for("output", elevation=prediction1, rainfall_predicted=prediction3, flood=floodp, landuse = prediction4,population = prediction5,area = prediction6,destruction = descp))

@app.route("/output", methods=["GET", "POST"])
def output():
    prediction1 = request.args.get("elevation")
    prediction3 = request.args.get("rainfall_predicted")
    floodp = request.args.get("flood")
    prediction4 = request.args.get("landuse")
    prediction5 = request.args.get("population")
    prediction6 = request.args.get("area")
    descp = request.args.get("destruction")
    return render_template("output1.html", elevation=prediction1, rainfall_predicted=prediction3, flood=floodp, landuse = prediction4,population = prediction5,area = prediction6,destruction = descp)

def get_range(value):
    if value < 1:
        lower_bound = max(round(value - 0.5, 1), 0)
        upper_bound = round(lower_bound + 1, 1)
    elif value <= 100:
        magnitude = 30
        lower_bound = max(int(value // magnitude) * magnitude, 0)
        upper_bound = min(lower_bound + magnitude, 100)
    else:
        magnitude = 50
        lower_bound = max(int(value // magnitude) * magnitude, 0)
        upper_bound = min(lower_bound + magnitude, value + magnitude)

    return f"{lower_bound} to {upper_bound}"

if __name__ == "__main__":
    app.run(port=5050,debug=True)