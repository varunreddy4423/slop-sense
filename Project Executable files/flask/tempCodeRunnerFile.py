import pickle
from flask import Flask, render_template, request
import numpy as np 
import pandas as pd

app = Flask(__name__, template_folder='Template')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/predict_page')
def predict():
    return render_template("predict.html")

@app.route('/predict', methods=['GET','POST'])
def prediction():

    rate=float(request.form["Rate"])
    star=float(request.form["Stars"])
    fkms=float(request.form["freekms"])
    cont=float([request.form["Continent"]])[0]
    count=float([request.form["Country"]])[0]
    alt=float(request.form["altitude"])
    easy=float(request.form["easy"])
    intermediate=float(request.form["intermediate"])
    difficult=float(request.form["difficult"])
    lc=float(request.form["lcrating"])
    rel=float(request.form ["reliabilty"])
    orien=float(request.form["orientation"])
    clean=float(request.form["clean"]) 
    env=float(request.form["environment"])
    amen=float(request.form["amenities"])
    beg=float(request.form["beginners"])
    spark=float(request.form["snowpark"])
    strail=float(request.form["skitrail"])
    
    features_values = np.array([[rate, star, fkms, cont, count, alt, easy, intermediate, difficult,lc,
       rel, orien, clean, env, amen, beg, spark, strail ]])
    
    df = pd.DataFrame(features_values, columns= ['rate', 'star', 'fkms', 'cont', 'count', 'alt', 'easy', 'intermediate', 'difficult','lc',
       'rel', 'orien', 'clean', 'env', 'amen', 'beg', 'spark', 'strail'])
    print(df)

    y_pred = model.predict(df)

    result="The Ski resort is "+str(round (y_pred[0] [0],2))+" kms total."
    return render_template("predictionpage.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True , port=5000)