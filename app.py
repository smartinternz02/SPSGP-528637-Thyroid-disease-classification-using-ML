from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open(r"C:\Users\J.NAGA NANDINI DEVI\OneDrive\Desktop\reference\thyroid_2_model.pkl", "rb"))
le = pickle.load(open(r"C:\Users\J.NAGA NANDINI DEVI\OneDrive\Desktop\reference\label_encoder2.pkl", "rb"))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        return redirect(url_for('submit', **request.form))
    return render_template('predict.html')

@app.route("/submit", methods=['POST'])
def submit():
    x = [[float(x) for x in request.form.values()]]
    col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    x_df = pd.DataFrame(x, columns=col)

    pred = model.predict(x_df)
    pred_label = le.inverse_transform(pred)[0]

    return render_template('submit.html', prediction_text=pred_label)

if __name__ == "__main__":
    app.run(debug=False)
