from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        company = request.form['company']
        type_ = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Compute PPI
        try:
            X_res, Y_res = map(int, resolution.split('x'))
            ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size
        except ZeroDivisionError:
            return render_template('index.html', error="Screen size cannot be zero.", df=df)

        # Create DataFrame for prediction
        input_df = pd.DataFrame([[company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]],
                                columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
                                         'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'OS'])
        try:
            prediction = pipe.predict(input_df)
            predicted_price = int(np.exp(prediction[0]))
            return render_template('index.html', prediction=f"₹{predicted_price}", df=df)
        except Exception as e:
            return render_template('index.html', error=f"Prediction failed: {e}", df=df)

    # GET request
    return render_template('index.html', df=df)

if __name__ == '__main__':
    app.run(debug=True)
