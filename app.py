from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm

app = Flask(__name__)

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(x, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[field]) for field in request.form]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = classifier.predict(std_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
