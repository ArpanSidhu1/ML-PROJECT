import numpy as np
from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__,template_folder='template')

pipe = pickle.load(open('D:/Linear1.pkl','rb'))

df = pd.read_csv("D:\PROJECT - 2 (HOUSE PRICE)\housing.csv")

@app.route('/')

def index():
    
    locations = sorted(df['Locality'].unique())
    return render_template('Index.html',locations=locations)

def home():
    return render_template('Index.html')


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    
    sqft = float(sqft)
    
    # Construct the DataFrame with columns in the expected order and names
    input_data = pd.DataFrame([[sqft, bhk, bath, location]],columns=['Area','BHK','Bathroom','Locality'])

    print(location,bhk,sqft,bath)    
    try:
        prediction = pipe.predict(input_data)[0]
        return str(np.round(prediction,2))
    except Exception as e:
        print(e)  # Print the exception for debugging purposes
        return "Error predicting price."



if __name__=='__main__':
    app.run()

