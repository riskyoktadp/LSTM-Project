import re, pickle

from flask import Flask, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier


import numpy as np


app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,config=swagger_config)

#sentiment = ['positif', 'netral', 'negatif']

def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    string = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

vectorizer = open('feature.pkl', 'rb')
# max_features = 100000
# tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
#file_sequencer = open('x_pad_sequences.pickle', 'rb')

load_vectorizer = pickle.load(vectorizer)
vectorizer.close()
model = joblib.load('model_neural.pkl')

@app.route('/')
def hello_world():
    return("Hello World")

@swag_from("docs/model_neural.yml", methods=['POST'])
@app.route('/neural_network', methods=['POST'])
def NN():
    
    text = request.form.get('text')
    cleanse_text = [cleansing(text)]

    count_vect = load_vectorizer.transform(cleanse_text)
    prediction = model.predict(count_vect)[0]
    
    json_response = {
        'status_code': 200,
        'description': "Original Teks",
        'data': "prediction"
    }

    response_data = jsonify(json_response)
    return response_data

@app.route('/data',methods=["POST"])
def data_clean():
    """upload files using data valid with the post method.
    ---
    tags:
      - name: upload data file
    parameters:
      - name: file
        in: formData
        type: file
        required: true  
    responses:
        200:
            description: output values
        400:
            description: bad request
        500: 
            description: internal server error            
    """
    df=pd.read_csv(request.files.get("file"))
    print(df.head())
    data=(df)

    return str(data)  
      
if __name__ == '__main__':
    app.run(debug=True)