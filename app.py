import numpy as np
from flask import Flask,render_template,request
from keras.models import load_model

app = Flask(__name__)
model = load_model('my_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_passengers',methods=['POST'])
def predict_passengers():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict([[[ 1.09327802,  0.07223151,  0.64168895]]])
    output = prediction

    return render_template('index.html', prediction_text='Predicted passengers for next week or month $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
