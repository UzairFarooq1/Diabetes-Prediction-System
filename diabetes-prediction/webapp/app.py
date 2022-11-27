import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    predicted_class = model.predict(features)



# #print(predicted)
    if predicted_class==0:
     print("...drizzle...")
    elif predicted_class==1:
     print("...foggy...")
    elif predicted_class==2: 
      print("...Rainy...")
    elif predicted_class==3: 
      print("...Snowy...")
    elif predicted_class==4: 
     print("...Sunny...")
    else:
     print("...Class out of Range...")
    return render_template("index.html", prediction_text = "Your diabetes status {}".format(predicted_class))

if __name__ == "__main__":
    flask_app.run(debug=True)