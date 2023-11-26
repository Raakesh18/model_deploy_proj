from flask import Flask, render_template,request
import pickle
app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl","rb"))
clf = pickle.load(open("models/clf.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method=="POST":
        query = request.form.get('content')
    tokenized_email = cv.transform([query]) # X 
    predictions = clf.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions, query=query)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)