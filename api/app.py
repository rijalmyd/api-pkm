import flask
from cfg import models as model_list
from flask import jsonify, render_template, request

models = model_list.models

app = flask.Flask(__name__)
app.config["debug"] = True

@app.route("/")
def home():
    return render_template("index.html", models=list(models.keys()))

@app.route("/predict", methods=["POST"])
def predict_gold():
    """
    Given the date, predict the gold price for next date
    """
    try:
        model_name = request.form.get("model_name")
        date_given = request.form.get("date")
        model = models[model_name]() # get and initialize the model class from dictionary
        pred = model.predict(date_given)
    except KeyError:
        model_name = request.headers.get("model_name")
        date_given = request.headers.get("date")
        model = models[model_name]() # get and initialize the model class from dictionary
        pred = model.predict(date_given)

    return jsonify(
        {
            "given_date": date_given,
            "next_date": model.get_next_date(date_given),
            "price": pred,
        },
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
