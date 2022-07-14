from flask import request, Flask, render_template

app = Flask(__name__)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle


@app.route('/')
def index():
    return render_template("index.html", result="待计算")


@app.route('/calc', methods=["POST"])
def post_data():
    # ingot rate 成锭率
    i1 = request.form.get("i1")
    i2 = request.form.get("i2")
    i3 = request.form.get("i3")
    i4 = request.form.get("i4")
    i5 = request.form.get("i5")
    i6 = request.form.get("i6")
    i7 = request.form.get("i7")
    i8 = request.form.get("i8")
    i9 = request.form.get("i9")
    i10 = request.form.get("i10")
    i11 = request.form.get("i11")
    i12 = request.form.get("i12")
    i13 = request.form.get("i13")
    i14 = request.form.get("i14")

    with open("ingot.pkl", "rb") as file:
        ingot_model = pickle.load(file)
        xi = [[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14]]
        # print(xi)
        yi = ingot_model.predict(xi)

    # product rate 成材率
    p1 = request.form.get("p1")
    p2 = request.form.get("p2")
    p3 = request.form.get("p3")
    p4 = request.form.get("p4")
    p5 = request.form.get("p5")
    p6 = request.form.get("p6")

    with open("product.pkl", "rb") as file:
        product_model = pickle.load(file)
        xp = [[p1, p2, p3, p4, p5, p6]]
        # print(xp)
        yp = product_model.predict(xp)

    print(yi[0] * yp[0])
    return render_template("index.html", result=yi[0] * yp[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5590)
