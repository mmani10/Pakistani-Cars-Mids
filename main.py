"""
CAR PRICE LAB - main.py  (FastAPI backend)

Run:  python -m uvicorn main:app --reload
Open: http://127.0.0.1:8000
(Extra: automatic API documentation page at http://127.0.0.1:8000/docs)
"""
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from train import load_clean_data, TEXT_COLUMNS, NUMBER_COLUMNS

app = FastAPI(title="Car Price Lab")

# Make the folders available to the website (css, js, charts)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

# ---- Load everything ONCE when the server starts ----
saved = joblib.load("models.joblib")            # all trained models
prep = saved["prep"]                            # the preprocessing step
data = load_clean_data()                        # the dataset (CSV)
results = json.load(open("results.json"))       # scores from training

# What the website must send us. FastAPI checks this automatically.
class Car(BaseModel):
    make: str
    model: str
    year: int
    engine: int
    mileage: int
    transmission: str
    fuel: str
    assembly: str
    city: str
    registered: str
    body: str


def most_common(series, n=1):
    return series.value_counts().head(n).index.tolist()


def build_options():
    """Dropdown lists for the website, taken straight from the dataset."""
    cars = {}
    for (make, name), g in data.groupby(["make", "model"]):
        if len(g) < 5:
            continue
        cars.setdefault(make, {})[name] = {
            "count": len(g),
            "year_min": int(g["year"].min()), "year_max": int(g["year"].max()),
            "engines": sorted(int(e) for e in g["engine"].value_counts().head(6).index),
            "typical_engine": int(g["engine"].mode()[0]),
            "typical_mileage": int(g["mileage"].median()),
            "body": most_common(g["body"])[0],
            "transmissions": most_common(g["transmission"], 2),
            "fuels": most_common(g["fuel"], 3),
        }
    makes = sorted(cars, key=lambda m: -sum(v["count"] for v in cars[m].values()))
    cities = data["city"].value_counts()
    regs = data["registered"].value_counts()
    return {
        "makes": makes,
        "cars": {m: dict(sorted(cars[m].items(), key=lambda kv: -kv[1]["count"])) for m in makes},
        "cities": cities[cities >= 30].index.tolist(),
        "registered": regs[regs >= 30].index.tolist(),
    }


OPTIONS = build_options()


# ------------------------------------------------------------------ pages
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/options")
def get_options():
    return OPTIONS


@app.get("/api/results")
def get_results():
    return results


@app.post("/api/predict")
def predict(car: Car):
    # 1) Turn the form into one row and preprocess it (same steps as training)
    row = pd.DataFrame([car.model_dump()])[TEXT_COLUMNS + NUMBER_COLUMNS]
    numbers = prep.transform(row)

    # 2) REGRESSION: every model guesses the price
    prices = {}
    for name, model in saved["regression"].items():
        prices[name] = max(float(np.expm1(model.predict(numbers))[0]), 0)

    # 3) CLASSIFICATION: every model guesses the group
    groups = {}
    for name, model in saved["classification"].items():
        groups[name] = saved["group_names"][int(model.predict(numbers)[0])]

    # 4) Similar real cars from the dataset
    similar = data[(data["make"] == car.make) & (data["model"] == car.model)
                   & (data["year"].between(car.year - 2, car.year + 2))]
    listings = []
    if len(similar):
        close = similar.assign(gap=(similar["mileage"] - car.mileage).abs()).nsmallest(4, "gap")
        listings = close[["year", "mileage", "city", "transmission", "price"]].to_dict("records")

    best = results["best_regression"]
    return {
        "best_model": best,
        "best_price": prices[best],
        "best_mae": results["regression"][best]["mae"],
        "prices": prices,
        "groups": groups,
        "best_group": groups[results["best_classification"]],
        "similar_count": len(similar),
        "similar_median": float(similar["price"].median()) if len(similar) else None,
        "listings": listings,
    }
