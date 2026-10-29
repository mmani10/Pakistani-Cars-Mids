# Car Price Lab: Machine Learning Models with FastAPI

Predicts the price of a used car in Pakistan using the PakWheels dataset (70,743 cars after cleaning).
Seven kinds of model from the course labs are trained and compared, and a website lets anyone try them.

## Models used

| Lab | Model | Used for |
|---|---|---|
| 04 | Decision Tree | Regression and classification |
| 05 | Naive Bayes | Classification |
| 06 | K-Nearest Neighbors (KNN) | Regression and classification |
| 07 | Artificial Neural Network (ANN) | Regression and classification |
| 08 | Support Vector Machine (SVR / SVC) | Regression and classification |
| - | Linear Regression | Regression |
| - | Logistic Regression | Classification |
| - | Gradient Boosting (extra) | Regression, most accurate |

- **Regression** = predict the exact price (a number).
- **Classification** = predict the price group: Budget (under Rs 19.5 lakh), Mid-range (up to Rs 38 lakh) or Premium.

## Results (on cars the models never saw)

| Regression model | MAE (Rs) | R2 |
|---|---|---|
| Linear Regression | 509,241 | 0.862 |
| Decision Tree | 372,898 | 0.926 |
| KNN | 416,643 | 0.925 |
| Neural Network (ANN) | 380,069 | 0.938 |
| SVM (SVR) | 392,505 | 0.926 |
| **Gradient Boosting** | **313,080** | **0.953** |

| Classification model | Accuracy |
|---|---|
| Logistic Regression | 91.8% |
| Decision Tree | 91.8% |
| Naive Bayes | 58.6% |
| KNN | 89.8% |
| **Neural Network (ANN)** | **92.4%** |
| SVM (SVC) | 91.4% |

Charts are saved in the `charts` folder and shown on the website's "How the models did" tab.

## Setup instructions

**1. Open a terminal inside this project folder** (the folder that contains `main.py`).

**2. Install the libraries (once):**
```bash
pip install -r requirements.txt
```

**3. Train all the models (about 2 minutes):**
```bash
python train.py
```

**4. Start the FastAPI backend:**
```bash
python -m uvicorn main:app --reload
```

**5. Open the website:** http://127.0.0.1:8000

Extra: FastAPI also creates automatic API documentation at http://127.0.0.1:8000/docs

> Mac/Linux: use `python3` and `pip3`. Tip: Python 3.11 to 3.13 is the safest choice for these libraries.

## Project files

| File | Purpose |
|---|---|
| `train.py` | Cleans the data, preprocesses it, trains and tests all models, makes charts, saves models |
| `main.py` | FastAPI backend: loads models and dataset, serves the website and the API |
| `static/index.html` | Website page (HTML) |
| `static/style.css` | Website design (CSS) |
| `static/script.js` | Website behaviour and connection to the backend (JavaScript) |
| `pakwheels_used_car_data_v02.csv` | The dataset |
| `models.joblib`, `results.json`, `charts/` | Created by `train.py` |

## How the pieces connect

1. `train.py` reads the CSV, trains the models, and saves them.
2. `main.py` loads the saved models and the same CSV when the server starts.
3. The page asks `/api/options` for dropdown lists (built from the CSV).
4. When you press the button, `script.js` sends the form as JSON to `/api/predict`.
5. FastAPI checks the data, preprocesses it, asks every model, and returns the answers.
6. The page shows the best price, every model's opinion, and similar real cars.

## Limitations

- The data has asking prices, not final sale prices, and no information about the car's condition.
- Naive Bayes is weak here because it assumes numbers follow a bell curve, which does not suit 0/1 columns.
- SVM models learn from a sample of 8,000 cars because they are slow on big data.

---

# Full source code

## requirements.txt

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
matplotlib
```

## train.py

```python
"""
CAR PRICE LAB - train.py
Trains every model from the lab on the PakWheels dataset and saves them.

Run:  python train.py

Two kinds of problems are solved:
  1) REGRESSION      -> predict the exact price (a number)
  2) CLASSIFICATION  -> predict the price group: Budget / Mid-range / Premium
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # lets us save charts without opening a window
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, ConfusionMatrixDisplay

# the models from the lab
from sklearn.linear_model import LinearRegression, LogisticRegression    # Regression / Classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree   # Decision Tree
from sklearn.naive_bayes import GaussianNB                               # Naive Bayes
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier  # KNN
from sklearn.neural_network import MLPRegressor, MLPClassifier           # ANN
from sklearn.svm import SVR, SVC                                         # SVM
from sklearn.ensemble import HistGradientBoostingRegressor               # extra: best accuracy

DATA_FILE = "pakwheels_used_car_data_v02.csv"
TEXT_COLUMNS = ["city", "assembly", "body", "make", "model", "transmission", "fuel", "registered"]
NUMBER_COLUMNS = ["year", "engine", "mileage"]


# ---------------------------------------------------------------- STEP 1: load + clean
def load_clean_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df = df.dropna(subset=["price", "year", "engine"])            # rows we cannot use
    df = df[df["price"].between(150_000, 60_000_000)]             # remove crazy prices
    df = df[df["engine"].between(600, 6600)]                      # remove crazy engines
    df = df[df["mileage"].between(100, 500_000)]                  # remove crazy mileage
    df["assembly"] = df["assembly"].fillna("Local")               # fill blanks
    df["fuel"] = df["fuel"].fillna("Petrol")
    df["body"] = df["body"].fillna("Unknown")
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def make_preprocessor():
    """Text columns -> 0/1 columns. Number columns -> scaled to a similar size."""
    return ColumnTransformer([
        ("text", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                               sparse_output=False), TEXT_COLUMNS),
        ("numbers", StandardScaler(), NUMBER_COLUMNS),
    ])


# ---------------------------------------------------------------- the models
# (the slow ones, SVM, learn from a smaller sample so they finish in seconds)
REGRESSION_MODELS = {
    "Linear Regression":    LinearRegression(),
    "Decision Tree":        DecisionTreeRegressor(max_depth=20, min_samples_leaf=5, random_state=1),
    "KNN":                  KNeighborsRegressor(n_neighbors=5),
    "Neural Network (ANN)": MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=100, random_state=1),
    "SVM (SVR)":            SVR(C=3),
    "Gradient Boosting":    HistGradientBoostingRegressor(learning_rate=0.08, max_iter=300, max_leaf_nodes=127,
                                                  min_samples_leaf=10, l2_regularization=1.0, max_features=0.5, random_state=1),
}
CLASSIFICATION_MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=500),
    "Decision Tree":        DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=1),
    "Naive Bayes":          GaussianNB(),
    "KNN":                  KNeighborsClassifier(n_neighbors=5),
    "Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=100, random_state=1),
    "SVM (SVC)":            SVC(C=3),
}
SLOW_MODELS = ["SVM (SVR)", "SVM (SVC)"]
SLOW_SAMPLE = 8000


def bar_chart(names, values, title, label, filename, lower_is_better):
    order = np.argsort(values)
    if not lower_is_better:
        order = order[::-1]
    plt.figure(figsize=(8, 4))
    bars = plt.barh([names[i] for i in order][::-1], [values[i] for i in order][::-1], color="#12263A")
    bars[-1].set_color("#FFC93C")           # best model in yellow
    plt.title(title); plt.xlabel(label); plt.tight_layout()
    plt.savefig(f"charts/{filename}", dpi=120); plt.close()


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Cars used: {len(df):,}\n")

    # ---- STEP 2: targets ----
    # Regression target: log of price (prices are very skewed, log makes learning easier)
    y_price = np.log1p(df["price"])
    # Classification target: 3 equal groups by price
    low, high = df["price"].quantile([1 / 3, 2 / 3])
    GROUP_NAMES = ["Budget", "Mid-range", "Premium"]            # 0, 1, 2
    df["price_group"] = np.digitize(df["price"], [low, high])   # gives 0, 1 or 2
    print(f"Budget: under Rs {low:,.0f} | Mid-range: up to Rs {high:,.0f} | Premium: above that\n")

    # ---- STEP 3: split into training (80%) and testing (20%) ----
    X = df[TEXT_COLUMNS + NUMBER_COLUMNS]
    X_train, X_test, yp_train, yp_test, yg_train, yg_test = train_test_split(
        X, y_price, df["price_group"], test_size=0.2, random_state=42)

    # ---- STEP 4: preprocessing (learn on training data only) ----
    prep = make_preprocessor()
    A_train = prep.fit_transform(X_train)
    A_test = prep.transform(X_test)
    print(f"Each car is now {A_train.shape[1]} numbers for the models\n")

    real_prices = np.expm1(yp_test)
    results = {"regression": {}, "classification": {}}
    trained_reg, trained_clf, predictions = {}, {}, {}

    # ---- STEP 5a: REGRESSION ----
    print("REGRESSION (predict the price)")
    for name, model in REGRESSION_MODELS.items():
        n = SLOW_SAMPLE if name in SLOW_MODELS else len(A_train)
        model.fit(A_train[:n], yp_train.iloc[:n])
        pred = np.expm1(model.predict(A_test))
        mae, r2 = mean_absolute_error(real_prices, pred), r2_score(real_prices, pred)
        results["regression"][name] = {"mae": round(mae), "r2": round(r2, 3)}
        trained_reg[name], predictions[name] = model, pred
        print(f"  {name:<22} MAE = Rs {mae:>10,.0f}   R2 = {r2:.3f}")
    best_reg = min(results["regression"], key=lambda k: results["regression"][k]["mae"])
    print(f"  --> Best: {best_reg}\n")

    # ---- STEP 5b: CLASSIFICATION ----
    print("CLASSIFICATION (Budget / Mid-range / Premium)")
    for name, model in CLASSIFICATION_MODELS.items():
        n = SLOW_SAMPLE if name in SLOW_MODELS else len(A_train)
        model.fit(A_train[:n], yg_train.iloc[:n])
        acc = accuracy_score(yg_test, model.predict(A_test))
        results["classification"][name] = {"accuracy": round(acc * 100, 1)}
        trained_clf[name] = model
        print(f"  {name:<22} Accuracy = {acc*100:.1f}%")
    best_clf = max(results["classification"], key=lambda k: results["classification"][k]["accuracy"])
    print(f"  --> Best: {best_clf}\n")

    # ---- STEP 6: charts for the report ----
    names = list(results["regression"])
    bar_chart(names, [results["regression"][n]["mae"] for n in names],
              "Regression: average error (lower is better)", "MAE in Rs", "regression_mae.png", True)
    names = list(results["classification"])
    bar_chart(names, [results["classification"][n]["accuracy"] for n in names],
              "Classification: accuracy (higher is better)", "Accuracy %", "classification_accuracy.png", False)

    plt.figure(figsize=(5, 5))                                   # predicted vs real price
    plt.scatter(real_prices / 1e6, predictions[best_reg] / 1e6, s=3, alpha=0.3, color="#12263A")
    top = 20
    plt.plot([0, top], [0, top], color="#FFC93C"); plt.xlim(0, top); plt.ylim(0, top)
    plt.xlabel("Real price (million Rs)"); plt.ylabel("Predicted price (million Rs)")
    plt.title(f"{best_reg}: predicted vs real"); plt.tight_layout()
    plt.savefig("charts/predicted_vs_real.png", dpi=120); plt.close()

    ConfusionMatrixDisplay.from_estimator(trained_clf[best_clf], A_test, yg_test, display_labels=GROUP_NAMES, cmap="Blues")
    plt.title(f"{best_clf}: confusion matrix"); plt.tight_layout()
    plt.savefig("charts/confusion_matrix.png", dpi=120); plt.close()

    small_tree = DecisionTreeClassifier(max_depth=3, random_state=1).fit(A_train, yg_train)   # small tree to look at
    plt.figure(figsize=(16, 7))
    plot_tree(small_tree, feature_names=list(prep.get_feature_names_out()), class_names=GROUP_NAMES,
              filled=True, fontsize=7, max_depth=3)
    plt.savefig("charts/decision_tree.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- STEP 7: save everything for the website ----
    joblib.dump({"prep": prep, "regression": trained_reg, "classification": trained_clf,
                 "best_regression": best_reg, "best_classification": best_clf, "group_names": GROUP_NAMES,
                 "budget_limit": float(low), "premium_limit": float(high)}, "models.joblib",
                compress=3)       # compress=3 keeps the file small (KNN stores all the cars)
    results["best_regression"], results["best_classification"] = best_reg, best_clf
    results["rows"], results["budget_limit"], results["premium_limit"] = len(df), float(low), float(high)
    json.dump(results, open("results.json", "w"), indent=2)
    print("Saved: models.joblib, results.json and the charts folder")
```

## main.py

```python
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
```

## static/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Car Price Lab</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<div class="wrap">
  <header>
    <h1>What is your used car worth?</h1>
    <p>Choose your car and 6 different machine learning models will estimate its price.</p>
    <nav>
      <button class="tab active" data-tab="predict">Price my car</button>
      <button class="tab" data-tab="results">How the models did</button>
    </nav>
  </header>

  <!-- TAB 1: the form and the answer -->
  <section id="predict" class="panel">
    <div class="grid">
      <form class="card" id="form" autocomplete="off">
        <h2>Your car</h2>
        <div class="row">
          <div><label for="make">Brand</label><select id="make"></select></div>
          <div><label for="model">Model</label><select id="model"></select></div>
        </div>
        <div class="row">
          <div><label for="year">Model year</label><select id="year"></select></div>
          <div><label for="engine">Engine (cc)</label><select id="engine"></select></div>
        </div>
        <div class="row">
          <div><label for="transmission">Gear type</label><select id="transmission"></select></div>
          <div><label for="fuel">Fuel</label><select id="fuel"></select></div>
        </div>
        <div class="row">
          <div><label for="assembly">Assembly</label>
            <select id="assembly"><option>Local</option><option>Imported</option></select></div>
          <div><label for="mileage">Kilometres driven</label><input type="number" id="mileage" min="100" max="500000" step="500"></div>
        </div>
        <div class="row">
          <div><label for="city">City for sale</label><select id="city"></select></div>
          <div><label for="registered">Registered in</label><select id="registered"></select></div>
        </div>
        <button class="go" id="go" type="submit">Get the price</button>
        <p class="error" id="error"></p>
      </form>

      <aside class="card">
        <div class="tag">
          <small>Best estimate</small>
          <div class="price" id="price">Rs --</div>
          <div class="sub" id="sub">Fill in the form and press the button</div>
        </div>
        <div id="answer"></div>
      </aside>
    </div>
  </section>

  <!-- TAB 2: scores and charts -->
  <section id="results" class="panel hidden">
    <div class="card"><div id="resultsBox">Loading...</div></div>
  </section>
</div>
<script src="/static/script.js"></script>
</body>
</html>
```

## static/style.css

```css
:root{--bg:#F3F5F8;--card:#fff;--ink:#12263A;--muted:#5B6B7B;--line:#DCE2E9;--accent:#FFC93C;--field:#F8FAFC}
@media(prefers-color-scheme:dark){:root{--bg:#0D1722;--card:#15222F;--ink:#EAF0F6;--muted:#9AAABB;--line:#26384A;--field:#101C28}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,Roboto,sans-serif;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:26px 18px 60px}
h1{font-size:clamp(1.8rem,4.5vw,2.6rem);margin:0 0 8px;line-height:1.1}
header p{color:var(--muted);margin:0 0 18px}
h2{font-size:1.15rem;margin:0 0 14px}
h3{margin:22px 0 8px;font-size:1rem}
nav{display:flex;gap:8px;margin-bottom:20px}
.tab{padding:10px 18px;border-radius:999px;border:1.5px solid var(--line);background:var(--card);color:var(--ink);font:inherit;font-weight:600;cursor:pointer}
.tab.active{background:var(--ink);color:var(--card);border-color:var(--ink)}
.hidden{display:none}
.grid{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:520px){.row{grid-template-columns:1fr}}
label{display:block;font-size:.85rem;font-weight:600;color:var(--muted);margin-bottom:5px}
select,input{width:100%;padding:11px 12px;border-radius:10px;border:1.5px solid var(--line);background:var(--field);color:var(--ink);font:inherit}
select:focus,input:focus,button:focus-visible{outline:3px solid var(--accent)}
.go{width:100%;margin-top:6px;padding:15px;border:0;border-radius:12px;background:var(--accent);color:#2B2100;font:inherit;font-weight:800;font-size:1.05rem;cursor:pointer}
.go:disabled{opacity:.6}
.error{color:#C0392B;font-weight:600;min-height:1.2em}
.tag{background:var(--ink);color:var(--card);border-radius:12px;padding:20px;margin-bottom:6px}
.tag small{opacity:.8}
.price{font-size:clamp(2rem,5vw,2.8rem);font-weight:800;margin:4px 0}
.sub{opacity:.85;font-size:.92rem}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600}
tr.best td{font-weight:800}
.chip{display:inline-block;padding:3px 10px;border-radius:999px;background:var(--accent);color:#2B2100;font-weight:700;font-size:.82rem}
.note{color:var(--muted);font-size:.88rem}
img{max-width:100%;border-radius:10px;border:1px solid var(--line);background:#fff}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.charts{grid-template-columns:1fr}}
.scroll{overflow-x:auto}
```

## static/script.js

```javascript
// ---------- small helpers ----------
const $ = id => document.getElementById(id);
const lakh = n => n >= 1e7 ? "Rs " + (n / 1e7).toFixed(2) + " crore" : "Rs " + (n / 1e5).toFixed(2) + " lakh";
let OPTIONS = null;

// plain-English description of each model (shown on the results tab)
const ABOUT = {
  "Linear Regression": "Draws the best straight line through the data.",
  "Logistic Regression": "Like linear regression, but for choosing a group.",
  "Decision Tree": "Asks a series of if-else questions, like a flowchart.",
  "Naive Bayes": "Uses probability and treats every feature as independent.",
  "KNN": "Looks at the 5 most similar cars and copies their answer.",
  "Neural Network (ANN)": "Layers of connected neurons that learn patterns.",
  "SVM (SVR)": "Finds the best line (boundary) that fits the data.",
  "SVM (SVC)": "Finds the best boundary between the groups.",
  "Gradient Boosting": "Many small trees, each fixing the mistakes of the last."
};

function fill(select, items, chosen) {
  select.innerHTML = items.map(v => `<option value="${v}">${v}</option>`).join("");
  if (chosen !== undefined && items.includes(chosen)) select.value = chosen;
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("hidden", p.id !== btn.dataset.tab));
}));

// ---------- dropdowns (filled from the dataset through the backend) ----------
let current = null;
function onMake() {
  fill($("model"), Object.keys(OPTIONS.cars[$("make").value]));
  onModel();
}
function onModel() {
  current = OPTIONS.cars[$("make").value][$("model").value];
  const years = [];
  for (let y = current.year_max; y >= current.year_min; y--) years.push(y);
  fill($("year"), years, Math.min(2015, current.year_max));
  fill($("engine"), current.engines);
  $("engine").value = current.typical_engine;
  fill($("transmission"), current.transmissions);
  fill($("fuel"), current.fuels);
  $("mileage").value = current.typical_mileage;
}

async function start() {
  OPTIONS = await (await fetch("/api/options")).json();
  fill($("make"), OPTIONS.makes, "Toyota");
  fill($("city"), OPTIONS.cities, "Lahore");
  fill($("registered"), OPTIONS.registered, "Lahore");
  onMake();
  if (OPTIONS.cars.Toyota && OPTIONS.cars.Toyota.Corolla) { $("model").value = "Corolla"; onModel(); }
  loadResults();
}
$("make").addEventListener("change", onMake);
$("model").addEventListener("change", onModel);

// ---------- send the form to the FastAPI backend ----------
$("form").addEventListener("submit", async event => {
  event.preventDefault();
  $("error").textContent = "";
  $("go").disabled = true;
  $("go").textContent = "Working...";

  const car = {
    make: $("make").value, model: $("model").value,
    year: +$("year").value, engine: +$("engine").value, mileage: +$("mileage").value,
    transmission: $("transmission").value, fuel: $("fuel").value,
    assembly: $("assembly").value, city: $("city").value,
    registered: $("registered").value, body: current.body
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(car)
    });
    if (!response.ok) throw new Error("Please check that every field is filled in correctly.");
    showAnswer(await response.json(), car);
  } catch (e) {
    $("error").textContent = e.message;
  }
  $("go").disabled = false;
  $("go").textContent = "Get the price";
});

function showAnswer(d, car) {
  $("price").textContent = lakh(d.best_price);
  $("sub").textContent = `${car.year} ${car.make} ${car.model} | best model: ${d.best_model}`;

  let html = `<h3>Price group: <span class="chip">${d.best_group}</span></h3>
    <h3>What each regression model says</h3>
    <table><tr><th>Model</th><th>Predicted price</th></tr>` +
    Object.entries(d.prices).map(([name, p]) =>
      `<tr class="${name === d.best_model ? "best" : ""}"><td>${name}</td><td>${lakh(p)}</td></tr>`).join("") +
    `</table>
    <h3>What each classification model says</h3>
    <table><tr><th>Model</th><th>Price group</th></tr>` +
    Object.entries(d.groups).map(([name, g]) => `<tr><td>${name}</td><td>${g}</td></tr>`).join("") + `</table>`;

  if (d.similar_count) {
    html += `<h3>${d.similar_count} similar cars in the dataset</h3>
      <p class="note">Typical asking price: <b>${lakh(d.similar_median)}</b></p>
      <table><tr><th>Year</th><th>Km</th><th>City</th><th>Price</th></tr>` +
      d.listings.map(l => `<tr><td>${l.year}</td><td>${l.mileage.toLocaleString()}</td><td>${l.city}</td><td>${lakh(l.price)}</td></tr>`).join("") + `</table>`;
  }
  $("answer").innerHTML = html;
}

// ---------- results tab ----------
async function loadResults() {
  const r = await (await fetch("/api/results")).json();
  const reg = Object.entries(r.regression).map(([n, s]) =>
    `<tr class="${n === r.best_regression ? "best" : ""}"><td>${n}</td><td>${ABOUT[n] || ""}</td><td>Rs ${s.mae.toLocaleString()}</td><td>${s.r2}</td></tr>`).join("");
  const clf = Object.entries(r.classification).map(([n, s]) =>
    `<tr class="${n === r.best_classification ? "best" : ""}"><td>${n}</td><td>${ABOUT[n] || ""}</td><td>${s.accuracy}%</td></tr>`).join("");

  $("resultsBox").innerHTML = `
    <h2>Regression: predict the exact price</h2>
    <p class="note">Tested on ${Math.round(r.rows * 0.2).toLocaleString()} cars the models had never seen. MAE = average mistake in rupees (lower is better).</p>
    <div class="scroll"><table><tr><th>Model</th><th>How it works</th><th>MAE</th><th>R2</th></tr>${reg}</table></div>
    <h2 style="margin-top:28px">Classification: predict the price group</h2>
    <p class="note">Budget: under ${lakh(r.budget_limit)} | Mid-range: up to ${lakh(r.premium_limit)} | Premium: above that.</p>
    <div class="scroll"><table><tr><th>Model</th><th>How it works</th><th>Accuracy</th></tr>${clf}</table></div>
    <h3>Charts</h3>
    <div class="charts">
      <img src="/charts/regression_mae.png" alt="Regression error chart">
      <img src="/charts/classification_accuracy.png" alt="Classification accuracy chart">
      <img src="/charts/predicted_vs_real.png" alt="Predicted versus real price">
      <img src="/charts/confusion_matrix.png" alt="Confusion matrix">
    </div>
    <h3>A small decision tree (first 3 questions)</h3>
    <img src="/charts/decision_tree.png" alt="Decision tree">`;
}

start();
```
