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
