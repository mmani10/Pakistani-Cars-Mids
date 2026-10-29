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
