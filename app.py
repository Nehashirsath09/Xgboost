"""
Medical Insurance Cost Predictor
--------------------------------
Flask app that serves an XGBoost regression model (xgboost.pkl) with a
single-page UI. Deploy on Render with:

    Build command : pip install -r requirements.txt
    Start command : gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
"""
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("insurance-app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def find_model_path() -> str:
    """MODEL_PATH env var -> xgboost.pkl -> any *.pkl sitting next to app.py."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path if os.path.isabs(env_path) else os.path.join(BASE_DIR, env_path)
    default = os.path.join(BASE_DIR, "xgboost.pkl")
    if os.path.exists(default):
        return default
    found = sorted(glob.glob(os.path.join(BASE_DIR, "*.pkl")))
    if found:
        return found[0]
    raise FileNotFoundError(
        "No model file found. Put your model next to app.py as 'xgboost.pkl' "
        "(or set the MODEL_PATH environment variable)."
    )


MODEL_PATH = find_model_path()
with open(MODEL_PATH, "rb") as fh:
    MODEL = pickle.load(fh)
log.info("Loaded model from %s (%s)", MODEL_PATH, type(MODEL).__name__)

# The model was trained on these 6 columns, in this order.
_names = getattr(MODEL, "feature_names_in_", None)
FEATURES = (list(_names) if _names is not None
            else ["age", "sex", "bmi", "children", "smoker", "region"])

# Label encodings the model was trained with (verified against known rows of
# the insurance dataset: alphabetical LabelEncoder, smoker no=0 / yes=1).
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

# Range the model has seen during training (used for soft warnings).
TRAIN_RANGE = {"age": (18, 64), "bmi": (15.9, 53.2), "children": (0, 5)}

LABELS = {
    "age": "Age", "sex": "Sex", "bmi": "BMI",
    "children": "Children", "smoker": "Smoking", "region": "Region",
}

app = Flask(__name__)


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
class ValidationError(ValueError):
    pass


def parse_payload(data: dict) -> dict:
    try:
        age = int(float(data.get("age")))
        bmi = float(data.get("bmi"))
        children = int(float(data.get("children")))
    except (TypeError, ValueError):
        raise ValidationError("Age, BMI and children must be numbers.")

    sex = str(data.get("sex", "")).lower()
    smoker = str(data.get("smoker", "")).lower()
    region = str(data.get("region", "")).lower()

    if not 18 <= age <= 100:
        raise ValidationError("Age must be between 18 and 100.")
    if not 10 <= bmi <= 70:
        raise ValidationError("BMI must be between 10 and 70.")
    if not 0 <= children <= 10:
        raise ValidationError("Children must be between 0 and 10.")
    if sex not in SEX_MAP:
        raise ValidationError("Sex must be 'male' or 'female'.")
    if smoker not in SMOKER_MAP:
        raise ValidationError("Smoker must be 'yes' or 'no'.")
    if region not in REGION_MAP:
        raise ValidationError("Please choose a valid region.")

    return {"age": age, "sex": sex, "bmi": round(bmi, 1),
            "children": children, "smoker": smoker, "region": region}


def to_frame(p: dict) -> pd.DataFrame:
    row = {
        "age": p["age"],
        "sex": SEX_MAP[p["sex"]],
        "bmi": p["bmi"],
        "children": p["children"],
        "smoker": SMOKER_MAP[p["smoker"]],
        "region": REGION_MAP[p["region"]],
    }
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)


def explain(frame: pd.DataFrame):
    """Per-feature contribution (in $) using XGBoost's built-in SHAP values."""
    try:
        import xgboost as xgb

        booster = MODEL.get_booster()
        contribs = booster.predict(xgb.DMatrix(frame), pred_contribs=True)[0]
        base = float(contribs[-1])
        items = [
            {"feature": LABELS.get(name, name), "impact": float(val)}
            for name, val in zip(FEATURES, contribs[:-1])
        ]
        items.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return base, items
    except Exception as exc:  # explanation is a bonus; never break the prediction
        log.warning("Could not compute contributions: %s", exc)
        return None, []


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=type(MODEL).__name__, features=FEATURES)


@app.post("/predict")
def predict():
    try:
        payload = parse_payload(request.get_json(silent=True) or {})
        frame = to_frame(payload)
        pred = float(np.asarray(MODEL.predict(frame)).ravel()[0])
        pred = max(pred, 0.0)
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Something went wrong while predicting."), 500

    base, contribs = explain(frame)

    warnings = []
    for key, (lo, hi) in TRAIN_RANGE.items():
        if not lo <= payload[key] <= hi:
            warnings.append(
                f"{LABELS[key]} is outside the {lo}–{hi} range the model was trained on, "
                "so this estimate is less reliable."
            )

    return jsonify(
        prediction=round(pred, 2),
        monthly=round(pred / 12, 2),
        bmi_category=bmi_category(payload["bmi"]),
        baseline=None if base is None else round(base, 2),
        contributions=[{"feature": c["feature"], "impact": round(c["impact"], 2)} for c in contribs],
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Front-end (single self-contained page)
# --------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insurance Cost Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b1020; --card:rgba(255,255,255,.06); --stroke:rgba(255,255,255,.12);
    --text:#eef1ff; --muted:#9aa4c7; --accent:#7c8cff; --accent2:#22d3ee;
    --good:#34d399; --bad:#fb7185; --warn:#fbbf24;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    color:var(--text); min-height:100vh; padding:40px 18px 60px;
    background:
      radial-gradient(900px 500px at 10% -10%, rgba(124,140,255,.25), transparent 60%),
      radial-gradient(800px 500px at 95% 10%, rgba(34,211,238,.18), transparent 60%),
      var(--bg);
  }
  .wrap{max-width:1040px;margin:0 auto}
  header{text-align:center;margin-bottom:34px}
  .badge{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    color:var(--accent2);border:1px solid rgba(34,211,238,.35);padding:6px 12px;border-radius:999px;background:rgba(34,211,238,.08)}
  h1{font-size:clamp(28px,5vw,46px);font-weight:800;margin:16px 0 10px;letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#b4bcff 60%,#7ee8fa);-webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;line-height:1.6}
  .grid{display:grid;grid-template-columns:1.1fr .9fr;gap:22px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--stroke);border-radius:22px;padding:28px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 20px 60px rgba(0,0,0,.35)}
  .card h2{font-size:15px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:22px}
  .field{margin-bottom:22px}
  .field label{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;font-weight:600;margin-bottom:10px}
  .field label span.val{color:var(--accent2);font-weight:700;font-size:15px}
  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:99px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));outline:none;opacity:.9}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;
    background:#fff;border:4px solid var(--accent);cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.4)}
  input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#fff;border:4px solid var(--accent);cursor:pointer}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:520px){.row{grid-template-columns:1fr}}
  .seg{display:flex;background:rgba(255,255,255,.06);border:1px solid var(--stroke);border-radius:12px;padding:4px;gap:4px}
  .seg input{display:none}
  .seg label{flex:1;margin:0;text-align:center;padding:10px 8px;border-radius:9px;cursor:pointer;font-size:14px;
    font-weight:600;color:var(--muted);transition:.2s;display:block}
  .seg input:checked + label{background:linear-gradient(135deg,var(--accent),#5b6cf0);color:#fff;box-shadow:0 4px 14px rgba(124,140,255,.4)}
  .seg.smoke input[value=yes]:checked + label{background:linear-gradient(135deg,#fb7185,#e11d48)}
  select,.num{width:100%;background:rgba(255,255,255,.06);border:1px solid var(--stroke);color:var(--text);
    padding:12px 14px;border-radius:12px;font:inherit;font-size:14px;font-weight:500;outline:none}
  select:focus,.num:focus{border-color:var(--accent)}
  select option{color:#111}
  .stepper{display:flex;align-items:center;gap:10px}
  .stepper button{width:42px;height:42px;border-radius:12px;border:1px solid var(--stroke);background:rgba(255,255,255,.06);
    color:var(--text);font-size:20px;cursor:pointer;transition:.15s}
  .stepper button:hover{background:rgba(255,255,255,.14)}
  .stepper output{flex:1;text-align:center;font-weight:700;font-size:18px}
  .chip{display:inline-block;margin-top:8px;font-size:12px;font-weight:600;padding:4px 10px;border-radius:99px;background:rgba(255,255,255,.08);color:var(--muted)}
  .chip.Healthy{color:var(--good);background:rgba(52,211,153,.12)}
  .chip.Overweight{color:var(--warn);background:rgba(251,191,36,.12)}
  .chip.Obese{color:var(--bad);background:rgba(251,113,133,.12)}
  .chip.Underweight{color:var(--accent2);background:rgba(34,211,238,.12)}
  .btn{width:100%;margin-top:6px;padding:16px;border:0;border-radius:14px;font:inherit;font-size:16px;font-weight:700;color:#fff;
    cursor:pointer;background:linear-gradient(135deg,var(--accent),#5b6cf0 55%,#22b8e0);
    box-shadow:0 10px 30px rgba(91,108,240,.45);transition:transform .15s,box-shadow .15s}
  .btn:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(91,108,240,.55)}
  .btn:disabled{opacity:.6;cursor:wait;transform:none}
  .result{display:flex;flex-direction:column;gap:18px}
  .placeholder{color:var(--muted);text-align:center;padding:60px 10px;line-height:1.7}
  .placeholder .big{font-size:46px;display:block;margin-bottom:8px}
  .price{font-size:clamp(40px,7vw,58px);font-weight:800;letter-spacing:-.03em;
    background:linear-gradient(90deg,#fff,#9fe9ff);-webkit-background-clip:text;background-clip:text;color:transparent}
  .sub{color:var(--muted);font-size:14px;margin-top:2px}
  .sub b{color:var(--text)}
  .bars h3{font-size:13px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin:6px 0 12px}
  .bar{display:grid;grid-template-columns:78px 1fr 84px;align-items:center;gap:10px;margin-bottom:10px;font-size:13px}
  .bar .track{position:relative;height:10px;background:rgba(255,255,255,.07);border-radius:99px}
  .bar .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.25)}
  .bar .fill{position:absolute;top:0;bottom:0;border-radius:99px;transition:width .8s cubic-bezier(.2,.8,.2,1)}
  .bar .amt{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
  .pos{color:var(--bad)} .neg{color:var(--good)}
  .warn{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--warn);padding:10px 14px;border-radius:12px;font-size:13px;line-height:1.5}
  .err{background:rgba(251,113,133,.1);border:1px solid rgba(251,113,133,.35);color:var(--bad);padding:12px 14px;border-radius:12px;font-size:14px}
  .note{color:var(--muted);font-size:12px;line-height:1.6;margin-top:6px}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:30px;line-height:1.7}
  .fade{animation:fade .5s ease}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">XGBoost &middot; Machine Learning</span>
    <h1>Insurance Cost Predictor</h1>
    <p>Estimate yearly medical insurance charges from a few personal details, powered by a gradient-boosted tree model.</p>
  </header>

  <div class="grid">
    <form class="card" id="form" autocomplete="off">
      <h2>Your details</h2>

      <div class="field">
        <label for="age">Age <span class="val" id="ageV">30</span></label>
        <input type="range" id="age" min="18" max="64" value="30">
      </div>

      <div class="field">
        <label for="bmi">BMI <span class="val" id="bmiV">26.0</span></label>
        <input type="range" id="bmi" min="15" max="55" step="0.1" value="26">
        <span class="chip" id="bmiChip">Overweight</span>
      </div>

      <div class="row">
        <div class="field">
          <label>Sex</label>
          <div class="seg">
            <input type="radio" name="sex" id="sexF" value="female" checked><label for="sexF">Female</label>
            <input type="radio" name="sex" id="sexM" value="male"><label for="sexM">Male</label>
          </div>
        </div>
        <div class="field">
          <label>Smoker</label>
          <div class="seg smoke">
            <input type="radio" name="smoker" id="smN" value="no" checked><label for="smN">No</label>
            <input type="radio" name="smoker" id="smY" value="yes"><label for="smY">Yes</label>
          </div>
        </div>
      </div>

      <div class="row">
        <div class="field">
          <label>Children</label>
          <div class="stepper">
            <button type="button" id="minus" aria-label="Fewer children">&minus;</button>
            <output id="kids">0</output>
            <button type="button" id="plus" aria-label="More children">+</button>
          </div>
        </div>
        <div class="field">
          <label for="region">Region</label>
          <select id="region">
            <option value="northeast">Northeast</option>
            <option value="northwest">Northwest</option>
            <option value="southeast">Southeast</option>
            <option value="southwest" selected>Southwest</option>
          </select>
        </div>
      </div>

      <button class="btn" id="go" type="submit">Predict my cost</button>
    </form>

    <section class="card result" id="result">
      <h2>Estimated yearly cost</h2>
      <div class="placeholder"><span class="big">🩺</span>Adjust the details and press <b>Predict</b><br>to see your estimate and what drives it.</div>
    </section>
  </div>

  <footer>
    Estimates are in US dollars and come from a statistical model trained on sample data.<br>
    For illustration only &mdash; not an insurance quote or financial advice.
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = n => '$' + Number(n).toLocaleString('en-US', {minimumFractionDigits:0, maximumFractionDigits:0});
const fmtSigned = n => (n >= 0 ? '+' : '−') + '$' + Math.abs(Math.round(n)).toLocaleString('en-US');
let kids = 0;

function bmiCat(b){ return b < 18.5 ? 'Underweight' : b < 25 ? 'Healthy' : b < 30 ? 'Overweight' : 'Obese'; }
function syncUI(){
  $('ageV').textContent = $('age').value;
  const b = parseFloat($('bmi').value);
  $('bmiV').textContent = b.toFixed(1);
  const c = bmiCat(b);
  $('bmiChip').textContent = c === 'Healthy' ? 'Healthy weight' : c;
  $('bmiChip').className = 'chip ' + c;
  $('kids').textContent = kids;
}
['age','bmi'].forEach(id => $(id).addEventListener('input', syncUI));
$('minus').onclick = () => { kids = Math.max(0, kids - 1); syncUI(); };
$('plus').onclick  = () => { kids = Math.min(5, kids + 1); syncUI(); };
syncUI();

function countUp(el, to){
  const t0 = performance.now(), dur = 900;
  (function step(t){
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(to * e);
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function render(d){
  let html = '<h2>Estimated yearly cost</h2><div class="fade">';
  html += '<div class="price" id="price">$0</div>';
  html += '<div class="sub">&asymp; <b>' + fmt(d.monthly) + '</b> per month &middot; BMI category: <b>' + esc(d.bmi_category) + '</b></div></div>';
  (d.warnings || []).forEach(w => html += '<div class="warn">⚠️ ' + esc(w) + '</div>');
  if (d.contributions && d.contributions.length){
    const max = Math.max(...d.contributions.map(c => Math.abs(c.impact)), 1);
    html += '<div class="bars fade"><h3>What drives this estimate</h3>';
    if (d.baseline != null) html += '<div class="note" style="margin:-4px 0 12px">Starting point (average profile): <b>' + fmt(d.baseline) + '</b></div>';
    d.contributions.forEach(c => {
      const w = Math.min(50, Math.abs(c.impact) / max * 50);
      const pos = c.impact >= 0;
      const style = pos ? 'left:50%;width:' + w + '%;background:linear-gradient(90deg,#fb7185,#f43f5e)'
                        : 'left:' + (50 - w) + '%;width:' + w + '%;background:linear-gradient(90deg,#10b981,#34d399)';
      html += '<div class="bar"><span>' + esc(c.feature) + '</span><div class="track"><div class="mid"></div><div class="fill" style="' + style + '"></div></div>' +
              '<span class="amt ' + (pos ? 'pos' : 'neg') + '">' + fmtSigned(c.impact) + '</span></div>';
    });
    html += '<div class="note">Red pushes the cost up, green pulls it down, relative to the average profile.</div></div>';
  }
  $('result').innerHTML = html;
  countUp($('price'), d.prediction);
}

$('form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('go'); btn.disabled = true; btn.textContent = 'Predicting…';
  const body = {
    age: $('age').value, bmi: $('bmi').value, children: kids,
    sex: document.querySelector('input[name=sex]:checked').value,
    smoker: document.querySelector('input[name=smoker]:checked').value,
    region: $('region').value
  };
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Request failed');
    render(d);
  } catch (err) {
    $('result').innerHTML = '<h2>Estimated yearly cost</h2><div class="err">' + esc(err.message) + '</div>';
  } finally {
    btn.disabled = false; btn.textContent = 'Predict my cost';
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
