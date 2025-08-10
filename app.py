from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# Import your detection engine
from malware_detector import EnhancedMalwareDetectionSystem

# -------- CONFIG --------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
MODEL_FILES_PREFIX = "models/enhanced_malware_models"
BINARY_DATASET = "malware_dataset.csv"
MULTICLASS_DATASET = "malware_classification_dataset.csv"
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB
# ------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILES_PREFIX) or ".", exist_ok=True)

# optional: basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autobreak")

# Initialize detector
detector = EnhancedMalwareDetectionSystem()

# --- Register Jinja filter ---
def malware_label_filter(type_id):
    mapping = {
        0: "Worm",
        1: "Trojan",
        2: "Adware",
        3: "Botnet",
        4: "Phishing",
        5: "Ransomware",
        6: "Rootkit",
        7: "Spyware",
        8: "Benign"
    }
    return mapping.get(type_id, f"Type {type_id}")

app.jinja_env.filters["malware_label"] = malware_label_filter



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def try_initialize_system(train_if_missing=True):
    logger.info("Initializing detection system...")
    loaded = detector.load_models(MODEL_FILES_PREFIX)
    if loaded:
        logger.info("Loaded saved models.")
        return True

    if not train_if_missing:
        logger.warning("Models not loaded and training forbidden.")
        return False

    if not (os.path.exists(BINARY_DATASET) and os.path.exists(MULTICLASS_DATASET)):
        logger.warning("Training datasets not found. Models cannot be trained automatically.")
        return False

    logger.info("Training new models from datasets...")
    try:
        binary_df = detector.load_binary_dataset(BINARY_DATASET)
        multiclass_df = detector.load_multiclass_dataset(MULTICLASS_DATASET)
        if binary_df is None or multiclass_df is None:
            logger.error("Failed to load required datasets for training.")
            return False

        X_bin, y_bin, _ = detector.preprocess_binary_data(binary_df)
        X_multi, y_multi, _ = detector.preprocess_multiclass_data(multiclass_df)

        detector.train_binary_classifier(X_bin, y_bin)
        detector.train_multiclass_classifier(X_multi, y_multi)
        detector.save_models(MODEL_FILES_PREFIX)
        logger.info("Training complete and models saved.")
        return True
    except Exception as e:
        logger.exception("Exception during training: %s", e)
        return False

# initialize at startup
try_initialize_system(train_if_missing=True)

# ---------- Routes ----------

@app.route("/")
def index():
    features = detector.binary_feature_columns or []
    return render_template("index.html", feature_columns=features)

@app.route("/health")
def health():
    status = {
        "models_loaded": bool(detector.binary_model),
        "binary_features_count": len(detector.binary_feature_columns),
        "multiclass_features_count": len(detector.multiclass_feature_columns),
        "timestamp": datetime.utcnow().isoformat()
    }
    return jsonify(status)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            flash("No file selected.", "warning")
            return redirect(request.url)

        f = request.files["file"]
        if not allowed_file(f.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(request.url)

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        result = analyze_file(filepath)
        if isinstance(result, dict) and result.get("error"):
            flash(result["error"], "danger")
            return redirect(request.url)

        return render_template(
            "results.html",
            result=result,
            filename=filename,
            models_loaded=bool(detector.binary_model)
        )

    return render_template("upload.html")

@app.route("/manual_input", methods=["GET", "POST"])
def manual_input():
    feature_columns = detector.binary_feature_columns or []
    if request.method == "POST":
        try:
            values = []
            for col in feature_columns:
                raw = request.form.get(col, "")
                val = 0.0 if raw == "" else float(raw)
                values.append(val)
            result = analyze_features(values, "Manual Input")
            if isinstance(result, dict) and result.get("error"):
                flash(result["error"], "danger")
                return redirect(request.url)
            return render_template(
                "results.html",
                result=result,
                filename="Manual Input",
                models_loaded=bool(detector.binary_model)
            )
        except ValueError:
            flash("Invalid numeric input provided.", "danger")
            return redirect(request.url)

    return render_template("manual_input.html", feature_columns=feature_columns)

@app.route("/dashboard")
def dashboard():
    stats = {
        "total_samples_analyzed": 0,
        "malware_detected": 0,
        "benign_samples": 0,
        "penetration_tests": 0,
        "system_status": "Active" if detector and detector.binary_model else "Inactive",
    }
    return render_template("dashboard.html", stats=stats)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" in request.files:
        f = request.files["file"]
        if not allowed_file(f.filename):
            return jsonify({"error": "Only CSV allowed"}), 400
        df = pd.read_csv(f)
        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400
        if all(col in df.columns for col in detector.binary_feature_columns):
            feats = df.iloc[0][detector.binary_feature_columns].values.astype(float)
            dtype = "binary"
        elif all(col in df.columns for col in detector.multiclass_feature_columns):
            feats = df.iloc[0][detector.multiclass_feature_columns].values.astype(float)
            dtype = "multiclass"
        else:
            return jsonify({"error": "CSV missing required feature columns"}), 400
        pred = detector.predict_comprehensive(feats, f.filename, dtype)
        return jsonify(pred)

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    feats = data.get("features")
    dtype = data.get("type", "binary")
    name = data.get("name", "api_input")
    if not isinstance(feats, list) or len(feats) == 0:
        return jsonify({"error": "Invalid or empty features list"}), 400
    try:
        feats = np.array(feats, dtype=float)
    except Exception:
        return jsonify({"error": "Features must be numeric"}), 400
    pred = detector.predict_comprehensive(feats, name, dtype)
    return jsonify(pred)

# helper: analyze CSV on disk
def analyze_file(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {"error": "CSV file is empty."}
        if detector.binary_feature_columns and all(col in df.columns for col in detector.binary_feature_columns):
            features = df.iloc[0][detector.binary_feature_columns].values.astype(float)
            dtype = "binary"
        elif detector.multiclass_feature_columns and all(col in df.columns for col in detector.multiclass_feature_columns):
            features = df.iloc[0][detector.multiclass_feature_columns].values.astype(float)
            dtype = "multiclass"
        else:
            missing = set(detector.binary_feature_columns) - set(df.columns) if detector.binary_feature_columns else set()
            return {"error": f"CSV missing required columns. Missing (example binary): {list(missing)[:10]}"}
        return analyze_features(features, os.path.basename(filepath), dtype)
    except Exception as e:
        logger.exception("Error analyzing file: %s", e)
        return {"error": f"File processing error: {str(e)}"}

def analyze_features(features, sample_name, dataset_type="binary"):
    if not detector.binary_model:
        return {"error": "Models not loaded."}
    try:
        prediction = detector.predict_comprehensive(features, sample_name, dataset_type)
        return {"sample_name": sample_name, "prediction": prediction, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.exception("Analysis error: %s", e)
        return {"error": f"Analysis error: {str(e)}"}

@app.route("/models/list")
def models_list():
    prefix = MODEL_FILES_PREFIX
    files = []
    for ext in ["_binary.pkl", "_multiclass.pkl", "_binary_scaler.pkl", "_multiclass_scaler.pkl", "_label_encoder.pkl", "_metadata.json"]:
        path = f"{prefix}{ext}"
        if os.path.exists(path):
            files.append(os.path.basename(path))
    return jsonify({"models": files})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
