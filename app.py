from flask import Flask, render_template, request, send_file, jsonify
import os
import io
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if api_key and GEMINI_AVAILABLE:
    genai.configure(api_key=api_key)
    print("✨ Gemini AI: Connected & Ready")
else:
    print("⚠️ Gemini AI: Key missing (GOOGLE_API_KEY). Internal ML only.")

# Global state to keep the last uploaded dataframe in memory
df = None
# Global variable to store latest prediction for the report
latest_prediction = None


# -----------------------------
# Data Cleaning (Optimized for Speed)
# -----------------------------
def clean_data(df):
    try:
        # 1. Quick Drop
        df = df.dropna(axis=1, how="all")
        df = df.dropna(how="all")
        df = df.drop_duplicates()

        # 2. Normalize Headlines
        df.columns = [str(c).lower().strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

        # 3. Smart Type Conversion (Sequential pd.to_numeric is slow)
        # Only attempt on columns that aren't already identified as numeric but might be
        potential_numeric = df.select_dtypes(include=['object']).columns
        for col in potential_numeric:
            # Sample check: if first 5 non-null values aren't numeric-ish, skip
            sample = df[col].dropna().head(5).astype(str)
            if sample.str.match(r'^-?\d*\.?\d*$').all():
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 4. Vectorized Imputation
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean().fillna(0))
        
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_res = df[col].mode()
                df[col] = df[col].fillna(mode_res[0] if not mode_res.empty else "Unknown")
        
        return df
    except Exception as e:
        print(f"Error in clean_data: {e}")
        return df

# -----------------------------
# Data Analysis (Performance Focused)
# -----------------------------
def analyze_data(original_df, cleaned_df):
    analysis = {}

    analysis["before"] = {
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "missing_total": int(original_df.isnull().sum().sum()),
    }

    analysis["after"] = {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "missing_total": int(cleaned_df.isnull().sum().sum()),
    }

    analysis["missing_by_column"] = {
        k: int(v) for k, v in cleaned_df.isnull().sum().head(20).to_dict().items()
    }

    # Column separation
    cat_cols = cleaned_df.select_dtypes(include="object").columns.tolist()
    num_cols = cleaned_df.select_dtypes(include="number").columns.tolist()

    # Generate Chart Data (Limit to avoid payload/processing bloat)
    analysis["charts"] = []
    
    # 1. Categorical Analysis
    for i, col in enumerate(cat_cols[:3]): 
        try:
            counts = cleaned_df[col].value_counts().head(8)
            chart_type = "doughnut" if i == 1 else "bar"
            
            analysis["charts"].append({
                "type": chart_type,
                "id": f"catChart_{i}",
                "title": f"{col.replace('_', ' ').title()} Split",
                "labels": [str(x) for x in counts.index.tolist()],
                "data": [float(x) for x in counts.values.tolist()],
                "color": "multi" if chart_type == "doughnut" else "gradient",
                "grid": "half"
            })
        except: continue

    # 2. Numerical Analysis
    for i, col in enumerate(num_cols[:2]):
        try:
            data_to_plot = cleaned_df[col].dropna()
            if not data_to_plot.empty:
                hist, bin_edges = np.histogram(data_to_plot, bins=8)
                labels = [f"{float(bin_edges[j]):.1f}" for j in range(len(hist))]
                
                analysis["charts"].append({
                    "type": "line",
                    "id": f"numChart_{i}",
                    "title": f"{col.replace('_', ' ').title()} Trend",
                    "labels": labels,
                    "data": [float(x) for x in hist.tolist()],
                    "color": "solid",
                    "grid": "half"
                })
        except: continue

    # 3. Correlation Heatmap (CRITICAL: Limit to top 15 numeric cols)
    # Correlation is O(N^2) - 500+ cols will fail 504 timeout on free tier
    if len(num_cols) > 1:
        try:
            target_cols = num_cols[:15] # Limit heatmap size for speed
            corr = cleaned_df[target_cols].corr().round(2).fillna(0)
            analysis["heatmap"] = {
                "columns": [str(c) for c in corr.columns],
                "data": corr.values.tolist()
            }
        except:
            analysis["heatmap"] = None
    else:
        analysis["heatmap"] = None

    # 4. Smart Insights
    try:
        # Use limited numeric cols for insights speed if too many
        sub_num = num_cols[:10]
        analysis["insights"] = generate_insights(cleaned_df, sub_num, cat_cols[:10], 
                                               cleaned_df[sub_num].corr() if len(sub_num) > 1 else None)
    except:
        analysis["insights"] = []
    
    analysis["columns"] = [str(c) for c in cleaned_df.columns.tolist()]
    return analysis

# -----------------------------
# Insight Generation
# -----------------------------
def generate_insights(df, num_cols, cat_cols, corr_matrix):
    insights = []
    
    # --- 1. Correlation Insights (Priority High) ---
    if corr_matrix is not None:
        c = corr_matrix.abs()
        np.fill_diagonal(c.values, 0)
        # Find top pairs
        pairs = c.unstack().sort_values(ascending=False).drop_duplicates()
        
        # Take top 2 unique strong correlations
        count = 0
        for (col1, col2), score in pairs.items():
            if score > 0.7 and count < 2:
                raw_score = corr_matrix.loc[col1, col2]
                relation = "positive" if raw_score > 0 else "negative"
                insights.append({
                    "icon": "fa-link",
                    "color": "text-blue-400",
                    "text": f"Strong {relation} correlation ({raw_score}) between <em>{col1}</em> and <em>{col2}</em>."
                })
                count += 1

    # --- 2. Significant Outliers (Grouped) ---
    outlier_list = []
    total_rows = len(df)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if count > 0:
            outlier_list.append((col, count))
    
    # Sort by number of outliers
    outlier_list.sort(key=lambda x: x[1], reverse=True)
    
    if outlier_list:
        # Show top specific outlier warning
        top_col, top_count = outlier_list[0]
        insights.append({
            "icon": "fa-triangle-exclamation",
            "color": "text-red-400",
            "text": f"<strong>{top_col}</strong> contains {top_count:,} outliers. Check for data quality issues."
        })
        
        # If many cols have outliers, add a summary insight
        if len(outlier_list) > 1:
            other_cols = [c for c, _ in outlier_list[1:3]] # Next 2 cols
            txt = f"Outliers also detected in <em>{', '.join(other_cols)}</em>"
            if len(outlier_list) > 3:
                txt += f" and {len(outlier_list) - 3} others."
            else:
                txt += "."
                
            insights.append({
                "icon": "fa-clipboard-list",
                "color": "text-red-400",
                "text": txt
            })

    # --- 3. Dominant Categories (Trends) ---
    for col in cat_cols:
        if df[col].nunique() < 20: # Only check categorical cols with few unique values
            top_val = df[col].mode()[0]
            freq = df[col].value_counts().iloc[0]
            pct = int((freq / total_rows) * 100)
            
            if pct > 60:
                insights.append({
                    "icon": "fa-crown",
                    "color": "text-yellow-400",
                    "text": f"<strong>{top_val}</strong> dominates <em>{col}</em> ({pct}% of entries)."
                })
                break # Only show one dominant category insight to save space

    # --- 4. High Variability ---
    max_cv = 0
    max_cv_col = None
    for col in num_cols:
        mean_val = df[col].mean()
        if mean_val != 0:
            cv = df[col].std() / mean_val
            if cv > 1.5 and cv > max_cv: # Threshold for "high" variability
                max_cv = cv
                max_cv_col = col
    
    if max_cv_col:
        insights.append({
            "icon": "fa-wave-square",
            "color": "text-purple-400",
            "text": f"High variability in <em>{max_cv_col}</em> (CV: {max_cv:.1f}). Data is widely spread."
        })

    # Fallback
    if not insights:
         insights.append({
            "icon": "fa-check-circle",
            "color": "text-green-400",
            "text": "Data looks balanced. No extreme relationships or outliers detected."
        })
        
    return insights[:5] # Limit to top 5


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return jsonify({"status": "Smart Data Analyzer API is running!", "version": "2.0"})


@app.route("/upload", methods=["POST"])
def upload():
    global df  # Declare df as global
    
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        if not file.filename.lower().endswith(".csv"):
            return "Only CSV files allowed", 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # ⚡ MEMORY-SAFE LOADING
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"📂 Processing file: {file.filename} ({file_size_mb:.2f}MB)")
        
        try:
            # Try multiple encodings
            for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    if file_size_mb > 10:
                        print(f"📦 Large file. Reading first 30k rows with {enc}...")
                        original_df = pd.read_csv(filepath, nrows=30000, encoding=enc, low_memory=False)
                    else:
                        print(f"📦 Reading full file with {enc}...")
                        original_df = pd.read_csv(filepath, encoding=enc, low_memory=False)
                    print("✅ Successfully read CSV")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("Could not decode CSV with any standard encoding.")
        except Exception as e:
            print(f"❌ Read failed: {e}")
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400
        
        # Clean and Analyze
        print("🧹 Cleaning data...")
        cleaned_dashboard_df = clean_data(original_df)
        df = cleaned_dashboard_df 
        
        # ⚡ SAVE CLEANED DATA: So the /download route can find it
        processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "cleaned_data.csv")
        cleaned_dashboard_df.to_csv(processed_path, index=False)
        print(f"💾 Cleaned data saved to: {processed_path}")

        print("📊 Analyzing data...")
        analysis = analyze_data(original_df, cleaned_dashboard_df)
        
        # ⚡ PREVIEW DATA: Add sample rows for the "Cleaned Data Preview" (Screen 4)
        analysis["sample_data"] = cleaned_dashboard_df.head(15).to_dict(orient="records")

        # JSON Safety: Robustly replace NaN/Inf/Non-serializable types
        import numpy as np
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(v) for v in obj]
            elif isinstance(obj, (float, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif pd.isna(obj): # Catch pandas specific NAs
                return None
            return obj

        print("🔏 Preparing JSON response...")
        safe_analysis = clean_nans(analysis)
        print("🚀 JSON prepared successfully. Sending response.")
        return jsonify(safe_analysis)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        # Log to server console
        print(f"⛔ SERVER CRASH during upload/analyze:\n{error_msg}")
        return jsonify({
            "error": "The server failed to process this specific dataset.",
            "details": str(e) if str(e) else "Internal processing error",
            "tip": "This dataset might be too complex or contain incompatible characters. Try a simpler version."
        }), 500


@app.route("/download")
def download():
    return send_file(
        os.path.join(app.config["PROCESSED_FOLDER"], "cleaned_data.csv"),
        as_attachment=True
    )


# -----------------------------
# ML Imports (Safe Import)
# -----------------------------
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False
    print(f"WARNING: scikit-learn import failed: {e}. Predictive features disabled.")

# ... (Existing Routes) ...

@app.route("/predict", methods=["POST"])
def predict():
    if not SKLEARN_AVAILABLE:
        return jsonify({"error": "ML Library (scikit-learn) not installed."}), 500
        
    global df
    if df is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    target = request.form.get("target")
    if not target or target not in df.columns:
        return jsonify({"error": "Please select a valid column to predict."}), 400

    try:
        # 1. Start with a fresh copy and sanitize target
        data = df.copy()
        
        # ⚡ CRITICAL: Clean NaNs and Infs FROM THE TARGET
        data[target] = pd.to_numeric(data[target], errors='coerce')
        data = data.dropna(subset=[target])
        data[target] = data[target].replace([np.inf, -np.inf], 0)
        
        if len(data) < 10:
            return jsonify({"error": f"The column '{target}' has too many missing or invalid values to train a model. Try another column."}), 400

        # 2. Convert Dates to Numbers (Universal)
        for col in data.columns:
            if data[col].dtype == "object":
                try:
                    # Quick date detection
                    if data[col].astype(str).str.contains('-|/|:').any():
                        temp_dates = pd.to_datetime(data[col], errors='coerce')
                        if temp_dates.notnull().any():
                             data[col] = temp_dates.apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
                except: continue

        # 3. Handle Features (Encoding & Cleaning)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        X_cols = [c for c in data.columns if c != target]
        for col in X_cols:
            if data[col].dtype == "object":
                # Drop IDs or very long text
                if data[col].nunique() > 100:
                    data = data.drop(columns=[col])
                    continue
                data[col] = le.fit_transform(data[col].astype(str))
            
            # 🧹 DEEP CLEAN: No Infs, No NaNs in features
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            data[col] = data[col].replace([np.inf, -np.inf], 0)

        # 4. Final Data Split Check
        y = data[target]
        X = data.drop(columns=[target])
        
        if X.empty or len(X.columns) == 0:
            return jsonify({"error": "No useful data columns found to help with the prediction."}), 400

        # Optimization for speed
        if len(data) > 8000:
            data = data.sample(8000, random_state=42)
            X = data.drop(columns=[target])
            y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(X_train) < 5:
            return jsonify({"error": "Not enough data samples for a reliable prediction."}), 400

        # 5. Train Model
        is_numeric_target = pd.api.types.is_numeric_dtype(y)
        unique_vals = y.nunique()

        if is_numeric_target and unique_vals > 20:
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
            model_type = "Regression"
        else:
            # Classification: Ensure y is discrete
            y_train = y_train.astype(int) if is_numeric_target else le.fit_transform(y_train.astype(str))
            y_test = y_test.astype(int) if is_numeric_target else le.fit_transform(y_test.astype(str))
            model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
            model_type = "Classification"

        model.fit(X_train, y_train)
        
        # 6. Scoring
        y_pred = model.predict(X_test)
        if model_type == "Regression":
            score = r2_score(y_test, y_pred)
            accuracy_text = f"{max(0, score):.1%} R² Confidence"
        else:
            score = accuracy_score(y_test, y_pred)
            accuracy_text = f"{score:.1%} Accuracy"

        # 7. Features & Predictions
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        features = [{"name": X.columns[i], "importance": round(importances[i]*100, 1)} for i in top_idx]

        # Convert y_test to a predictable format for indexing
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        
        predictions = []
        for i in range(min(15, len(y_test_list))):
            predictions.append({
                "actual": float(y_test_list[i]),
                "predicted": float(y_pred[i]),
                "label": f"#{i+1}"
            })

        global latest_prediction
        latest_prediction = {
            "model_type": model_type,
            "accuracy": accuracy_text,
            "features": features,
            "predictions": predictions,
            "target_name": target
        }
        return jsonify(latest_prediction)

    except Exception as e:
        import traceback
        print(f"⛔ PREDICT CRASH:\n{traceback.format_exc()}")
        return jsonify({
            "error": "Model training failed due to data inconsistencies.",
            "details": str(e),
            "tip": "This usually happens if a column has mixed text and numbers. Try a different column."
        }), 500

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    if not api_key or not GEMINI_AVAILABLE:
        return jsonify({"error": "Google Gemini API Key is missing. Please add GOOGLE_API_KEY to environment variables."}), 400
        
    global latest_prediction
    if not latest_prediction:
        return jsonify({"error": "No prediction results to analyze. Please train a model first."}), 400

    try:
        # Use 'gemini-pro' for maximum compatibility across all API versions
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare context for Gemini
        context = f"""
        Dataset column to predict: {latest_prediction['target_name']}
        Algorithm used: {latest_prediction['model_type']}
        Model Confidence/Accuracy: {latest_prediction['accuracy']}
        
        Most Important Factors (Drivers):
        {json.dumps(latest_prediction['features'], indent=2)}
        
        Recent Prediction Samples (Actual vs Predicted):
        {json.dumps(latest_prediction['predictions'][:10], indent=2)}
        """
        
        prompt = f"""
        You are an expert Data Scientist. Analyze these AI results and provide 3 ultra-short professional "Executive Insights" (in bullet points).
        Focus on:
        1. Whether the model is reliable based on accuracy.
        2. What the 'Top Drivers' mean for this specific business/data case.
        3. One 'Smart Forecast' or action the user should take.
        
        Results Context:
        {context}
        
        Response format: <b>Executive summary:</b><br/>• Point 1<br/>• Point 2<br/>• Point 3
        Keep it very concise.
        """
        
        response = model.generate_content(prompt)
        return jsonify({"insight": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
