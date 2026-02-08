from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os

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
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Global variable to store uploaded dataframe
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
        return jsonify({"error": f"Invalid target column: {target}"}), 400

    try:
        # ⚡ TURBO MODE: Extreme optimizations for Render Free Tier
        print(f"🔮 Training Turbo AI for target: {target}")
        
        # 1. Feature Selection: Keep target + top 20 most relevant columns only
        # This prevents OOM (Out of Memory) crashes on large datasets
        all_cols = df.columns.tolist()
        if len(all_cols) > 25:
             # Just keep a subset to avoid memory bloat
             num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
             num_cols = [c for c in num_cols if c != target][:20]
             subset = [target] + num_cols
             data = df[subset].copy()
             print(f"✂️ Column reduction: Kept 20 numeric columns to save memory.")
        else:
             data = df.copy()

        # 2. Handle Missing Values
        data = data.dropna(subset=[target])
        
        # 3. Sampling: 5,000 rows is the "sweet spot" for speed vs accuracy on free tier
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
            print("📦 Aggressive Sampling: 5,000 rows.")

        if len(data) < 10:
             return jsonify({"error": "Not enough data points with valid target values."}), 400

        # 4. Smart Encoding: Ignore high-cardinality text (ID/Date strings)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        cols_to_drop = []
        for col in data.columns:
            if col != target and data[col].dtype == "object":
                if data[col].nunique() > 50:
                    cols_to_drop.append(col)
        
        # ⚡ SAFETY: Don't drop everything! Keep at least 5 features if possible
        if len(data.columns) - len(cols_to_drop) < 5:
             cols_to_drop = cols_to_drop[:max(0, len(cols_to_drop) - 5)]
             print("🛡️ Safety: Retained some complex columns to prevent empty feature set.")

        data = data.drop(columns=cols_to_drop)
        for col in data.select_dtypes(include="object").columns:
            if col != target:
                data[col] = le.fit_transform(data[col].astype(str))

        # 5. Split Features/Target
        X = data.drop(columns=[target])
        y = data[target]
        
        if X.empty or X.columns.empty:
             return jsonify({"error": "This column cannot be predicted because the other columns are too complex or empty. Try a different column."}), 400
        
        # Ensure y is numeric if classification
        is_numeric_target = pd.api.types.is_numeric_dtype(y)
        if not is_numeric_target:
            y = le.fit_transform(y.astype(str))

        # 6. Scaling
        num_features = X.select_dtypes(include=[np.number]).columns
        if not num_features.empty:
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X[num_features] = scaler.fit_transform(X[num_features])
            except: pass

        # 7. Train Model (Ultralight Settings)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Low estimators + Max Depth = Reliable & Fast
        if is_numeric_target and data[target].nunique() > 10:
            model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            score_text = f"{score:.1%} R² Score"
            model_type = "Regression"
        else:
            model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            score_text = f"{score:.1%} Accuracy"
            model_type = "Classification"

        # 8. Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        features = [{"name": X.columns[i], "importance": round(importances[i] * 100, 1)} for i in indices]

        # 9. Predictions for Chart
        y_pred = model.predict(X_test)
        predictions = []
        for i in range(min(15, len(y_test))):
            predictions.append({
                "actual": float(y_test.iloc[i]),
                "predicted": float(y_pred[i]),
                "label": f"P{i+1}"
            })

        print(f"✅ Success! {model_type} trained.")
        return jsonify({
            "model_type": model_type,
            "accuracy": score_text,
            "features": features,
            "predictions": predictions,
            "target_name": target
        })

    except Exception as e:
        import traceback
        print(f"❌ ML ERROR: {traceback.format_exc()}")
        return jsonify({"error": "Training failed due to data complexity.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
