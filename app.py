from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Global variable to store uploaded dataframe
df = None


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df):
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


# -----------------------------
# Data Analysis
# -----------------------------
import numpy as np

# -----------------------------
# Data Analysis
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
        k: int(v) for k, v in cleaned_df.isnull().sum().to_dict().items()
    }

    # Generate Chart Data
    analysis["charts"] = []
    
    # Identify columns
    cat_cols = cleaned_df.select_dtypes(include="object").columns.tolist()
    num_cols = cleaned_df.select_dtypes(include="number").columns.tolist()
    
    # 1. Categorical Analysis
    if cat_cols:
        col = cat_cols[0]
        # Bar Chart Data
        vc = cleaned_df[col].astype(str).value_counts().head(10)
        analysis["charts"].append({
            "type": "bar",
            "id": "barChart",
            "title": f"Top Categories in {col}",
            "labels": vc.index.tolist(),
            "data": vc.tolist(),
            "color": "gradient",
            "grid": "full"
        })
        
        # Pie Chart Data (Top 5)
        vc_pie = cleaned_df[col].astype(str).value_counts().head(5)
        analysis["charts"].append({
            "type": "doughnut",
            "id": "pieChart",
            "title": f"Composition of {col}",
            "labels": vc_pie.index.tolist(),
            "data": vc_pie.tolist(),
            "color": "multi",
            "grid": "half"
        })

    # 2. Numerical Analysis
    if num_cols:
        col = num_cols[0]
        # Histogram Data
        hist, bin_edges = np.histogram(cleaned_df[col].dropna(), bins=10)
        labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist))]
        
        analysis["charts"].append({
            "type": "line",
            "id": "lineChart",
            "title": f"Distribution of {col}",
            "labels": labels,
            "data": hist.tolist(),
            "color": "solid",
            "grid": "half"
        })

    # 3. Correlation Heatmap
    if len(num_cols) > 1:
        corr = cleaned_df[num_cols].corr().round(2)
        print("DEBUG: Correlation matrix generated") # Debug print
        analysis["heatmap"] = {
            "columns": corr.columns.tolist(),
            "data": corr.values.tolist() # Renamed from 'values' to 'data'
        }
    else:
        analysis["heatmap"] = None
        corr = None

    # 4. Smart Insights
    analysis["insights"] = generate_insights(cleaned_df, num_cols, cat_cols, corr)
    
    # 5. Metadata for Prediction
    analysis["columns"] = cleaned_df.columns.tolist()

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
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global df  # Declare df as global
    
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    if not file.filename.lower().endswith(".csv"):
        return "Only CSV files allowed"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    original_df = pd.read_csv(filepath)
    cleaned_df = clean_data(original_df)
    
    # Store cleaned dataframe globally for prediction
    df = cleaned_df
    
    analysis = analyze_data(original_df, cleaned_df)


    cleaned_path = os.path.join(app.config["PROCESSED_FOLDER"], "cleaned_data.csv")
    cleaned_df.to_csv(cleaned_path, index=False)

    preview = cleaned_df.head(10).to_html(
        classes="table",
        index=False
    )

    return render_template(
        "index.html",
        tables=preview,
        analysis=analysis,
        download_link="/download"
    )


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
    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    # Prepare Data
    data = df.copy()
    
    # 1. Handle Missing Values
    data = data.dropna()
    
    if len(data) < 10:
        return jsonify({"error": "Not enough data points after cleaning"}), 400

    # 2. Encode Categorical Data
    le = LabelEncoder()
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # 3. Split Features/Target
    X = data.drop(columns=[target])
    y = data[target]
    
    # 4. Train Model
    try:
        # Detect if Regression (Number) or Classification (Text/category)
        is_numeric = pd.api.types.is_numeric_dtype(df[target])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if is_numeric and df[target].nunique() > 10:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            score_text = f"{score:.2%} (R² Score)"
            model_type = "Regression"
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            score_text = f"{score:.2%} (Accuracy)"
            model_type = "Classification"

        # 5. Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5] # Top 5
        
        features = []
        for i in indices:
            features.append({
                "name": X.columns[i],
                "importance": round(importances[i] * 100, 1)
            })

        # 6. Generate Predictions (first 20 test samples for visualization)
        y_pred = model.predict(X_test)
        predictions = []
        sample_size = min(20, len(y_test))  # Show max 20 points
        
        for i in range(sample_size):
            # Try to get a meaningful label (date from original data)
            try:
                idx = y_test.index[i]
                # Try to find a date column in the original dataframe
                if 'from_date' in df.columns:
                    label = str(df.loc[idx, 'from_date'])
                elif 'to_date' in df.columns:
                    label = str(df.loc[idx, 'to_date'])
                elif 'date' in df.columns:
                    label = str(df.loc[idx, 'date'])
                else:
                    # If no date column, just show the index
                    label = str(idx)
            except:
                label = f"Sample {i + 1}"
            
            predictions.append({
                "actual": float(y_test.iloc[i]),
                "predicted": float(y_pred[i]),
                "label": label
            })

        return jsonify({
            "model_type": model_type,
            "accuracy": score_text,
            "features": features,
            "predictions": predictions,
            "target_name": target
        })

    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
