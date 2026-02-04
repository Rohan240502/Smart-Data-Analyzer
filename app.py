from flask import Flask, render_template, request, send_file
import pandas as pd
import os

# -----------------------------
# App Configuration
# -----------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER


# -----------------------------
# Data Cleaning Function
# -----------------------------
def clean_data(df):
    # Drop completely empty rows & columns
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    # Try converting columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# -----------------------------
# Data Analysis Function
# -----------------------------
def analyze_data(original_df, cleaned_df):
    analysis = {}

    # -------- BEFORE CLEANING --------
    analysis["before"] = {
        "rows": original_df.shape[0],
        "columns": original_df.shape[1],
        "missing_total": int(original_df.isnull().sum().sum()),
    }

    # -------- AFTER CLEANING --------
    analysis["after"] = {
        "rows": cleaned_df.shape[0],
        "columns": cleaned_df.shape[1],
        "missing_total": int(cleaned_df.isnull().sum().sum()),
    }

    # -------- IMPACT --------
    analysis["impact"] = {
        "rows_removed": analysis["before"]["rows"] - analysis["after"]["rows"],
        "columns_removed": analysis["before"]["columns"] - analysis["after"]["columns"],
    }

    # -------- Missing values per column (after) --------
    analysis["missing_by_column"] = cleaned_df.isnull().sum().to_dict()

    return analysis


# -----------------------------
# Home Page
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------------
# Upload & Process CSV File
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # CSV only
    if not file.filename.lower().endswith(".csv"):
        return "Only CSV files are allowed"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Read CSV
    df = pd.read_csv(filepath)

    # Clean data
    cleaned_df = clean_data(df)

    # Analyze data
    analysis = analyze_data(df,cleaned_df)

    # Save cleaned file
    cleaned_path = os.path.join(
        app.config["PROCESSED_FOLDER"], "cleaned_data.csv"
    )
    cleaned_df.to_csv(cleaned_path, index=False)

    # Preview
    preview = cleaned_df.head(10).to_html(
        classes="table table-bordered",
        index=False
    )

    return render_template(
        "index.html",
        tables=preview,
        analysis=analysis,          # ✅ FIXED
        download_link="/download"
    )


# -----------------------------
# Download Cleaned CSV
# -----------------------------
@app.route("/download", methods=["GET"])
def download():
    return send_file(
        os.path.join(app.config["PROCESSED_FOLDER"], "cleaned_data.csv"),
        as_attachment=True
    )


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
