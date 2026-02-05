from flask import Flask, render_template, request, send_file
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER


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

    analysis["impact"] = {
        "rows_removed": analysis["before"]["rows"] - analysis["after"]["rows"],
        "columns_removed": analysis["before"]["columns"] - analysis["after"]["columns"],
    }

    analysis["missing_by_column"] = {
        k: int(v) for k, v in cleaned_df.isnull().sum().to_dict().items()
    }

    # Bar chart (categorical)
    cat_cols = cleaned_df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        col = cat_cols[0]
        vc = cleaned_df[col].astype(str).value_counts().head(10)
        analysis["chart"] = {
            "title": f"Top 10 values in '{col}'",
            "labels": vc.index.tolist(),
            "values": vc.tolist(),
        }
    else:
        analysis["chart"] = None

    return analysis


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    if not file.filename.lower().endswith(".csv"):
        return "Only CSV files allowed"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    cleaned_df = clean_data(df)
    analysis = analyze_data(df, cleaned_df)

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


if __name__ == "__main__":
    app.run(debug=True)
