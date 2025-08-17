import os
import uuid
import json
import numpy as np
import pandas as pd
from typing import Dict, Callable, List, Union

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import aiplatform

# ========= Config / setup =========
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("BUCKET")                  # no "gs://" prefix; just the bucket name
GCS_PREFIX = os.getenv("GCS_PREFIX", "gotcha/datasets")

# Local source CSVs (exact file names you uploaded)
ESSAY_CSV = "Essay_test_data__human_vs_AI-assisted_.csv"
CODE_CSV  = "Code_test_data__human_vs_AI-assisted_.csv"

assert PROJECT_ID and REGION and BUCKET, "Please set PROJECT_ID, REGION and BUCKET in .env"

def init_vertex():
    aiplatform.init(project=PROJECT_ID, location=REGION)

# ========= GCS helpers =========
def gcs_upload(local_path: str, dest_blob: str) -> str:
    """Upload a local file to gs://BUCKET/dest_blob and return gs:// URI."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET}/{dest_blob}"

# ========= Data prep =========
def ensure_label_is_str(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df[label_col].dtype != object:
        df[label_col] = df[label_col].astype(str)
    return df

def drop_unused_columns(df: pd.DataFrame, extra_drop: List[str] = None) -> pd.DataFrame:
    """
    Drop columns that are IDs or long free-text. Keep categorical columns like
    'language' or 'variable_naming_style' (AutoML can handle them).
    """
    base_drops = [
        "text_excerpt",
        "code_snippet",
        "notes",
        "student_id",
        "submission_id",
    ]
    drops = base_drops + (extra_drop or [])
    keep_df = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore")
    return keep_df

def load_and_union_csvs(paths: List[str]) -> pd.DataFrame:
    """Read multiple CSVs, aligning on union of columns; missing columns are filled with NaN.
    Skips paths that do not exist and warns. Raises if none are found.
    """
    frames = []
    cols_union: List[str] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] CSV not found, skipping: {p}")
            continue
        df = pd.read_csv(p)
        frames.append(df)
        for c in df.columns:
            if c not in cols_union:
                cols_union.append(c)
    if not frames:
        raise FileNotFoundError("None of the provided CSVs were found: " + ", ".join(paths))
    # Reindex columns to the union
    aligned = [f.reindex(columns=cols_union) for f in frames]
    return pd.concat(aligned, ignore_index=True)

def add_predefined_split(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Add a 'split' column with TRAIN/VALIDATE/TEST based on fractions."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    val_end = int(n * val_ratio)
    test_end = val_end + int(n * test_ratio)
    splits = ["VALIDATE"] * val_end + ["TEST"] * (test_end - val_end) + ["TRAIN"] * (n - test_end)
    df["split"] = splits
    return df

def add_predefined_stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Stratified predefined split by label with minimum 1 per split per class when possible."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    if label_col not in df.columns:
        return add_predefined_split(df, val_ratio, test_ratio, seed)

    # Group indices by label
    label_to_idxs = {}
    for idx, lbl in enumerate(df[label_col].astype(str).tolist()):
        label_to_idxs.setdefault(lbl, []).append(idx)

    split = ["TRAIN"] * n

    # First, assign one VALIDATE and one TEST per label if possible
    for lbl, idxs in label_to_idxs.items():
        if not idxs:
            continue
        split[idxs[0]] = "VALIDATE"
        if len(idxs) >= 2:
            split[idxs[1]] = "TEST"

    # Count current VAL/TEST sizes
    current_val = sum(1 for s in split if s == "VALIDATE")
    current_test = sum(1 for s in split if s == "TEST")

    target_val = max(1, int(round(n * val_ratio)))
    target_test = max(1, int(round(n * test_ratio)))
    # Ensure we don't exceed n- at least one train row if possible
    max_target_val = max(1, min(target_val, n - 2))
    max_target_test = max(1, min(target_test, n - 1 - max_target_val))

    # Fill remaining VAL slots
    for i in range(n):
        if current_val >= max_target_val:
            break
        if split[i] == "TRAIN":
            split[i] = "VALIDATE"
            current_val += 1
    # Fill remaining TEST slots
    for i in range(n):
        if current_test >= max_target_test:
            break
        if split[i] == "TRAIN":
            split[i] = "TEST"
            current_test += 1

    out = df.copy()
    out["split"] = split
    return out

def bootstrap_augment(df: pd.DataFrame, label_col: str, min_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Upsample by bootstrapping per class until at least `min_rows` rows.
    Adds small noise to numeric columns to avoid exact duplicates (AutoML safe).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n >= min_rows:
        return df

    groups = {k: v.copy() for k, v in df.groupby(label_col)}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    out_parts = [df]
    needed = min_rows - n
    labels = list(groups.keys()) or [None]
    i = 0
    while needed > 0:
        lbl = labels[i % len(labels)]
        pool = groups[lbl] if lbl is not None else df
        take = min(len(pool), needed)
        boot = pool.sample(n=take, replace=True, random_state=seed + i)
        # add tiny noise to numeric columns to avoid identical duplicates
        for col in numeric_cols:
            base = df[col].dropna()
            if base.empty:
                continue
            std = float(base.std()) if base.std() and not np.isnan(base.std()) else 1.0
            noise = rng.normal(0.0, 0.01 * std if std > 0 else 0.01, size=len(boot))
            boot[col] = (boot[col].astype(float) + noise).astype(boot[col].dtype, errors="ignore")
        out_parts.append(boot)
        needed -= take
        i += 1

    aug = pd.concat(out_parts, ignore_index=True)
    return aug.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ========= Vertex AI: train / deploy / predict =========
def create_tabular_dataset(gcs_csv_uri: str, display_name: str):
    init_vertex()
    ds = aiplatform.TabularDataset.create(
        display_name=f"{display_name}-dataset",
        gcs_source=[gcs_csv_uri],
    )
    return ds

def train_automl_tabular(dataset, display_name: str, target_col: str):
    """
    Trains an AutoML Tabular classification model.
    """
    init_vertex()
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"{display_name}-job",
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss",
    )
    model = job.run(
        dataset=dataset,
        target_column=target_col,
        predefined_split_column_name="split",  # <-- correct kwarg
        budget_milli_node_hours=1000,
        model_display_name=f"{display_name}-model",
        sync=True,
    )
    return model

def print_eval(model):
    """Fetch and print key evaluation metrics for quick debugging in a hackathon demo."""
    try:
        evals = model.list_model_evaluations()
        if not evals:
            print("No evaluations found for the model.")
            return
        for e in evals:
            metrics = e.metrics or {}
            print("\n=== Evaluation:", e.display_name or "overall", "===")
            for k in ("auRoc", "auPrc", "logLoss", "accuracy"):
                if k in metrics:
                    print(f"{k}: {metrics[k]}")
            # Print a few threshold trade-off points if available
            cmes = metrics.get("confidenceMetricsEntries") or []
            for point in cmes[:3]:
                th = point.get("confidenceThreshold")
                prec = point.get("precision")
                rec = point.get("recall")
                if th is not None:
                    print(f"th={th} precision={prec} recall={rec}")
    except Exception as ex:
        print("Could not fetch evaluation metrics:", ex)


def batch_predict(model, gcs_source_uri: str, gcs_dest_prefix: str):
    """Run a batch prediction job for many rows stored in GCS (CSV in, JSONL out)."""
    init_vertex()
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=f"{model.display_name}-batch",
        model=model.resource_name,
        gcs_source=gcs_source_uri,
        gcs_destination_prefix=gcs_dest_prefix,
        instances_format="csv",
        predictions_format="jsonl",
        machine_type="n1-standard-2",
        sync=True,
    )
    print("Batch output directory:", job.output_info.gcs_output_directory)
    return job


def build_sample_from_df(df: pd.DataFrame, target_col: str) -> Dict:
    """Return the first row as a prediction instance, excluding the label column."""
    cols = [c for c in df.columns if c != target_col]
    return df[cols].iloc[0].to_dict()

def get_or_create_endpoint(display_name: str):
    """Return an existing Endpoint by display name, or create it if missing."""
    init_vertex()
    try:
        # Filter by display_name to avoid listing everything
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{display_name}"')
        for ep in endpoints:
            if ep.display_name == display_name:
                return ep
    except Exception as ex:
        print("Endpoint.list failed (will create a new endpoint):", ex)
    return aiplatform.Endpoint.create(display_name=display_name)

def deploy_to_endpoint(model, endpoint_display_name: str):
    """
    Reuse an endpoint if it exists; otherwise create and deploy the model.
    If the model version is already deployed to the endpoint, skip deployment.
    """
    init_vertex()

    endpoint = get_or_create_endpoint(endpoint_display_name)

    # Skip deploy if this exact model is already on the endpoint
    try:
        for deployed in endpoint.list_models():
            if getattr(deployed, "model", None) == model.resource_name:
                print("Model already deployed to endpoint; skipping deploy.")
                return endpoint
    except Exception as ex:
        print("Could not list deployed models; will attempt to deploy anyway:", ex)

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{endpoint_display_name}-deployed",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    return endpoint

def online_predict(endpoint: aiplatform.Endpoint, instance: Dict):
    """
    Sends one instance for online prediction. Keys must match training columns (minus label).
    """
    prediction = endpoint.predict(instances=[instance])
    # AutoML Tabular typically returns a dict with classes and scores
    return prediction.predictions

# ========= Orchestration =========
def run_one_dataset(
    local_csv_path: Union[str, List[str]],
    target_col: str,
    display_prefix: str,
    sample_builder: Callable[[pd.DataFrame], Dict],
    extra_drop: List[str] = None,
):
    if isinstance(local_csv_path, list):
        print(f"\n=== Processing multiple CSVs ({len(local_csv_path)}) ===")
        df = load_and_union_csvs(local_csv_path)
    else:
        if not os.path.exists(local_csv_path):
            raise FileNotFoundError(f"Local CSV not found: {local_csv_path}")
        print(f"\n=== Processing {local_csv_path} ===")
        df = pd.read_csv(local_csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data")

    # ensure label is categorical/string
    df = ensure_label_is_str(df, target_col)

    # Filter columns
    feature_df = drop_unused_columns(df, extra_drop=extra_drop)

    feature_df = add_predefined_stratified_split(feature_df, target_col)

    # Ensure we meet AutoML Tabular's minimum row requirement (~1000 rows)
    if len(feature_df) < 1000:
        print(f"[INFO] Dataset has {len(feature_df)} rows; augmenting to 1000 for AutoML...")
        feature_df = bootstrap_augment(feature_df, target_col, min_rows=1000)
        print(f"[INFO] After augmentation: {len(feature_df)} rows")

    # Save a temp CSV that Vertex will ingest
    tmp_name = f"/tmp/{display_prefix}_{uuid.uuid4().hex}.csv"
    feature_df.to_csv(tmp_name, index=False)
    print(f"Prepared training CSV: {tmp_name}")

    # Upload to GCS
    dest_blob = f"{GCS_PREFIX}/{display_prefix}/{os.path.basename(tmp_name)}"
    gcs_uri = gcs_upload(tmp_name, dest_blob)
    print(f"Uploaded training data → {gcs_uri}")

    # Create dataset
    dataset = create_tabular_dataset(gcs_csv_uri=gcs_uri, display_name=display_prefix)
    print("Vertex dataset:", dataset.resource_name)

    # Train
    model = train_automl_tabular(dataset=dataset, display_name=display_prefix, target_col=target_col)
    print_eval(model)
    print("Vertex model:", model.resource_name)

    # Deploy
    endpoint = deploy_to_endpoint(model, endpoint_display_name=f"{display_prefix}-endpoint")
    print("Vertex endpoint:", endpoint.resource_name)

    # Build and send a sample prediction
    instance = sample_builder(feature_df)
    preds = online_predict(endpoint, instance)

    print("\n--- Sample instance ---")
    print(json.dumps(instance, indent=2))
    print("\n--- Prediction response ---")
    print(json.dumps(preds, indent=2))

def build_essay_sample(df: pd.DataFrame) -> Dict:
    return build_sample_from_df(df, "label")


def build_code_sample(df: pd.DataFrame) -> Dict:
    return build_sample_from_df(df, "label")

if __name__ == "__main__":
    """
    Before running:
      1) gcloud auth application-default login
      2) gcloud config set project YOUR_PROJECT_ID
      3) Enable "Vertex AI API" and "Cloud Storage" APIs in the Google Cloud console
      4) Ensure your BUCKET exists and is in a location compatible with REGION
    """

    # --- Essays model ---
    run_one_dataset(
        local_csv_path=[ESSAY_CSV, "One-student_essays__rich____preview.csv"],
        target_col="label",
        display_prefix="gotcha-essays",
        sample_builder=build_essay_sample,
        extra_drop=None,
    )

    # --- Code model ---
    # Keep 'language' and 'variable_naming_style' (categorical). We already drop long text/IDs in drop_unused_columns.
    run_one_dataset(
        local_csv_path=[CODE_CSV, "One-student_code__rich____preview.csv"],
        target_col="label",
        display_prefix="gotcha-code",
        sample_builder=build_code_sample,
        extra_drop=None,
    )