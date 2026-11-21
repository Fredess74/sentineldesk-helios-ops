"""Central configuration, constants, paths, and thresholds for SentinelDesk."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

KB_TRAIN_PATH = DATA_DIR / "kb_train.parquet"
HOLDOUT_TEST_PATH = DATA_DIR / "holdout_test.parquet"

KB_AUDIT_PATH = ARTIFACTS_DIR / "kb_audit_summary.json"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "kb_faiss.index"
KB_METADATA_PATH = ARTIFACTS_DIR / "kb_metadata.parquet"
KB_EMBEDDINGS_PATH = ARTIFACTS_DIR / "kb_embeddings.npy"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"
ML_MODELS_PATH = ARTIFACTS_DIR / "ml_models.joblib"
HOLDOUT_METRICS_PATH = ARTIFACTS_DIR / "holdout_metrics.json"

# Official dataset identifiers (clean 10-queue versions only)
HF_DATASET_REPO = "Tobi-Bueck/customer-support-tickets"
DATASET_FILE_20K = "dataset-tickets-multi-lang-4-20k.csv"
DATASET_FILE_50K = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8

# Expected clean label spaces
EXPECTED_TYPES = ["Incident", "Request", "Problem", "Change"]
EXPECTED_PRIORITIES = ["low", "medium", "high"]
EXPECTED_LANGUAGES = ["en", "de"]
EXPECTED_QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

# Embedding configuration
PRIMARY_EMBEDDING_MODEL = "google/embeddinggemma-2"
FALLBACK_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 256

# Model & API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DECISION_MODEL = os.getenv("DECISION_MODEL", "gemini-decision-1")
CLOUD_DRAFTER_MODELS = [
    os.getenv("CLOUD_DRAFTER_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.1-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemini-3.5-flash",
]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Hybrid triage blend weights (favor calibrated domain SVM + semantic Decision-1 tie-breaking)
HYBRID_ML_WEIGHT = float(os.getenv("HYBRID_ML_WEIGHT", "0.75"))
HYBRID_D1_WEIGHT = float(os.getenv("HYBRID_D1_WEIGHT", "0.25"))

# Escalation gate thresholds
MIN_CONFIDENCE_THRESHOLD = 0.65
MIN_RETRIEVAL_SIMILARITY = 0.70
PROMPT_INJECTION_THRESHOLD = 0.50
UNSUPPORTED_COMMITMENT_THRESHOLD = 0.65
P1_JOINT_RISK_THRESHOLD = 0.35

# Pass-2 Grounding Audit thresholds
MIN_CLAIMS_SUPPORTED = 0.85
MAX_FABRICATES_REFUND_SLA = 0.15

# Multilingual Outage Burst Radar settings
BURST_WINDOW_SECONDS = 900  # 15-minute rolling window
BURST_SIMILARITY_THRESHOLD = 0.78
BURST_MIN_TICKETS = 3
BURST_MAX_BUFFER = 200
