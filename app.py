import hashlib
import json
from collections import Counter
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_PATH = Path("model/e-waste-model-MobileNetV2.keras")
DATA_DIR = Path("data")
USERS_PATH = DATA_DIR / "users.json"
HISTORY_DIR = DATA_DIR / "history"
TRAINING_DIR = DATA_DIR / "training"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = [
    "Battery",
    "Keyboard",
    "Microwave",
    "Mobile",
    "Mouse",
    "PCB",
    "Player",
    "Printer",
    "Television",
    "Washing Machine",
]
CLASS_DETAILS = {
    "Battery": {
        "description": "Portable energy storage device that may contain toxic metals and chemicals.",
        "risk": "Do not crush, puncture, or throw into household trash.",
        "action": "Take it to a battery recycling collection point or authorized e-waste center.",
    },
    "Keyboard": {
        "description": "Input device made of plastic, circuit boards, and wiring.",
        "risk": "Useful parts may still be reusable even when the outer body looks damaged.",
        "action": "Donate working units or send broken ones to an electronics recycler.",
    },
    "Microwave": {
        "description": "Large household appliance with metal casing and electronic controls.",
        "risk": "Should be handled carefully because internal components can remain hazardous.",
        "action": "Use a municipal bulky e-waste collection service or appliance recycler.",
    },
    "Mobile": {
        "description": "Handheld smart device with battery, display, camera modules, and circuit boards.",
        "risk": "Contains personal data in addition to recyclable electronics.",
        "action": "Back up data, factory reset the device, then recycle through an authorized center.",
    },
    "Mouse": {
        "description": "Compact pointing device with plastic shell, cable or battery, and sensors.",
        "risk": "Often mixed with general waste even though it contains recoverable parts.",
        "action": "Bundle with other small electronics for proper e-waste recycling.",
    },
    "PCB": {
        "description": "Printed circuit board used inside electronic equipment.",
        "risk": "May contain lead, solder residues, and recoverable metals.",
        "action": "Send only to certified e-waste handlers or specialized recyclers.",
    },
    "Player": {
        "description": "Portable or home media playback device with electronics and mixed materials.",
        "risk": "Batteries or small electronic parts can become unsafe if damaged.",
        "action": "Recycle as a small consumer electronic item.",
    },
    "Printer": {
        "description": "Office or home printing device with motors, cartridges, and boards.",
        "risk": "Ink or toner cartridges should be handled separately when possible.",
        "action": "Use an e-waste drop-off site and return cartridges through refill or take-back programs.",
    },
    "Television": {
        "description": "Large display device with screen panel, internal circuitry, and power components.",
        "risk": "Screens can break and older units may contain hazardous materials.",
        "action": "Use certified TV recycling or municipality-approved collection services.",
    },
    "Washing Machine": {
        "description": "Heavy appliance with electronics, wiring, and metal components.",
        "risk": "Requires safe transport because of size and residual water exposure.",
        "action": "Contact local appliance recycling, retailer take-back, or municipal pickup.",
    },
}
REPORT_COLUMNS = [
    "timestamp",
    "source",
    "filename",
    "predicted_class",
    "confidence_percent",
    "confidence_level",
    "top_2",
    "top_2_percent",
    "top_3",
    "top_3_percent",
]


st.set_page_config(
    page_title="E-Waste Intelligence Suite",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ew-primary: #0f766e;
            --ew-secondary: #ff9f1c;
            --ew-accent: #42c96b;
            --ew-alert: #ff5f6d;
            --ew-electric: #00b8d9;
            --ew-ink: #14332d;
            --ew-surface: rgba(255, 255, 255, 0.84);
            --ew-border: rgba(16, 83, 64, 0.14);
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 10%, rgba(66, 201, 107, 0.18), transparent 22%),
                radial-gradient(circle at 90% 8%, rgba(255, 159, 28, 0.18), transparent 22%),
                radial-gradient(circle at 80% 28%, rgba(0, 184, 217, 0.10), transparent 16%),
                linear-gradient(180deg, #f4fbf6 0%, #eaf5ee 58%, #f8fbff 100%);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: var(--ew-ink);
            letter-spacing: -0.02em;
        }
        p, label, .stCaption, .stMarkdown, .stText {
            color: #264740;
        }
        .hero-card {
            padding: 2rem 2.1rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 84% 18%, rgba(255, 159, 28, 0.28), transparent 18%),
                radial-gradient(circle at 18% 82%, rgba(66, 201, 107, 0.20), transparent 20%),
                linear-gradient(135deg, rgba(7, 65, 66, 0.99), rgba(16, 116, 95, 0.96), rgba(19, 143, 113, 0.92));
            color: #f5fff8;
            box-shadow: 0 24px 60px rgba(15, 56, 43, 0.18);
            margin-bottom: 1.2rem;
            overflow: hidden;
            position: relative;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            right: -48px;
            bottom: -48px;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.18), rgba(255,255,255,0.02));
        }
        .info-card {
            padding: 1.2rem 1.2rem;
            border-radius: 20px;
            background: var(--ew-surface);
            border: 1px solid var(--ew-border);
            box-shadow: 0 14px 30px rgba(30, 54, 45, 0.08);
            margin-bottom: 1rem;
        }
        .mini-stat {
            min-height: 108px;
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--ew-border);
            box-shadow: 0 12px 28px rgba(30, 54, 45, 0.06);
        }
        .gallery-card {
            padding: 0.85rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--ew-border);
            margin-bottom: 0.75rem;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.45fr 0.9fr;
            gap: 1rem;
            align-items: stretch;
            margin-bottom: 1rem;
        }
        .hero-side {
            padding: 1.4rem 1.35rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.08));
            border: 1px solid rgba(255,255,255,0.18);
            backdrop-filter: blur(10px);
        }
        .hero-chip {
            display: inline-block;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            color: #f6fffb;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .symbol-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 0.95rem;
        }
        .symbol-tile {
            min-width: 108px;
            padding: 0.8rem 0.85rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.08));
            border: 1px solid rgba(255,255,255,0.16);
            text-align: center;
            color: #f7fffb;
        }
        .symbol-mark {
            display: block;
            font-size: 1.45rem;
            line-height: 1;
            margin-bottom: 0.4rem;
        }
        .symbol-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .workspace-shell {
            padding: 1.1rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.66), rgba(255,255,255,0.48));
            border: 1px solid var(--ew-border);
            box-shadow: 0 16px 36px rgba(30, 54, 45, 0.06);
        }
        .metric-band {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin: 1rem 0 1.1rem 0;
        }
        .metric-tile {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,249,247,0.88));
            border: 1px solid var(--ew-border);
            box-shadow: 0 12px 28px rgba(30, 54, 45, 0.07);
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #5d7f76;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }
        .metric-value {
            font-size: 1.35rem;
            color: var(--ew-ink);
            font-weight: 800;
            line-height: 1.1;
        }
        .metric-copy {
            font-size: 0.92rem;
            color: #45665d;
            margin-top: 0.35rem;
        }
        .auth-card {
            max-width: 580px;
            margin: 1rem auto;
            padding: 1.3rem 1.4rem;
            border-radius: 26px;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245, 251, 247, 0.90));
            box-shadow: 0 20px 44px rgba(15, 56, 43, 0.12);
            border: 1px solid var(--ew-border);
        }
        .mode-card {
            padding: 1rem 1.15rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(248, 251, 249, 0.84));
            border: 1px solid var(--ew-border);
            box-shadow: 0 14px 28px rgba(30, 54, 45, 0.06);
            margin: 0.5rem 0 1rem 0;
        }
        .section-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #4f7d72;
            margin-bottom: 0.45rem;
        }
        .webcam-hero {
            padding: 1.25rem 1.35rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(21, 94, 82, 0.98), rgba(27, 127, 94, 0.92));
            color: #f6fffb;
            box-shadow: 0 18px 42px rgba(16, 83, 64, 0.16);
            margin-bottom: 1rem;
        }
        .webcam-tip {
            padding: 0.95rem 1rem;
            border-left: 5px solid var(--ew-secondary);
            background: rgba(255, 250, 240, 0.9);
            border-radius: 14px;
            margin-bottom: 1rem;
        }
        .result-card {
            padding: 1.2rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247, 250, 248, 0.9));
            border: 1px solid var(--ew-border);
            box-shadow: 0 16px 34px rgba(30, 54, 45, 0.08);
            margin-bottom: 1rem;
        }
        .result-highlight {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(255,159,28,0.12), rgba(66,201,107,0.10));
            border: 1px solid rgba(21, 94, 82, 0.12);
            margin-bottom: 0.85rem;
        }
        .confidence-badge {
            display: inline-block;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            background: rgba(21, 94, 82, 0.10);
            color: #155e52;
            font-weight: 700;
            font-size: 0.84rem;
            margin-bottom: 0.75rem;
        }
        .soft-panel {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.8);
            border: 1px solid var(--ew-border);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
        }
        .auth-feature-list {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
            margin-top: 0.9rem;
        }
        .auth-feature {
            padding: 0.9rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(244, 250, 246, 0.86));
            border: 1px solid var(--ew-border);
        }
        .auth-banner {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(66,201,107,0.12), rgba(0,184,217,0.10), rgba(255,159,28,0.12));
            border: 1px solid rgba(15, 118, 110, 0.14);
            margin-top: 1rem;
        }
        .sidebar-symbol {
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15,118,110,0.95), rgba(66,201,107,0.86));
            color: white;
            margin-bottom: 0.9rem;
            box-shadow: 0 14px 28px rgba(15, 118, 110, 0.16);
        }
        @media (max-width: 900px) {
            .hero-grid, .metric-band, .auth-feature-list {
                grid-template-columns: 1fr;
            }
        }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 0;
            min-height: 2.9rem;
            font-weight: 700;
            background: linear-gradient(135deg, #155e52, #1f8366);
            color: white;
            box-shadow: 0 12px 24px rgba(21, 94, 82, 0.18);
        }
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #104a40, #1a6f58);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.7);
            padding: 0.55rem 1rem;
            border: 1px solid var(--ew-border);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #155e52, #1f8366);
            color: white;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 16px;
            border-color: rgba(16, 83, 64, 0.14);
        }
        .st-emotion-cache-13k62yr, .st-emotion-cache-ocqkz7 {
            gap: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text("{}", encoding="utf-8")


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def user_history_path(username: str) -> Path:
    safe_username = "".join(char for char in username if char.isalnum() or char in {"-", "_"}).strip("_")
    return HISTORY_DIR / f"{safe_username or 'user'}.json"


def load_user_history(username: str) -> list[dict[str, object]]:
    return read_json(user_history_path(username), [])


def save_user_history(username: str, history: list[dict[str, object]]) -> None:
    write_json(user_history_path(username), history)


@st.cache_resource(show_spinner="Loading classification model...")
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Please add the trained model before running the app."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def initialize_state() -> None:
    ensure_data_dirs()
    if "username" not in st.session_state:
        st.session_state.username = None
    if "history" not in st.session_state:
        st.session_state.history = []


def login_user(username: str) -> None:
    st.session_state.username = username
    st.session_state.history = load_user_history(username)


def logout_user() -> None:
    st.session_state.username = None
    st.session_state.history = []


def register_account(username: str, password: str) -> tuple[bool, str]:
    users = read_json(USERS_PATH, {})
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if username in users:
        return False, "That username already exists."
    users[username] = {"password_hash": hash_password(password), "created_at": datetime.now().isoformat()}
    write_json(USERS_PATH, users)
    save_user_history(username, [])
    return True, "Account created successfully. You can sign in now."


def authenticate_account(username: str, password: str) -> tuple[bool, str]:
    users = read_json(USERS_PATH, {})
    account = users.get(username.strip())
    if not account:
        return False, "No account found for that username."
    if account["password_hash"] != hash_password(password):
        return False, "Incorrect password."
    login_user(username.strip())
    return True, f"Signed in as {username.strip()}."


def preprocess_image(image: Image.Image) -> np.ndarray:
    fitted = ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    image_array = np.asarray(fitted, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image: Image.Image, model: tf.keras.Model) -> tuple[str, float, list[tuple[str, float]]]:
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]
    ranked_indices = np.argsort(probabilities)[::-1]
    top_predictions = [(CLASS_NAMES[index], float(probabilities[index])) for index in ranked_indices[:3]]
    predicted_label, predicted_probability = top_predictions[0]
    return predicted_label, predicted_probability, top_predictions


def confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High confidence"
    if score >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


def render_metric_tile(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_rows(top_predictions: list[tuple[str, float]]) -> None:
    for index, (label, score) in enumerate(top_predictions, start=1):
        st.markdown(f"**Top {index}: {label}**")
        st.progress(int(score * 100))
        st.caption(f"{score * 100:.2f}% probability")


def record_history(source: str, filename: str, top_predictions: list[tuple[str, float]]) -> None:
    if not st.session_state.username:
        return
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "filename": filename,
        "predicted_class": top_predictions[0][0],
        "confidence_percent": round(top_predictions[0][1] * 100, 2),
        "confidence_level": confidence_label(top_predictions[0][1]),
        "top_2": top_predictions[1][0],
        "top_2_percent": round(top_predictions[1][1] * 100, 2),
        "top_3": top_predictions[2][0],
        "top_3_percent": round(top_predictions[2][1] * 100, 2),
    }
    st.session_state.history.append(entry)
    save_user_history(st.session_state.username, st.session_state.history)


def build_csv(history: list[dict[str, object]]) -> str:
    output = StringIO()
    writer = DictWriter(output, fieldnames=REPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()


def class_distribution_figure(history: list[dict[str, object]]) -> go.Figure:
    counts = Counter(item["predicted_class"] for item in history)
    labels = list(counts.keys())
    values = list(counts.values())
    figure = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color="#1c8769"),
                text=values,
                textposition="outside",
            )
        ]
    )
    figure.update_layout(
        title="Predicted Class Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    return figure


def confidence_breakdown_figure(history: list[dict[str, object]]) -> go.Figure:
    counts = Counter(item["confidence_level"] for item in history)
    labels = ["High confidence", "Moderate confidence", "Low confidence"]
    values = [counts.get(label, 0) for label in labels]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.48,
                marker=dict(colors=["#1c8769", "#ffb74d", "#d6604d"]),
            )
        ]
    )
    figure.update_layout(
        title="Confidence Breakdown",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    return figure


def render_history_panel() -> None:
    history = st.session_state.history
    st.markdown(
        """
        <div class="info-card">
            <div class="section-kicker">Saved Records</div>
            <h3 style="margin-top:0;">User History</h3>
            <p style="margin-bottom:0;">A persistent log of predictions tied to your local account.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not history:
        st.info("Run some predictions after signing in and your saved history will appear here.")
        return

    total_items = len(history)
    avg_confidence = sum(item["confidence_percent"] for item in history) / total_items
    dominant_class = Counter(item["predicted_class"] for item in history).most_common(1)[0][0]

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Items Analyzed", total_items)
    metric_col_2.metric("Average Confidence", f"{avg_confidence:.2f}%")
    metric_col_3.metric("Most Common Class", dominant_class)

    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV Report",
        data=build_csv(history),
        file_name=f"{st.session_state.username}_e_waste_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.button("Clear Saved History", use_container_width=True):
        st.session_state.history = []
        save_user_history(st.session_state.username, [])
        st.rerun()


def render_analytics_panel() -> None:
    history = st.session_state.history
    st.markdown(
        """
        <div class="info-card">
            <div class="section-kicker">Insights</div>
            <h3 style="margin-top:0;">Plotly Analytics Dashboard</h3>
            <p style="margin-bottom:0;">Track confidence, class trends, and overall performance patterns across your saved runs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not history:
        st.info("Analyze a few images to unlock your charts.")
        return

    chart_col, band_col = st.columns(2, gap="large")
    with chart_col:
        st.plotly_chart(class_distribution_figure(history), use_container_width=True)
    with band_col:
        st.plotly_chart(confidence_breakdown_figure(history), use_container_width=True)

    low_confidence_items = [item for item in history if item["confidence_level"] == "Low confidence"]
    if low_confidence_items:
        st.warning(
            f"{len(low_confidence_items)} item(s) were classified with low confidence. "
            "Those are strong candidates for retesting with cleaner photos."
        )
    else:
        st.success("No low-confidence classifications in this account history.")


def render_detail_tabs(predicted_class: str, confidence: float, top_predictions: list[tuple[str, float]]) -> None:
    details = CLASS_DETAILS[predicted_class]
    overview_tab, guidance_tab, model_tab = st.tabs(
        ["Item Overview", "Disposal Guidance", "Model Notes"]
    )

    with overview_tab:
        st.markdown(
            f"""
            <div class="soft-panel">
                <div class="section-kicker">Classification Summary</div>
                <h3 style="margin-top:0;">{predicted_class}</h3>
                <p>{details["description"]}</p>
                <p><strong>Handling note:</strong> {details['risk']}</p>
                <p style="margin-bottom:0;"><strong>Confidence band:</strong> {confidence_label(confidence)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with guidance_tab:
        st.markdown(
            f"""
            <div class="soft-panel">
                <div class="section-kicker">Action</div>
                <h3 style="margin-top:0;">Recommended Next Step</h3>
                <p>{details["action"]}</p>
                <p style="margin-bottom:0;"><strong>Alternative possibilities:</strong> {top_predictions[1][0]} and {top_predictions[2][0]}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Always follow your local e-waste rules before disposal. Hazardous components may require special handling."
        )

    with model_tab:
        st.markdown(
            f"""
            <div class="soft-panel">
                <div class="section-kicker">Model Info</div>
                <h3 style="margin-top:0;">Technical Notes</h3>
                <p><strong>Architecture:</strong> MobileNetV2-based image classifier</p>
                <p><strong>Input size:</strong> {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}</p>
                <p><strong>Supported categories:</strong> {len(CLASS_NAMES)}</p>
                <p style="margin-bottom:0;"><strong>Active model file:</strong> {MODEL_PATH}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def analyze_single_image(source_file, source_name: str, source_type: str) -> None:
    model = load_model()
    image = Image.open(source_file).convert("RGB")
    predicted_class, confidence, top_predictions = predict_image(image, model)
    record_history(source_type, source_name, top_predictions)

    st.markdown("---")
    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.markdown(
            """
            <div class="result-card">
                <div class="section-kicker">Preview</div>
                <h3 style="margin-top:0;">Uploaded Image</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image, caption=source_name, use_container_width=True)

    with result_col:
        st.markdown(
            """
            <div class="result-card">
                <div class="section-kicker">Prediction</div>
                <h3 style="margin-top:0;">Primary Result</h3>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="confidence-badge">{confidence_label(confidence)}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-highlight">
                <div class="metric-label">Predicted Category</div>
                <div class="metric-value">{predicted_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        if confidence < 0.60:
            st.warning(
                "The model is not very confident about this image. Try another angle or a cleaner background."
            )
        elif confidence < 0.85:
            st.info("This looks reasonable, but a second image could improve certainty.")

        st.subheader("Top Predictions")
        render_prediction_rows(top_predictions)
        st.markdown("</div>", unsafe_allow_html=True)

    render_detail_tabs(predicted_class, confidence, top_predictions)


def analyze_batch_images(files: list) -> None:
    if not files:
        st.info("Upload multiple images to generate a batch report.")
        return

    model = load_model()
    st.subheader("Batch Results")
    progress = st.progress(0)
    results = []

    for index, uploaded in enumerate(files, start=1):
        image = Image.open(uploaded).convert("RGB")
        predicted_class, confidence, top_predictions = predict_image(image, model)
        record_history("batch-upload", uploaded.name, top_predictions)
        results.append(
            {
                "filename": uploaded.name,
                "predicted_class": predicted_class,
                "confidence_percent": round(confidence * 100, 2),
                "confidence_level": confidence_label(confidence),
                "top_2": top_predictions[1][0],
                "top_3": top_predictions[2][0],
            }
        )
        progress.progress(int(index / len(files) * 100))

    st.dataframe(results, use_container_width=True, hide_index=True)

    gallery_cols = st.columns(3)
    for index, uploaded in enumerate(files):
        image = Image.open(uploaded).convert("RGB")
        result = results[index]
        with gallery_cols[index % 3]:
            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
            st.image(image, caption=uploaded.name, use_container_width=True)
            st.write(f"**{result['predicted_class']}**")
            st.caption(
                f"{result['confidence_percent']:.2f}% confidence | Alternatives: {result['top_2']}, {result['top_3']}"
            )
            st.markdown("</div>", unsafe_allow_html=True)


def render_auth_screen() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="section-kicker" style="color:#cfeee3;">Eco Tech Identity</div>
                    <h1 style="color:#f6fffb; margin-bottom:0.55rem;">E-Waste Intelligence Suite</h1>
                    <p style="font-size:1.05rem; margin-bottom:0.8rem; color:#e8f8f1;">
                        A brighter, smarter workspace for electronic waste screening, sorting, and model improvement.
                    </p>
                    <span class="hero-chip">&#9851; Circular workflow</span>
                    <span class="hero-chip">&#128267; Hazard-aware guidance</span>
                    <span class="hero-chip">&#128421; Device detection</span>
                    <div class="symbol-row">
                        <div class="symbol-tile">
                            <span class="symbol-mark">&#9851;</span>
                            <span class="symbol-label">Recycle</span>
                        </div>
                        <div class="symbol-tile">
                            <span class="symbol-mark">&#128241;</span>
                            <span class="symbol-label">Mobile</span>
                        </div>
                        <div class="symbol-tile">
                            <span class="symbol-mark">&#128268;</span>
                            <span class="symbol-label">Battery</span>
                        </div>
                        <div class="symbol-tile">
                            <span class="symbol-mark">&#128421;</span>
                            <span class="symbol-label">Devices</span>
                        </div>
                    </div>
                </div>
                <div class="hero-side">
                    <div class="section-kicker" style="color:#d4f4ea;">Welcome Back</div>
                    <h3 style="color:#f6fffb; margin-top:0;">Sign in to your recycling lab</h3>
                    <p style="color:#e3f5ee; margin-bottom:0.8rem;">Save every result, review colorful analytics, test webcam detection, and keep model experiments in one branded dashboard.</p>
                    <div class="hero-chip" style="background:rgba(255,255,255,0.18);">&#127807; Green intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                success, message = authenticate_account(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                st.error(message)

    with register_tab:
        with st.form("register_form"):
            username = st.text_input("New Username")
            password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                success, message = register_account(username, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="auth-feature-list">
            <div class="auth-feature">
                <div class="section-kicker">History</div>
                <strong>Persistent records</strong>
                <p style="margin:0.4rem 0 0 0;">Store every prediction in a saved account timeline.</p>
            </div>
            <div class="auth-feature">
                <div class="section-kicker">Analytics</div>
                <strong>Visual dashboards</strong>
                <p style="margin:0.4rem 0 0 0;">Review class trends and confidence patterns quickly.</p>
            </div>
            <div class="auth-feature">
                <div class="section-kicker">Training</div>
                <strong>Model iteration</strong>
                <p style="margin:0.4rem 0 0 0;">Create improved model variants from your own datasets.</p>
            </div>
        </div>
        <div class="auth-banner">
            <div class="section-kicker">Designed For E-Waste</div>
            <strong>Colorful eco-tech interface</strong>
            <p style="margin:0.45rem 0 0 0;">The login area now mirrors the recycling theme with device symbols, bright eco colors, and a stronger branded first impression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_webcam_panel() -> None:
    st.markdown(
        """
        <div class="webcam-hero">
            <div class="section-kicker" style="color:#cfeee3;">Live Detection</div>
            <h2 style="color:#f6fffb; margin-bottom:0.45rem;">Webcam Analysis Workspace</h2>
            <p style="margin-bottom:0;">
                Point your camera at an item and review AI classification in a more direct, demo-friendly way.
                Use live streaming if available, or use the quick snapshot fallback below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="webcam-tip">
            <strong>Best webcam results:</strong> keep one object in frame, use bright lighting, and fill most of the view.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
    except ImportError:
        st.info(
            "Live streaming is not active in this environment yet. Install the updated requirements and reopen the app "
            "to enable the real-time webcam stream."
        )
        st.markdown(
            """
            <div class="info-card">
                <div class="section-kicker">Fallback Mode</div>
                <h3 style="margin-top:0;">Quick Snapshot Analysis</h3>
                <p style="margin-bottom:0;">
                    Your browser camera can still capture a photo and run the same analysis workflow even without live streaming.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fallback = st.camera_input("Capture webcam snapshot")
        if fallback is not None:
            analyze_single_image(fallback, "webcam_snapshot.png", "webcam-fallback")
        return

    model = load_model()
    start_live = st.toggle("Enable live webcam stream", value=True, help="Turn on real-time webcam classification.")
    if not start_live:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-kicker">Standby</div>
                <h3 style="margin-top:0;">Live stream is paused</h3>
                <p style="margin-bottom:0;">Turn on the toggle above whenever you want to start webcam analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            image_array = frame.to_ndarray(format="bgr24")
            rgb_array = image_array[:, :, ::-1]
            image = Image.fromarray(rgb_array)
            predicted_class, confidence, _ = predict_image(image, model)
            try:
                import cv2

                annotated = image_array.copy()
                cv2.putText(
                    annotated,
                    f"{predicted_class} {confidence * 100:.1f}%",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (40, 180, 90),
                    2,
                )
                return annotated
            except Exception:
                return image_array

    webrtc_streamer(
        key="live-detection",
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


def retrain_model(dataset_dir: Path, epochs: int, learning_rate: float, output_name: str) -> tuple[bool, str]:
    if not dataset_dir.exists():
        return False, f"Dataset folder not found: {dataset_dir}"
    class_folders = [folder for folder in dataset_dir.iterdir() if folder.is_dir()]
    if len(class_folders) < 2:
        return False, "Dataset folder should contain at least two class subfolders with images."

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=16,
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=16,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(class_folders), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=0)
    output_path = MODEL_PATH.parent / output_name
    model.save(output_path)
    load_model.clear()
    return True, (
        f"Retraining completed. Saved model to {output_path}. "
        f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%"
    )


def render_retrain_panel() -> None:
    st.subheader("Retrain Model")
    st.write(
        "Use a labeled dataset folder where each class has its own subfolder. "
        "Example: `data/training/my_dataset/Battery`, `data/training/my_dataset/Mobile`, and so on."
    )
    st.code(
        "data/training/my_dataset/\n"
        "├── Battery/\n"
        "├── Keyboard/\n"
        "├── Mobile/\n"
        "└── ...",
        language="text",
    )

    dataset_root = st.text_input(
        "Dataset folder",
        value=str(TRAINING_DIR / "my_dataset"),
        help="Point this to a folder containing one subfolder per class.",
    )
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=3)
    learning_rate = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3], value=3e-4)
    output_name = st.text_input("Output model filename", value="e-waste-model-retrained.keras")

    if st.button("Start Retraining", type="primary", use_container_width=True):
        with st.spinner("Training model. This may take a while depending on dataset size and hardware..."):
            success, message = retrain_model(Path(dataset_root), epochs, learning_rate, output_name)
        if success:
            st.success(message)
        else:
            st.error(message)


initialize_state()

if not st.session_state.username:
    render_auth_screen()
    st.stop()

st.sidebar.title("Control Panel")
st.sidebar.markdown(
    """
    <div class="sidebar-symbol">
        <div style="font-size:1.4rem; margin-bottom:0.35rem;">&#9851; &#128241; &#128268;</div>
        <div style="font-weight:800; letter-spacing:0.04em;">E-WASTE LAB</div>
        <div style="font-size:0.86rem; opacity:0.92;">Eco-aware device screening</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(f"Signed in as **{st.session_state.username}**")
st.sidebar.metric("Supported Classes", len(CLASS_NAMES))
st.sidebar.metric("Input Resolution", f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
st.sidebar.markdown("**Accepted formats:** JPG, JPEG, PNG")
if st.sidebar.button("Sign Out", use_container_width=True):
    logout_user()
    st.rerun()
st.sidebar.info(
    "Best results come from a single object in clear lighting with minimal background clutter."
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-grid">
            <div>
                <div class="section-kicker" style="color:#cfeee3;">Colorful Recycling Dashboard</div>
                <h1 style="color:#f6fffb; margin-bottom:0.5rem;">E-Waste Intelligence Suite</h1>
                <p style="font-size:1.05rem; margin-bottom:0.85rem; color:#e7f7f0;">
                    Advanced image-based classification with saved user history, Plotly analytics,
                    webcam workflows, and a built-in retraining workspace.
                </p>
                <span class="hero-chip">&#9851; AI classification</span>
                <span class="hero-chip">&#128247; Batch review</span>
                <span class="hero-chip">&#128249; Live webcam</span>
                <span class="hero-chip">&#9881; Training lab</span>
                <div class="symbol-row">
                    <div class="symbol-tile">
                        <span class="symbol-mark">&#9851;</span>
                        <span class="symbol-label">Sort</span>
                    </div>
                    <div class="symbol-tile">
                        <span class="symbol-mark">&#128187;</span>
                        <span class="symbol-label">Electronics</span>
                    </div>
                    <div class="symbol-tile">
                        <span class="symbol-mark">&#128268;</span>
                        <span class="symbol-label">Power</span>
                    </div>
                    <div class="symbol-tile">
                        <span class="symbol-mark">&#127758;</span>
                        <span class="symbol-label">Planet</span>
                    </div>
                </div>
            </div>
            <div class="hero-side">
                <div class="section-kicker" style="color:#d4f4ea;">Session Status</div>
                <h3 style="color:#f6fffb; margin-top:0;">Ready for colorful analysis</h3>
                <p style="color:#e3f5ee; margin-bottom:0.7rem;">Work across uploads, account history, webcam detection, and retraining from one interface built around e-waste symbols and eco-tech colors.</p>
                <p style="color:#f6fffb; margin:0;"><strong>Signed in:</strong> {st.session_state.username}</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="metric-band">', unsafe_allow_html=True)
metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric_tile("Accounts", "Persistent", "Local sign-in with saved history across sessions.")
with metric_cols[1]:
    render_metric_tile("Analysis", "3 Modes", "Single image, batch upload, and camera capture.")
with metric_cols[2]:
    render_metric_tile("Charts", "Plotly", "Professional visual dashboards for trends and confidence.")
with metric_cols[3]:
    render_metric_tile("Training", "Built In", "Retrain and save improved model variants in-app.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="mode-card">
        <div class="section-kicker">Workspace Navigation</div>
        <h3 style="margin-top:0; margin-bottom:0.35rem;">Choose what you want to do</h3>
        <p style="margin-bottom:0;">Switch between analysis, saved results, charts, live webcam mode, and model training.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
workspace_mode = st.segmented_control(
    "Workspace",
    options=["Analysis Workspace", "Saved History", "Analytics", "Live Webcam", "Retrain Model"],
    selection_mode="single",
    default="Analysis Workspace",
    label_visibility="collapsed",
)

if workspace_mode == "Analysis Workspace":
    st.markdown('<div class="workspace-shell">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.2, 0.8], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-kicker">Primary Workflow</div>
                <h3 style="margin-top:0;">Analysis Workspace</h3>
                <p style="margin-bottom:0;">
                    Run a single prediction or upload several images to compare results in one pass.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        single_tab, batch_tab, camera_tab = st.tabs(["Single Upload", "Batch Upload", "Camera Capture"])
        with single_tab:
            uploaded_file = st.file_uploader(
                "Choose one image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image where the e-waste object is the main subject.",
            )
        with batch_tab:
            batch_files = st.file_uploader(
                "Choose multiple images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Ideal for running a quick comparison set or demo batch.",
            )
        with camera_tab:
            camera_file = st.camera_input("Capture an item using your camera")

    with right_col:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-kicker">Product View</div>
                <h3 style="margin-top:0;">Advanced Capabilities</h3>
                <p style="margin-bottom:0.35rem;">Persistent accounts, better charts, webcam support, and retraining tools.</p>
                <p style="margin-bottom:0;">This now behaves more like a compact AI product than a one-page model demo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Classification Tips", expanded=True):
            st.write("Keep the object centered and avoid multiple devices in the same frame.")
            st.write("Use bright, even lighting for better visual detail.")
            st.write("For batch mode, keep image quality consistent for easier comparison.")

    try:
        if camera_file is not None:
            analyze_single_image(camera_file, "camera_capture.png", "camera")
        elif uploaded_file is not None:
            analyze_single_image(uploaded_file, uploaded_file.name, "single-upload")
        elif batch_files:
            analyze_batch_images(batch_files)
        else:
            st.markdown(
                """
                <div class="info-card">
                    <div class="section-kicker">Start Here</div>
                    <h3 style="margin-top:0;">Ready for Analysis</h3>
                    <p style="margin-bottom:0;">
                        Upload or capture an image to generate a prediction report, or use batch mode for a multi-image review.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error("The app could not process one or more images.")
        st.exception(error)
    st.markdown("</div>", unsafe_allow_html=True)

elif workspace_mode == "Saved History":
    render_history_panel()

elif workspace_mode == "Analytics":
    render_analytics_panel()

elif workspace_mode == "Live Webcam":
    render_live_webcam_panel()

elif workspace_mode == "Retrain Model":
    render_retrain_panel()
