"""
Tourism Experience Analytics - Streamlit Application
=====================================================
A production-ready dashboard for classification, prediction,
and recommendation in the tourism domain.
"""

import os
import json
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary: #1a1a2e;
    --accent: #e94560;
    --gold: #f5a623;
    --teal: #0f9b8e;
    --card-bg: #16213e;
    --surface: #0f3460;
    --text: #eaeaea;
    --muted: #9aa5b4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-right: 1px solid rgba(233,69,96,0.2);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 12px;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(233,69,96,0.2);
}
.kpi-card .icon { font-size: 2rem; margin-bottom: 6px; }
.kpi-card .value {
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
    color: #f5a623;
}
.kpi-card .label {
    font-size: 0.82rem;
    color: #9aa5b4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, rgba(15,155,142,0.15), rgba(233,69,96,0.10));
    border-left: 4px solid #0f9b8e;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
}

/* Recommendation card */
.rec-card {
    background: linear-gradient(135deg, #16213e, #0f3460);
    border: 1px solid rgba(245,166,35,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.rec-card .rec-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f5a623;
    font-family: 'Playfair Display', serif;
}
.rec-card .rec-meta {
    font-size: 0.82rem;
    color: #9aa5b4;
    margin-top: 4px;
}

/* Section header */
.section-header {
    border-bottom: 2px solid rgba(233,69,96,0.4);
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(233,69,96,0.5), transparent);
    margin: 20px 0;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-accent { background: rgba(233,69,96,0.2); color: #e94560; border: 1px solid rgba(233,69,96,0.4); }
.badge-gold   { background: rgba(245,166,35,0.2); color: #f5a623; border: 1px solid rgba(245,166,35,0.4); }
.badge-teal   { background: rgba(15,155,142,0.2); color: #0f9b8e; border: 1px solid rgba(15,155,142,0.4); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"

# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """Load all saved model artefacts."""

    import joblib

    models: Dict[str, Any] = {}

    artefacts = {

        # Regression
        "regression_model": MODELS_DIR / "rating_model.pkl",

        # Classification
        "classification_model": MODELS_DIR / "visit_mode_model.pkl",
        "visit_mode_label_encoder": MODELS_DIR / "visit_mode_label_encoder.pkl",

        # Recommendation
        "recommendation_model": MODELS_DIR / "recommender.pkl",
        "item_similarity_df": MODELS_DIR / "item_similarity_df.pkl",
        "user_item_matrix": MODELS_DIR / "user_item_matrix.pkl",
    }

    for key, path in artefacts.items():

        if path.exists():
            try:
                models[key] = joblib.load(path)

            except Exception as e:
                models[key] = None

                st.sidebar.warning(
                    f"⚠️ Could not load `{path.name}`: {e}"
                )

        else:
            models[key] = None

    return models


@st.cache_data(show_spinner=False)
def load_data() -> Optional[pd.DataFrame]:
    """Load processed dataset; fall back to merging raw CSVs."""
    proc_path = PROC_DIR / "final_processed_data.csv"
    if proc_path.exists():
        try:
            return pd.read_csv(proc_path, low_memory=False)
        except Exception:
            pass

    # Attempt raw merge
    try:
        tx   = pd.read_csv(RAW_DIR / "transaction.csv",  low_memory=False)
        user = pd.read_csv(RAW_DIR / "user.csv",         low_memory=False)
        item = pd.read_csv(RAW_DIR / "item.csv",         low_memory=False)

        optional = {
            "type_df":    ("type.csv",       "AttractionTypeId"),
            "mode_df":    ("visit_mode.csv",  None),
            "city_df":    ("city.csv",        "CityId"),
            "country_df": ("country.csv",     "CountryId"),
            "region_df":  ("region.csv",      "RegionId"),
            "continent_df":("continent.csv",  "ContinentId"),
        }

        df = tx.merge(user, on="UserId", how="left")
        df = df.merge(item, left_on="AttractionId", right_on="AttractionId", how="left")

        for name, (fname, key) in optional.items():
            fpath = RAW_DIR / fname
            if fpath.exists() and key:
                try:
                    tmp = pd.read_csv(fpath, low_memory=False)
                    if key in df.columns and key in tmp.columns:
                        df = df.merge(tmp, on=key, how="left")
                except Exception:
                    pass
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> Tuple[Dict, Dict, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load optional metric JSON / CSV files."""
    def _json(path: Path) -> Dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _csv(path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return None

    reg_m  = _json(MODELS_DIR / "regression_metrics.json")
    cls_m  = _json(MODELS_DIR / "classification_metrics.json")
    cmp_df = _csv(MODELS_DIR / "model_comparison.csv")
    fi_df  = _csv(MODELS_DIR / "feature_importance.csv")
    return reg_m, cls_m, cmp_df, fi_df


# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────
def safe_predict(model, X: pd.DataFrame):
    """Predict with robust error handling."""
    if model is None:
        raise ValueError("Model not loaded.")
    return model.predict(X)


def _build_input_row(
    continent: str,
    region: str,
    country: str,
    city: str,
    attraction_type: str,
    visit_year: int,
    visit_month: int,
    visit_mode: str,
) -> pd.DataFrame:

    row_data = {
        "Continent": continent,
        "Region": region,
        "Country": country,
        "CityName": city,
        "AttractionType": attraction_type,
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "VisitMode": visit_mode,
        "Rating": 4.0
    }

    return pd.DataFrame([row_data])

    
def predict_rating(
    models: Dict,
    row_df: pd.DataFrame
) -> Optional[float]:

    model = models.get("regression_model")

    try:
        if model is None:
            raise ValueError("Regression model not loaded.")

        prediction = model.predict(row_df)[0]

        return float(np.clip(prediction, 1, 5))

    except Exception as e:
        st.error(f"Rating prediction failed: {e}")
        return None


def predict_visit_mode(
    models: Dict,
    row_df: pd.DataFrame
) -> Tuple[Optional[str], Optional[np.ndarray]]:

    model = models.get("classification_model")
    label_encoder = models.get("visit_mode_label_encoder")

    try:
        if model is None:
            raise ValueError("Classification model not loaded.")

        pred_class = model.predict(row_df)[0]

        proba = (
            model.predict_proba(row_df)[0]
            if hasattr(model, "predict_proba")
            else None
        )

        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform(
                [int(pred_class)]
            )[0]
        else:
            pred_label = str(pred_class)

        return pred_label, proba

    except Exception as e:
        st.error(f"Visit mode prediction failed: {e}")
        return None, None


def get_recommendations(
    models: Dict,
    df: Optional[pd.DataFrame],
    attraction_type: str,
    country: str,
    top_n: int,
    user_id: Optional[int] = None,
) -> pd.DataFrame:
    """Return top-N recommended attractions."""
    rec_model = models.get("recommendation_model")

    # Model-based recommendation
    if rec_model is not None and user_id is not None:
        try:
            if hasattr(rec_model, "recommend"):
                result = rec_model.recommend(user_id, N=top_n)
                return pd.DataFrame(result)
        except Exception:
            pass

    # Content-based fallback using the dataset
    if df is not None and not df.empty:
        cols_needed = ["Attraction", "AttractionType", "CityName", "Country", "Rating"]
        avail = [c for c in cols_needed if c in df.columns]
        rec_df = df[avail].dropna().copy()

        if "AttractionType" in rec_df.columns and attraction_type != "All":
            rec_df = rec_df[rec_df["AttractionType"] == attraction_type]
        if "Country" in rec_df.columns and country != "All":
            rec_df = rec_df[rec_df["Country"] == country]

        if "Rating" in rec_df.columns and "Attraction" in rec_df.columns:
            rec_df = (
                rec_df.groupby(
                    [c for c in ["Attraction", "AttractionType", "CityName", "Country"] if c in rec_df.columns],
                    as_index=False,
                )["Rating"]
                .mean()
                .sort_values("Rating", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            rec_df.index += 1
            return rec_df

    return pd.DataFrame({"Message": ["No recommendations available. Load data to enable this feature."]})


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def chart_rating_dist(df: pd.DataFrame) -> go.Figure:
    counts = df["Rating"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str),
        y=counts.values,
        marker=dict(color=["#e94560", "#f5a623", "#0f9b8e", "#4a90e2", "#9b59b6"]),
        text=counts.values,
        textposition="auto",
    ))
    fig.update_layout(title="⭐ Rating Distribution", xaxis_title="Rating", yaxis_title="Count", **DARK_TEMPLATE)
    return fig


def chart_top_attractions(df: pd.DataFrame, n: int = 10) -> go.Figure:
    col = "Attraction" if "Attraction" in df.columns else None
    if col is None:
        return go.Figure()
    top = (
        df.groupby(col)["Rating"].mean()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    fig = px.bar(
        top, x="Rating", y=col, orientation="h",
        color="Rating", color_continuous_scale="Plasma",
        title=f"🏆 Top {n} Attractions by Avg Rating",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), **DARK_TEMPLATE)
    return fig


def chart_visit_mode(df: pd.DataFrame) -> go.Figure:
    col = next((c for c in ["VisitMode", "Visit Mode"] if c in df.columns), None)
    if col is None:
        return go.Figure()
    counts = df[col].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.42,
        marker=dict(colors=["#e94560", "#f5a623", "#0f9b8e", "#4a90e2", "#9b59b6"]),
    ))
    fig.update_layout(title="🧳 Visit Mode Distribution", **DARK_TEMPLATE)
    return fig


def chart_top_countries(df: pd.DataFrame, n: int = 10) -> go.Figure:
    if "Country" not in df.columns or "Rating" not in df.columns:
        return go.Figure()
    top = (
        df.groupby("Country")["Rating"].count()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top.columns = ["Country", "Visits"]
    fig = px.bar(
        top, x="Country", y="Visits",
        color="Visits", color_continuous_scale="Teal",
        title=f"🌍 Top {n} Countries by Visits",
    )
    fig.update_layout(**DARK_TEMPLATE)
    return fig


def chart_monthly_trend(df: pd.DataFrame) -> go.Figure:
    if "VisitMonth" not in df.columns or "Rating" not in df.columns:
        return go.Figure()
    monthly = df.groupby("VisitMonth")["Rating"].mean().reset_index()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["Month"] = monthly["VisitMonth"].apply(lambda x: months[int(x)-1] if 1 <= int(x) <= 12 else x)
    fig = go.Figure(go.Scatter(
        x=monthly["Month"], y=monthly["Rating"],
        mode="lines+markers",
        line=dict(color="#e94560", width=3),
        marker=dict(size=8, color="#f5a623"),
    ))
    fig.update_layout(title="📅 Monthly Avg Rating Trend", **DARK_TEMPLATE)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <div style='font-size:2.6rem;'>🌍</div>
            <div style='font-family:"Playfair Display",serif; font-size:1.2rem; color:#f5a623; font-weight:700;'>Tourism Analytics</div>
            <div style='font-size:0.74rem; color:#9aa5b4; letter-spacing:2px; text-transform:uppercase; margin-top:4px;'>Experience Intelligence</div>
        </div>
        <hr style='border-color:rgba(233,69,96,0.3); margin:14px 0;'>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            options=[
                "🏠  Home Dashboard",
                "⭐  Rating Prediction",
                "🧳  Visit Mode Prediction",
                "💡  Recommendation System",
                "🔍  Data Explorer",
                "📊  Model Performance",
                "ℹ️  About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(233,69,96,0.3); margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.75rem; color:#9aa5b4; text-align:center; padding-bottom:10px;'>
            <b style='color:#e94560;'>Domain</b> · Tourism<br>
            <b style='color:#e94560;'>Models</b> · Regression · Classification · CBF<br>
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────
# SHARED INPUT FORM (USED BY BOTH PREDICTION PAGES)
# ─────────────────────────────────────────────
def _dropdown_options(df: Optional[pd.DataFrame], col: str, fallback: List[str]) -> List[str]:
    if df is not None and col in df.columns:
        vals = sorted(df[col].dropna().unique().tolist())
        return vals if vals else fallback
    return fallback

def shared_input_form(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Shared input form used by:
    - Rating Prediction
    - Visit Mode Prediction
    """

    # ------------------------------------------------------------
    # Dropdown options
    # ------------------------------------------------------------
    continents = _dropdown_options(
        df,
        "Continent",
        ["Asia", "Europe", "Americas", "Africa", "Oceania"]
    )

    regions = _dropdown_options(
        df,
        "Region",
        ["East Asia", "Western Europe", "South America", "South Asia"]
    )

    countries = _dropdown_options(
        df,
        "Country",
        ["USA", "France", "Japan", "India", "Brazil"]
    )

    cities = _dropdown_options(
        df,
        "CityName",
        ["Paris", "Tokyo", "New York", "Mumbai", "Sydney"]
    )

    att_types = _dropdown_options(
        df,
        "AttractionType",
        ["Beach", "Museum", "Park", "Historical Site", "Adventure"]
    )

    visit_modes = _dropdown_options(
        df,
        "VisitMode",
        ["Business", "Family", "Couples", "Friends", "Solo"]
    )

    # ------------------------------------------------------------
    # Convert all values to strings
    # ------------------------------------------------------------
    visit_modes = [str(v) for v in visit_modes]

    # Remove duplicates and sort
    visit_modes = sorted(list(set(visit_modes)))

    # Fallback if empty
    if len(visit_modes) == 0:
        visit_modes = [
            "Business",
            "Couples",
            "Family",
            "Friends",
            "Solo"
        ]

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------
    c1, c2 = st.columns(2)

    with c1:
        continent = st.selectbox(
            "🌐 Continent",
            continents
        )

        region = st.selectbox(
            "📍 Region",
            regions
        )

        country = st.selectbox(
            "🏳 Country",
            countries
        )

        city = st.selectbox(
            "🏙 City",
            cities
        )

    with c2:
        attraction_type = st.selectbox(
            "🎡 Attraction Type",
            att_types
        )

        visit_mode = st.selectbox(
            "🧳 Visit Mode",
            visit_modes
        )

        visit_year = st.slider(
            "📅 Visit Year",
            min_value=2010,
            max_value=2024,
            value=2023
        )

        visit_month = st.slider(
            "📅 Visit Month",
            min_value=1,
            max_value=12,
            value=6,
            help="1 = January, 12 = December"
        )

    # ------------------------------------------------------------
    # Return user inputs
    # ------------------------------------------------------------
    return {
        "continent": continent,
        "region": region,
        "country": country,
        "city": city,
        "attraction_type": attraction_type,
        "visit_mode": visit_mode,
        "visit_year": visit_year,
        "visit_month": visit_month,
    }

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

# ── HOME ──────────────────────────────────────
def page_home(df: Optional[pd.DataFrame]) -> None:
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px;'>
        <div style='font-size:3.2rem; font-family:"Playfair Display",serif; color:#f5a623;'>Tourism Experience Analytics</div>
        <div style='color:#9aa5b4; font-size:1rem; margin-top:8px; letter-spacing:1px;'>
            Classification · Prediction · Recommendation · Intelligence
        </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS
    if df is not None and not df.empty:
        n_users      = df["UserId"].nunique()        if "UserId"      in df.columns else "–"
        n_attrs      = df["AttractionId"].nunique()  if "AttractionId" in df.columns else "–"
        n_countries  = df["Country"].nunique()       if "Country"     in df.columns else "–"
        avg_rating   = f"{df['Rating'].mean():.2f}"  if "Rating"      in df.columns else "–"
        n_modes      = df["VisitMode"].nunique()     if "VisitMode"   in df.columns else "–"
    else:
        n_users = n_attrs = n_countries = n_modes = "–"
        avg_rating = "–"

    kpis = [
        ("👤", n_users,     "Total Users"),
        ("🏛",  n_attrs,     "Attractions"),
        ("🌍", n_countries, "Countries"),
        ("⭐", avg_rating,  "Avg Rating"),
        ("🧳", n_modes,     "Visit Modes"),
    ]

    cols = st.columns(5)
    for col, (icon, val, label) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='icon'>{icon}</div>
                <div class='value'>{val}</div>
                <div class='label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if df is not None and not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            if "Rating" in df.columns:
                st.plotly_chart(chart_rating_dist(df), use_container_width=True)
        with c2:
            if "VisitMode" in df.columns:
                st.plotly_chart(chart_visit_mode(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Attraction" in df.columns and "Rating" in df.columns:
                st.plotly_chart(chart_top_attractions(df), use_container_width=True)
        with c4:
            if "Country" in df.columns and "Rating" in df.columns:
                st.plotly_chart(chart_top_countries(df), use_container_width=True)

        if "VisitMonth" in df.columns and "Rating" in df.columns:
            st.plotly_chart(chart_monthly_trend(df), use_container_width=True)
    else:
        st.info("📂 Load data files to see charts and KPIs. Place CSVs in `data/raw/` or `data/processed/`.")

    # Key insights
    st.markdown("<div class='section-header'><h3>💡 Key Business Insights</h3></div>", unsafe_allow_html=True)
    insights = [
        "📍 Personalized recommendations increase user engagement and reduce churn in travel platforms.",
        "📊 Predicting visit mode enables targeted marketing — family vs business packages drive higher conversion.",
        "🌟 Attractions with consistently high ratings can be promoted as premium experiences.",
        "🗺 Geographic clustering of popular attractions helps agencies plan regional campaigns.",
        "📅 Seasonal rating trends allow tourism boards to optimize promotional timing.",
    ]
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)


# ── REGRESSION ────────────────────────────────
def page_regression(df: Optional[pd.DataFrame], models: Dict) -> None:
    st.markdown(
        "<div class='section-header'><h2>⭐ Attraction Rating Prediction</h2></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "Enter your trip details and we'll predict the **rating** you are likely "
        "to give to a tourist attraction.",
        unsafe_allow_html=True,
    )

    # Input Form
    with st.form("regression_form"):
        inputs = shared_input_form(df)
        submitted = st.form_submit_button(
            "🔮 Predict Rating",
            use_container_width=True
        )

    # Prediction Logic
    if submitted:
        # Build a complete input row using one sample row from the dataset
        row_df = _build_input_row(**inputs)

        # Predict rating
        rating = predict_rating(models, row_df)

        # Show result
        if rating is not None:
            c1, c2 = st.columns([1, 1])

            # Gauge Chart
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rating,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": "Predicted Rating",
                        "font": {"size": 22, "color": "#f5a623"}
                    },
                    number={
                        "font": {"size": 56, "color": "#f5a623"},
                        "suffix": " / 5"
                    },
                    gauge={
                        "axis": {"range": [1, 5], "tickcolor": "#eaeaea"},
                        "bar": {"color": "#e94560"},
                        "steps": [
                            {"range": [1, 2], "color": "#2d1b1b"},
                            {"range": [2, 3], "color": "#2d2519"},
                            {"range": [3, 4], "color": "#1b2d27"},
                            {"range": [4, 5], "color": "#1a2a1a"},
                        ],
                        "threshold": {
                            "line": {"color": "#f5a623", "width": 4},
                            "thickness": 0.8,
                            "value": rating,
                        },
                    },
                ))

                fig.update_layout(**DARK_TEMPLATE, height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Rating Summary Card
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)

                if rating >= 4.5:
                    emoji = "🌟"
                    msg = "Outstanding experience expected!"
                    badge = "badge-teal"
                elif rating >= 3.5:
                    emoji = "😊"
                    msg = "Good experience predicted."
                    badge = "badge-gold"
                elif rating >= 2.5:
                    emoji = "😐"
                    msg = "Average experience expected."
                    badge = "badge-accent"
                else:
                    emoji = "😕"
                    msg = "Below-average experience predicted."
                    badge = "badge-accent"

                st.markdown(f"""
                <div class='kpi-card' style='padding:30px;'>
                    <div style='font-size:2.8rem;'>{emoji}</div>
                    <div class='value'>{rating:.2f}</div>
                    <div style='margin:10px 0;'>
                        <span class='badge {badge}'>Predicted Rating</span>
                    </div>
                    <div class='label'
                         style='font-size:0.95rem; margin-top:10px;'>
                        {msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Model Missing Warning
    elif models.get("regression_model") is None:
        st.warning("⚠️ `rating_model.pkl` not found in `models/`.")

# ── CLASSIFICATION ────────────────────────────
def page_classification(df: Optional[pd.DataFrame], models: Dict) -> None:
    st.markdown(
        "<div class='section-header'><h2>🧳 Visit Mode Prediction</h2></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "Enter trip details to predict the **visit mode** — "
        "Business, Family, Couples, Friends, or Solo."
    )

    # Input Form
    with st.form("classification_form"):
        inputs = shared_input_form(df)
        submitted = st.form_submit_button(
            "🔮 Predict Visit Mode",
            use_container_width=True
        )

    # Prediction Logic
    if submitted:
        # Build input row from one sample dataset row
        row_df = _build_input_row(**inputs)

        # Predict visit mode
        pred_mode, proba = predict_visit_mode(models, row_df)

        # Display result
        if pred_mode is not None:
            mode_icons = {
                "Business": "💼",
                "Family": "👨‍👩‍👧",
                "Couples": "💑",
                "Friends": "👫",
                "Solo": "🧍"
            }

            icon = mode_icons.get(pred_mode, "🧳")

            c1, c2 = st.columns([1, 1])

            # Prediction Card
            with c1:
                st.markdown(f"""
                <div class='kpi-card' style='padding:36px;'>
                    <div style='font-size:3rem;'>{icon}</div>
                    <div class='value' style='font-size:2.4rem;'>
                        {pred_mode}
                    </div>
                    <div class='label' style='margin-top:10px;'>
                        Predicted Visit Mode
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability Chart
            with c2:
                if proba is not None:
                    model = models.get("classification_model")

                    # Class labels
                    classes = np.arange(len(proba))

                    # Decode labels
                    label_encoder = models.get(
                        "visit_mode_label_encoder"
                    )

                    if label_encoder is not None:
                        classes = label_encoder.inverse_transform(
                            classes.astype(int)
                        )

                    # Probability DataFrame
                    proba_df = pd.DataFrame({
                        "Visit Mode": [str(c) for c in classes],
                        "Probability": proba
                    }).sort_values(
                        "Probability",
                        ascending=True
                    )

                    # Plot
                    fig = px.bar(
                        proba_df,
                        x="Probability",
                        y="Visit Mode",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale="RdYlGn",
                        title="Class Probabilities",
                        range_x=[0, 1],
                        text=proba_df["Probability"].apply(
                            lambda x: f"{x * 100:.1f}%"
                        ),
                    )

                    fig.update_layout(
                        **DARK_TEMPLATE,
                        height=300
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

    # Model Missing Warning
    elif models.get("classification_model") is None:
        st.warning(
            "⚠️ `visit_mode_model.pkl` not found in `models/`."
        )

# ── RECOMMENDATION ────────────────────────────
def page_recommendation(df: Optional[pd.DataFrame], models: Dict) -> None:
    st.markdown("<div class='section-header'><h2>💡 Personalized Attraction Recommendations</h2></div>", unsafe_allow_html=True)
    st.markdown("Get a ranked list of attractions based on your preferences and similar user behavior.")

    att_types = _dropdown_options(df, "AttractionType", ["Beach","Museum","Park","Historical Site","Adventure"])
    countries  = _dropdown_options(df, "Country",       ["USA","France","Japan","India","Brazil"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        att_type = st.selectbox("🎡 Attraction Type", ["All"] + att_types)
    with c2:
        country  = st.selectbox("🌍 Country", ["All"] + countries)
    with c3:
        top_n    = st.slider("🔢 Top N", 3, 20, 10)
    with c4:
        user_id  = st.number_input("👤 User ID (optional)", min_value=0, value=0, step=1)
        user_id  = int(user_id) if user_id > 0 else None

    if st.button("🚀 Get Recommendations", use_container_width=True):
        with st.spinner("Finding best attractions for you…"):
            recs = get_recommendations(models, df, att_type, country, top_n, user_id)

        if recs.empty or "Message" in recs.columns:
            st.info("No results found. Try different filters or load more data.")
        else:
            st.success(f"✅ Found **{len(recs)}** recommendations!")

            for _, row in recs.iterrows():
                name     = row.get("Attraction", "–")
                a_type   = row.get("AttractionType", "–")
                city     = row.get("CityName", "–")
                ctry     = row.get("Country", "–")
                rating   = f"{row['Rating']:.2f}" if "Rating" in row and pd.notna(row["Rating"]) else "–"

                st.markdown(f"""
                <div class='rec-card'>
                    <div class='rec-title'>🏛 {name}</div>
                    <div class='rec-meta'>
                        <span class='badge badge-teal'>{a_type}</span>&nbsp;
                        <span class='badge badge-gold'>{city}, {ctry}</span>&nbsp;
                        <span class='badge badge-accent'>⭐ {rating}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Download
            csv_bytes = recs.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Recommendations CSV",
                data=csv_bytes,
                file_name="recommendations.csv",
                mime="text/csv",
            )


# ── DATA EXPLORER ─────────────────────────────
def page_data_explorer(df: Optional[pd.DataFrame]) -> None:
    st.markdown("<div class='section-header'><h2>🔍 Data Explorer</h2></div>", unsafe_allow_html=True)

    if df is None or df.empty:
        st.error("No data loaded. Place CSV files in `data/raw/` or `data/processed/`.")
        return

    st.markdown(f"**Shape:** `{df.shape[0]:,} rows × {df.shape[1]} columns`")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Preview", "📋 Data Types & Nulls", "📈 Statistics", "🔥 Correlation"])

    with tab1:
        n = st.slider("Rows to preview", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)

    with tab2:
        info = pd.DataFrame({
            "dtype":    df.dtypes,
            "non-null": df.notnull().sum(),
            "nulls":    df.isnull().sum(),
            "null%":    (df.isnull().mean() * 100).round(2),
            "unique":   df.nunique(),
        })
        st.dataframe(info, use_container_width=True)

    with tab3:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
            )
            fig.update_layout(**DARK_TEMPLATE, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation matrix.")

    # Filters
    with st.expander("🔎 Filter & Explore", expanded=False):
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            col_filter = st.selectbox("Filter by column", cat_cols)
            vals = df[col_filter].dropna().unique().tolist()
            chosen = st.multiselect("Select values", vals, default=vals[:3])
            filtered = df[df[col_filter].isin(chosen)]
            st.write(f"Filtered: **{len(filtered):,} rows**")
            st.dataframe(filtered.head(50), use_container_width=True)


# ── MODEL PERFORMANCE ─────────────────────────
def page_model_performance() -> None:
    st.markdown("<div class='section-header'><h2>📊 Model Performance</h2></div>", unsafe_allow_html=True)
    reg_m, cls_m, cmp_df, fi_df = load_metrics()

    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Regression", "🧳 Classification", "🏆 Comparison", "📌 Feature Importance"])

    with tab1:
       if reg_m:
        regression_metrics = [
            ("R2 Score", "R²", "📐"),
            ("RMSE", "RMSE", "📉"),
            ("MAE", "MAE", "📏")
        ]

        c1, c2, c3 = st.columns(3)

        for col, (json_key, display_label, icon) in zip(
            [c1, c2, c3],
            regression_metrics
        ):
            with col:
                val = reg_m.get(json_key, "–")

                if isinstance(val, (int, float)):
                    val_display = f"{val:.4f}"
                else:
                    val_display = val

                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='icon'>{icon}</div>
                    <div class='value'>{val_display}</div>
                    <div class='label'>{display_label}</div>
                </div>
                """, unsafe_allow_html=True)
       else:
           st.info(
            "No regression metrics file found (`models/regression_metrics.json`)."
          )

    with tab2:
        if cls_m:
            metrics = [("Accuracy","🎯"),("F1 Score","📊"),("Precision","🔬"),("Recall","📡")]
            cols = st.columns(len(metrics))
            for col, (key, icon) in zip(cols, metrics):
                with col:
                    val = cls_m.get(key, cls_m.get(key.lower().replace(" ","_"), "–"))
                    st.markdown(f"""
                    <div class='kpi-card'>
                        <div class='icon'>{icon}</div>
                        <div class='value'>{val if isinstance(val,str) else f"{val:.4f}"}</div>
                        <div class='label'>{key}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No classification metrics file found (`models/classification_metrics.json`).")

    with tab3:
        if cmp_df is not None:
            st.dataframe(cmp_df, use_container_width=True)
            if "Model" in cmp_df.columns and "Accuracy" in cmp_df.columns:
                fig = px.bar(cmp_df, x="Model", y="Accuracy", color="Accuracy",
                             color_continuous_scale="Plasma", title="Model Accuracy Comparison")
                fig.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model comparison file found (`models/model_comparison.csv`).")

    with tab4:
        if fi_df is not None:
            if "Feature" in fi_df.columns and "Importance" in fi_df.columns:
                fi_df_sorted = fi_df.sort_values("Importance", ascending=True).tail(20)
                fig = px.bar(fi_df_sorted, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Viridis",
                             title="Top Feature Importances")
                fig.update_layout(**DARK_TEMPLATE, height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance file found (`models/feature_importance.csv`).")


# ── ABOUT ─────────────────────────────────────
def page_about() -> None:
    st.markdown("<div class='section-header'><h2>ℹ️ About This Project</h2></div>", unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Overview", "🏗 Architecture", "📦 Tech Stack", "💼 Business Use Cases"])

    with tabs[0]:
        st.markdown("""
        ### Problem Statement
        Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide
        personalized recommendations, predict user satisfaction, and classify potential user behavior.

        ### Three Core Objectives
        | # | Task | Description |
        |---|------|-------------|
        | 1 | **Regression** | Predict the rating a user might give to a tourist attraction |
        | 2 | **Classification** | Predict the visit mode (Business, Family, Couples, etc.) |
        | 3 | **Recommendation** | Suggest personalized attractions based on user profile & behavior |
        """)

    with tabs[1]:
        st.markdown("""
        ```
        tourism-experience-analytics/
        ├── app/
        │   └── streamlit_app.py        ← This application
        ├── data/
        │   ├── raw/                    ← Raw CSVs (transaction, user, item, …)
        │   └── processed/
        │       └── final_processed_data.csv
        ├── models/
        │   ├── rating_model.pkl
        │   ├── visit_mode_model.pkl
        │   ├── visit_mode_label_encoder.pkl
        │   ├── recommender.pkl
        │   ├── item_similarity_df.pkl
        │   └── user_item_matrix.pkl
        ├── notebooks/
        │   ├── 01_data_loading.ipynb
        │   ├── 02_eda.ipynb
        │   ├── 03_preprocessing.ipynb
        │   ├── 04_regression.ipynb
        │   ├── 05_classification.ipynb
        │   └── 06_recommendation.ipynb
        ├── requirements.txt
        └── README.md
        ```
        """)

    with tabs[2]:
        technologies = [
            ("🐍 Python",        "Core language"),
            ("🐼 Pandas",        "Data manipulation"),
            ("🔢 NumPy",         "Numerical computing"),
            ("🤖 Scikit-learn",  "ML models"),
            ("💾 Joblib",        "Model serialisation"),
            ("📊 Plotly",        "Interactive charts"),
            ("🖥 Streamlit",     "Web application"),
            ("📓 Jupyter",       "Exploratory notebooks"),
        ]
        c1, c2 = st.columns(2)
        for i, (tech, desc) in enumerate(technologies):
            col = c1 if i % 2 == 0 else c2
            with col:
                st.markdown(f"<div class='insight-box'><b>{tech}</b> — {desc}</div>", unsafe_allow_html=True)

    with tabs[3]:
        use_cases = [
            ("🎯 Personalized Recommendations", "Suggest attractions based on past visits, preferences & demographics."),
            ("📈 Tourism Analytics",            "Insights into popular attractions and regions for business decisions."),
            ("👥 Customer Segmentation",        "Classify users for targeted promotions."),
            ("🔄 Retention Improvement",        "Boost loyalty through hyper-personalization."),
            ("🗺 Destination Planning",         "Help DMOs identify emerging trends and optimize campaigns."),
        ]
        for title, desc in use_cases:
            with st.expander(title):
                st.write(desc)

    st.markdown("""
    <hr style='border-color:rgba(233,69,96,0.3);'>
    <div style='text-align:center; color:#9aa5b4; font-size:0.85rem; padding:16px 0;'>
        Built for the <b style='color:#f5a623;'>Tourism Experience Analytics</b> capstone project · Domain: Tourism
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    # Load shared resources
    with st.spinner("Loading models and data..."):
        models = load_models()
        df = load_data()

        

    # Render sidebar
    page = render_sidebar()

    # Route pages
    if "Home" in page:
        page_home(df)

    elif "Rating" in page:
        page_regression(df, models)

    elif "Visit Mode" in page:
        page_classification(df, models)

    elif "Recommendation" in page:
        page_recommendation(df, models)

    elif "Data Explorer" in page:
        page_data_explorer(df)

    elif "Performance" in page:
        page_model_performance()

    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
