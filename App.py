import ast
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "all_rawis.csv"
TARGET = "death_date_gregorian"


st.set_page_config(
    page_title="Hadith Narrators Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #182024;
        --muted: #697176;
        --line: #d9dedc;
        --paper: #fbfaf6;
        --panel: #ffffff;
        --accent: #126c61;
        --accent-2: #a64f2a;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(251,250,246,0.96), rgba(245,247,244,0.96));
        color: var(--ink);
    }
    section[data-testid="stSidebar"] {
        background: #f1f4ef;
        border-right: 1px solid var(--line);
    }
    .hero {
        padding: 1.1rem 0 0.4rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.05;
        margin: 0;
        letter-spacing: 0;
        color: var(--ink);
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 920px;
        margin-top: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 22px rgba(30, 42, 38, 0.04);
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent);
    }
    .section-note {
        color: var(--muted);
        margin-top: -0.35rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    for col in [
        "birth_date_hijri",
        "birth_date_gregorian",
        "death_date_hijri",
        "death_date_gregorian",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"birth_date_gregorian", "death_date_gregorian"}.issubset(df.columns):
        df["lifespan"] = df["death_date_gregorian"] - df["birth_date_gregorian"]
    return add_engineered_features(df)


def parse_list_cell(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, (str, int, float)):
            return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        pass

    return [part.strip() for part in re.split(r",|;|\|", text) if part.strip()]


def count_items(value):
    return len(parse_list_cell(value))


def add_engineered_features(df):
    data = df.copy()
    source_cols = {
        "num_teachers": "teachers_inds",
        "num_students": "students_inds",
        "num_tags": "tags",
        "num_interest_areas": "area_of_interest",
        "num_places_of_stay": "places_of_stay",
        "num_children": "children",
        "num_siblings": "siblings",
    }
    for new_col, source_col in source_cols.items():
        if source_col in data.columns:
            data[new_col] = data[source_col].apply(count_items)
        else:
            data[new_col] = 0

    for col in ["spouse", "siblings", "children", "books", "parents"]:
        data[f"has_{col}"] = data[col].notna().astype(int) if col in data.columns else 0

    return data


def exploded_counts(df, column, top_n):
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    values = df[column].apply(parse_list_cell).explode()
    counts = (
        values.dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: series != ""]
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )
    return counts


def value_counts_frame(df, column, top_n):
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    return (
        df[column]
        .fillna("Missing")
        .astype(str)
        .str.strip()
        .replace("", "Missing")
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )


def horizontal_bar(data, label_col, title, color="#126c61"):
    if data.empty:
        st.info("No values available for this selection.")
        return
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, color=color)
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{label_col}:N", sort="-x", title=None),
            tooltip=[alt.Tooltip(f"{label_col}:N", title=label_col), "count:Q"],
        )
        .properties(title=title, height=max(280, min(620, 28 * len(data))))
    )
    st.altair_chart(chart, use_container_width=True)


def histogram(data, column, title, bins=30, color="#a64f2a"):
    clean = data[[column]].dropna()
    if clean.empty:
        st.info("No numeric values available for this chart.")
        return
    chart = (
        alt.Chart(clean)
        .mark_bar(color=color, opacity=0.88)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=bins), title=column.replace("_", " ").title()),
            y=alt.Y("count():Q", title="Records"),
            tooltip=[alt.Tooltip("count():Q", title="Records")],
        )
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def metric_row(df):
    total = len(df)
    target_known = int(df[TARGET].notna().sum()) if TARGET in df.columns else 0
    birth_known = int(df["birth_date_gregorian"].notna().sum()) if "birth_date_gregorian" in df.columns else 0
    grades = int(df["grade"].nunique(dropna=True)) if "grade" in df.columns else 0
    places = int(df["birth_place"].nunique(dropna=True)) if "birth_place" in df.columns else 0

    cols = st.columns(5)
    cols[0].metric("Narrators", f"{total:,}")
    cols[1].metric("Known death years", f"{target_known:,}", f"{target_known / total:.1%}" if total else None)
    cols[2].metric("Known birth years", f"{birth_known:,}", f"{birth_known / total:.1%}" if total else None)
    cols[3].metric("Grade classes", f"{grades:,}")
    cols[4].metric("Birth places", f"{places:,}")


def show_overview(df):
    metric_row(df)
    st.markdown("### Dataset Snapshot")
    left, right = st.columns([1.2, 1])
    with left:
        missing = (
            (df.isna().mean() * 100)
            .round(2)
            .sort_values(ascending=False)
            .rename_axis("column")
            .reset_index(name="missing_percent")
        )
        chart = (
            alt.Chart(missing.head(12))
            .mark_bar(cornerRadiusEnd=3, color="#7c8f3f")
            .encode(
                x=alt.X("missing_percent:Q", title="Missing %"),
                y=alt.Y("column:N", sort="-x", title=None),
                tooltip=["column:N", "missing_percent:Q"],
            )
            .properties(title="Most Missing Columns", height=360)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        if "grade" in df.columns:
            horizontal_bar(value_counts_frame(df, "grade", 10), "grade", "Top Grades", "#126c61")

    st.markdown("### Numeric Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)


def show_quality(df):
    st.markdown("### Missing Values")
    st.markdown('<p class="section-note">This combines the missing-value lab with a column-level completeness audit.</p>', unsafe_allow_html=True)
    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_percent": (df.isna().mean() * 100).round(2).values,
                "dtype": [str(dtype) for dtype in df.dtypes],
            }
        )
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(missing, use_container_width=True, hide_index=True)

    chart = (
        alt.Chart(missing)
        .mark_bar(cornerRadiusEnd=3, color="#a64f2a")
        .encode(
            x=alt.X("missing_percent:Q", title="Missing %"),
            y=alt.Y("column:N", sort="-x", title=None),
            tooltip=["column:N", "missing_count:Q", "missing_percent:Q", "dtype:N"],
        )
        .properties(height=620)
    )
    st.altair_chart(chart, use_container_width=True)


def show_categories(df):
    st.markdown("### Categorical Exploration")
    regular_cols = [col for col in ["grade", "birth_place", "death_place", "death_reason"] if col in df.columns]
    list_cols = [col for col in ["area_of_interest", "places_of_stay", "tags", "teachers", "students"] if col in df.columns]

    col1, col2 = st.columns(2)
    with col1:
        selected = st.selectbox("Single-value column", regular_cols, index=0 if regular_cols else None)
        top_n = st.slider("Top categories", 5, 40, 15, key="regular_top")
        if selected:
            horizontal_bar(value_counts_frame(df, selected, top_n), selected, f"Top {selected.replace('_', ' ').title()}", "#126c61")

    with col2:
        selected_list = st.selectbox("List-like column", list_cols, index=0 if list_cols else None)
        list_top_n = st.slider("Top list values", 5, 40, 15, key="list_top")
        if selected_list:
            horizontal_bar(exploded_counts(df, selected_list, list_top_n), selected_list, f"Top {selected_list.replace('_', ' ').title()}", "#7c8f3f")


def show_chronology(df):
    st.markdown("### Chronology and Lifespan")
    col1, col2 = st.columns(2)
    with col1:
        histogram(df, "birth_date_gregorian", "Birth Year Distribution", 35, "#126c61")
    with col2:
        histogram(df, TARGET, "Death Year Distribution", 35, "#a64f2a")

    if "lifespan" in df.columns:
        clean_life = df[(df["lifespan"].notna()) & (df["lifespan"] >= 0) & (df["lifespan"] <= 130)]
        histogram(clean_life, "lifespan", "Lifespan Distribution", 30, "#7c8f3f")
        if not clean_life.empty:
            cols = st.columns(4)
            cols[0].metric("Known lifespans", f"{len(clean_life):,}")
            cols[1].metric("Mean lifespan", f"{clean_life['lifespan'].mean():.1f}")
            cols[2].metric("Median lifespan", f"{clean_life['lifespan'].median():.1f}")
            cols[3].metric("Max filtered lifespan", f"{clean_life['lifespan'].max():.0f}")

    if {"birth_date_gregorian", TARGET}.issubset(df.columns):
        scatter_df = df[["birth_date_gregorian", TARGET, "grade", "name"]].dropna()
        if not scatter_df.empty:
            chart = (
                alt.Chart(scatter_df)
                .mark_circle(size=54, opacity=0.7, color="#126c61")
                .encode(
                    x=alt.X("birth_date_gregorian:Q", title="Birth Year"),
                    y=alt.Y(f"{TARGET}:Q", title="Death Year"),
                    tooltip=["name:N", "grade:N", "birth_date_gregorian:Q", f"{TARGET}:Q"],
                )
                .properties(title="Birth Year vs Death Year", height=420)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)


def train_regression(df, include_calendar_leakage):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        return None, f"scikit-learn is not installed in this Python environment: {exc}"

    data = df.copy()
    original_rows = len(data)
    data = data.dropna(subset=[TARGET])
    dropped_rows = original_rows - len(data)

    numeric_features = [
        "birth_date_hijri",
        "birth_date_gregorian",
        "num_teachers",
        "num_students",
        "num_tags",
        "num_interest_areas",
        "num_places_of_stay",
        "num_children",
        "num_siblings",
        "has_spouse",
        "has_siblings",
        "has_children",
        "has_books",
        "has_parents",
    ]
    if include_calendar_leakage:
        numeric_features.append("death_date_hijri")

    categorical_features = ["grade", "birth_place", "death_place", "death_reason", "area_of_interest"]
    numeric_features = [col for col in numeric_features if col in data.columns]
    categorical_features = [col for col in categorical_features if col in data.columns]

    X = data[numeric_features + categorical_features]
    y = data[TARGET]

    if len(data) < 5:
        return None, "Not enough rows with a known death_date_gregorian value."

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_int = np.round(y_pred).astype(int)

    y_true = y_test.to_numpy()
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ssr = float(sst - sse)

    results = pd.DataFrame(
        {
            "actual_death_year": y_true.astype(int),
            "predicted_death_year": y_pred_int,
            "absolute_error": np.abs(y_true - y_pred),
        }
    ).sort_values("absolute_error", ascending=False)

    metrics = {
        "original_rows": original_rows,
        "dropped_rows": dropped_rows,
        "used_rows": len(data),
        "features": len(numeric_features) + len(categorical_features),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "sse": sse,
        "ssr": ssr,
        "sst": sst,
        "results": results,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return metrics, None


def show_regression(df):
    st.markdown("### Linear Regression: Predict Death Year")
    st.markdown(
        '<p class="section-note">The model trains only on records where death_date_gregorian is known. Predictions are displayed as integer years.</p>',
        unsafe_allow_html=True,
    )
    include_leakage = st.toggle(
        "Include death_date_hijri as a feature",
        value=False,
        help="This can improve scores but leaks the target through the Hijri calendar equivalent.",
    )
    metrics, error = train_regression(df, include_leakage)
    if error:
        st.warning(error)
        return

    cols = st.columns(6)
    cols[0].metric("Rows used", f"{metrics['used_rows']:,}")
    cols[1].metric("Rows dropped", f"{metrics['dropped_rows']:,}")
    cols[2].metric("Features", f"{metrics['features']}")
    cols[3].metric("MAE", f"{metrics['mae']:.2f} years")
    cols[4].metric("RMSE", f"{metrics['rmse']:.2f} years")
    cols[5].metric("R2", f"{metrics['r2']:.3f}")

    ss_cols = st.columns(3)
    ss_cols[0].metric("SSE", f"{metrics['sse']:,.0f}")
    ss_cols[1].metric("SSR", f"{metrics['ssr']:,.0f}")
    ss_cols[2].metric("SST", f"{metrics['sst']:,.0f}")

    st.markdown("#### Largest Prediction Errors")
    st.dataframe(metrics["results"].head(20), use_container_width=True, hide_index=True)

    chart_data = metrics["results"].copy()
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=60, opacity=0.72, color="#a64f2a")
        .encode(
            x=alt.X("actual_death_year:Q", title="Actual Death Year"),
            y=alt.Y("predicted_death_year:Q", title="Predicted Death Year"),
            tooltip=["actual_death_year:Q", "predicted_death_year:Q", "absolute_error:Q"],
        )
        .properties(title="Actual vs Predicted Death Year", height=430)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Features used by the model"):
        st.write("Numeric features")
        st.code(", ".join(metrics["numeric_features"]) or "None")
        st.write("Categorical features")
        st.code(", ".join(metrics["categorical_features"]) or "None")


def show_data_explorer(df):
    st.markdown("### Raw Data Explorer")
    search = st.text_input("Search narrator name")
    filtered = df.copy()
    if search and "name" in filtered.columns:
        filtered = filtered[filtered["name"].astype(str).str.contains(search, case=False, na=False)]

    cols = st.multiselect(
        "Columns to display",
        df.columns.tolist(),
        default=[col for col in ["scholar_indx", "name", "grade", "birth_date_gregorian", TARGET, "birth_place", "death_place"] if col in df.columns],
    )
    st.caption(f"Showing {len(filtered):,} records")
    st.dataframe(filtered[cols] if cols else filtered, use_container_width=True, hide_index=True)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

st.markdown(
    """
    <div class="hero">
        <h1>Hadith Narrators Creative Dashboard</h1>
        <p>
            A combined dashboard for the rawis labs: inspection, missing values,
            categorical distributions, chronology, engineered features, and
            linear regression for death_date_gregorian.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard Controls")
    st.caption("Dataset: all_rawis.csv")
    grade_filter = st.multiselect(
        "Filter by grade",
        sorted(df["grade"].dropna().astype(str).unique().tolist()) if "grade" in df.columns else [],
        default=[],
    )
    if grade_filter and "grade" in df.columns:
        df_view = df[df["grade"].astype(str).isin(grade_filter)].copy()
    else:
        df_view = df.copy()

    st.divider()
    st.write("Filtered records")
    st.metric("Rows", f"{len(df_view):,}")

tabs = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Categories",
        "Chronology",
        "Regression",
        "Data Explorer",
    ]
)

with tabs[0]:
    show_overview(df_view)
with tabs[1]:
    show_quality(df_view)
with tabs[2]:
    show_categories(df_view)
with tabs[3]:
    show_chronology(df_view)
with tabs[4]:
    show_regression(df_view)
with tabs[5]:
    show_data_explorer(df_view)
