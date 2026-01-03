import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_and_clean_data

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Titanic Survival Intelligence",
    page_icon="🛳️",
    layout="wide"
)

# ==================================================
# THEME STATE
# ==================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# ==================================================
# THEME COLORS
# ==================================================
bg_color = "#000000" if is_dark else "#f8fafc"        # App background (pure black)
card_color = "#0b0b0b" if is_dark else "#ffffff"     # Cards (slightly lifted)
text_color = "#e5e7eb" if is_dark else "#0f172a"     # Soft white text
border_color = "#1f2937" if is_dark else "#e5e7eb"   # Subtle borders

plot_bg = "#000000" if is_dark else "#ffffff"        # Plot background
paper_bg = "#000000" if is_dark else "#ffffff"      # Plot paper
grid_color = "#1f2937" if is_dark else "#e5e7eb"     # Grid lines
font_color = "#e5e7eb" if is_dark else "#0f172a"     # Plot text

# ==================================================
# GLOBAL PLOTLY THEME FIX (NO BLACK BACKGROUND)
# ==================================================
def apply_plotly_theme(fig, title=None):
    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
        title=dict(
            text=title,
            font=dict(color=font_color, size=18),
            x=0.5
        ) if title else None,
        xaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=font_color),
            tickfont=dict(color=font_color)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=font_color),
            tickfont=dict(color=font_color)
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=font_color)
        )
    )
    return fig

# ==================================================
# THEME CSS (TEXT VISIBILITY FIX)
# ==================================================
# ==================================================
# THEME CSS (TEXT VISIBILITY + SINGLE TAB UNDERLINE FIX)
# ==================================================
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background-color: {bg_color};
        transition: background-color 0.4s ease;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700;
    }}

    /* General text */
    p, span, label {{
        color: {text_color} !important;
    }}

    /* Metric card */
    div[data-testid="stMetric"] {{
        background-color: {card_color};
        border-radius: 14px;
        padding: 16px;
        border: 1px solid {border_color};
    }}

    /* Metric label */
    div[data-testid="stMetricLabel"] {{
        color: {text_color} !important;
        font-weight: 600;
        font-size: 0.95rem;
    }}

    /* Metric value */
    div[data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-weight: 800;
        font-size: 1.8rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {card_color};
        border-right: 1px solid {border_color};
    }}

    /* ============================
       Tabs – SINGLE underline fix
       ============================ */

    /* Remove Streamlit default underline */
    button[data-baseweb="tab"] {{
        color: {text_color} !important;
        font-weight: 600;
        border-bottom: none !important;
        box-shadow: none !important;
    }}

    /* Custom single underline (red) */
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #ef4444 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# LOAD MODEL & DATA
# ==================================================
model = joblib.load("models/titanic_model.pkl")

raw_df = pd.read_csv("data/train.csv")
raw_df["Age"] = raw_df["Age"].fillna(raw_df["Age"].median())

processed_df = load_and_clean_data("data/train.csv")

# ==================================================
# SIDEBAR FILTERS
# ==================================================
st.sidebar.header(" Passenger Filters")

class_filter = st.sidebar.multiselect(
    "Passenger Class",
    sorted(raw_df["Pclass"].unique()),
    sorted(raw_df["Pclass"].unique())
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    sorted(raw_df["Sex"].unique()),
    sorted(raw_df["Sex"].unique())
)

age_min, age_max = st.sidebar.slider(
    "Age Range",
    int(raw_df["Age"].min()),
    int(raw_df["Age"].max()),
    (int(raw_df["Age"].min()), int(raw_df["Age"].max()))
)

# ==================================================
# APPLY FILTERS
# ==================================================
filtered_raw = raw_df[
    (raw_df["Pclass"].isin(class_filter)) &
    (raw_df["Sex"].isin(gender_filter)) &
    (raw_df["Age"].between(age_min, age_max))
]

filtered_processed = processed_df.loc[filtered_raw.index]

# ==================================================
# HEADER + THEME TOGGLE
# ==================================================
col_title, col_toggle = st.columns([8, 1])

with col_toggle:
    if st.button("🌙 Dark" if not is_dark else "☀️ Light"):
        st.session_state.theme = "dark" if not is_dark else "light"
        st.rerun()

with col_title:
    st.title("🛳️ Titanic Survival Intelligence")
    st.caption("Advanced ML-powered analytics for predicting passenger survival probability")

# ==================================================
# QUICK STATISTICS
# ==================================================
st.subheader("📊 Quick Statistics")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Passengers", len(filtered_raw))
c2.metric("📅 Average Age", round(filtered_raw["Age"].mean(), 1))
c3.metric("👨 Male", f"{(filtered_raw['Sex']=='male').mean()*100:.1f}%")
c4.metric("👩 Female", f"{(filtered_raw['Sex']=='female').mean()*100:.1f}%")
c5.metric("💚 Avg Survival", f"{filtered_raw['Survived'].mean()*100:.1f}%")

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [" Overview", " Predictions", " Insights", " Analysis"]
)

# ==================================================
# 📈 OVERVIEW
# ==================================================
# ==================================================
# 📈 OVERVIEW
# ==================================================
with tab1:
    if not filtered_raw.empty:

        # --- Row 1 ---
        c1, c2 = st.columns(2)

        fig_age = apply_plotly_theme(
            px.histogram(filtered_raw, x="Age", nbins=30),
            " Age Distribution"
        )
        c1.plotly_chart(fig_age, use_container_width=True)

        fig_class = apply_plotly_theme(
            px.pie(
                filtered_raw,
                names="Pclass",
                hole=0.45
            ),
            " Class Distribution"
        )

        fig_class.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="radial"
        )

        fig_class.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.15
            ),
            margin=dict(
                t=80,
                b=60,
                l=0,
                r=0
            ),
            autosize=True
        )

        c2.plotly_chart(fig_class, use_container_width=True)

        # --- Row 2 ---
        c3, c4 = st.columns(2)

        gender_counts = filtered_raw["Sex"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]

        fig_gender = apply_plotly_theme(
            px.bar(gender_counts, x="Gender", y="Count"),
            " Gender Distribution"
        )
        c3.plotly_chart(fig_gender, use_container_width=True)

        survival_rate = filtered_raw["Survived"].mean() * 100

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=survival_rate,
                number={"suffix": "%"},
                title={"text": " Actual Survival Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22c55e"}
                }
            )
        )

        fig_gauge.update_layout(
            paper_bgcolor=paper_bg,
            font=dict(color=font_color)
        )

        c4.plotly_chart(fig_gauge, use_container_width=True)
    
# ==================================================
# 🔮 PREDICTIONS
# ==================================================
with tab2:
    if not filtered_processed.empty:
        X = filtered_processed.drop("Survived", axis=1)
        filtered_raw["Survival Probability"] = model.predict_proba(X)[:, 1]

        filtered_raw["Survival Chance"] = filtered_raw["Survival Probability"].apply(
            lambda p: "🟢 High" if p >= 0.7 else "🟡 Medium" if p >= 0.4 else "🔴 Low"
        )

        st.dataframe(
            filtered_raw[
                ["PassengerId", "Name", "Sex", "Age", "Pclass",
                 "Survival Chance", "Survival Probability"]
            ].sort_values("Survival Probability", ascending=False),
            use_container_width=True,
            height=420
        )

        st.markdown("###  Survival Category Breakdown")

        high = (filtered_raw["Survival Probability"] >= 0.7).sum()
        medium = ((filtered_raw["Survival Probability"] >= 0.4) &
                  (filtered_raw["Survival Probability"] < 0.7)).sum()
        low = (filtered_raw["Survival Probability"] < 0.4).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 High Survival (≥70%)", f"{high} passengers")
        c2.metric("🟡 Medium Survival (40–70%)", f"{medium} passengers")
        c3.metric("🔴 Low Survival (<40%)", f"{low} passengers")

# ==================================================
# 💡 INSIGHTS
# ==================================================
with tab3:
    if not filtered_raw.empty:
        c1, c2 = st.columns(2)

        fig1 = apply_plotly_theme(
            px.box(filtered_raw, x="Sex", y="Survival Probability", color="Sex"),
            "Survival Probability by Gender"
        )
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = apply_plotly_theme(
            px.box(filtered_raw, x="Pclass", y="Survival Probability", color="Pclass"),
            "Survival Probability by Class"
        )
        c2.plotly_chart(fig2, use_container_width=True)

        fig3 = apply_plotly_theme(
            px.scatter(filtered_raw, x="Age", y="Survival Probability",
                       color="Sex", size="Pclass"),
            "Age vs Survival Probability"
        )
        st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# 🎯 ANALYSIS (KEY FINDINGS)
# ==================================================
with tab4:
    if not filtered_raw.empty:
        st.markdown("###  Key Findings")

        female_rate = filtered_raw[filtered_raw["Sex"] == "female"]["Survived"].mean() * 100
        male_rate = filtered_raw[filtered_raw["Sex"] == "male"]["Survived"].mean() * 100

        c1, c2 = st.columns(2)
        c1.metric("👩 Female Passengers", f"{female_rate:.1f}%")
        c1.caption("Average Survival Rate")

        c2.metric("👨 Male Passengers", f"{male_rate:.1f}%")
        c2.caption("Average Survival Rate")

        st.markdown("###  Survival by Passenger Class")

        cols = st.columns(3)
        for i, cls in enumerate([1, 2, 3]):
            cls_df = filtered_raw[filtered_raw["Pclass"] == cls]
            rate = cls_df["Survived"].mean() * 100
            count = len(cls_df)

            cols[i].metric(f"Class {cls}", f"{rate:.1f}%", f"{count} passengers")

SHOW_FOOTER = False

if SHOW_FOOTER:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:gray;">
            🛳️ Titanic Survival Intelligence<br>
            Powered by Machine Learning & Advanced Analytics | Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True
    )
