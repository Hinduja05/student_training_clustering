import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Student Analytics Platform",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#2563EB;
}

.sub-title{
    text-align:center;
    color:#6B7280;
    font-size:18px;
    margin-bottom:20px;
}

.stTabs [data-baseweb="tab"]{
    font-size:16px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD FILES
# ==================================================

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustered_student.csv"
)

df = pd.read_csv(csv_path)

model = joblib.load(
    BASE_DIR / "models" / "kmeans_model.pkl"
)
scaler = joblib.load(
    BASE_DIR / "models" / "scaler.pkl"
)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🎓 Student Analytics Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Performance Clustering & Training Recommendation System</div>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👨‍🎓 Total Students",
        len(df)
    )

with col2:
    st.metric(
        "📈 Avg Exam Score",
        round(df["Exam_Score"].mean(), 2)
    )

with col3:
    st.metric(
        "🏆 Highest Score",
        round(df["Exam_Score"].max(), 2)
    )

with col4:
    st.metric(
        "🎯 Total Clusters",
        df["Cluster"].nunique()
    )

st.divider()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard",
        "👨‍🎓 Student Analysis",
        "🎯 Recommendations",
        "📁 Reports"
    ]
)

# ==================================================
# DASHBOARD TAB
# ==================================================

with tab1:

    st.subheader("Cluster Distribution")

    fig1 = px.pie(
        df,
        names="Cluster",
        hole=0.55,
        title="Student Cluster Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader("Performance Analysis")

    fig2 = px.scatter(
        df,
        x="Hours_Studied",
        y="Exam_Score",
        color="Cluster",
        size="Attendance",
        hover_data=["Previous_Score"],
        title="Hours Studied vs Exam Score"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Cluster Statistics")

    cluster_stats = (
        df.groupby("Cluster")
        .mean(numeric_only=True)
        .round(2)
    )

    st.dataframe(
        cluster_stats,
        use_container_width=True
    )

# ==================================================
# STUDENT ANALYSIS TAB
# ==================================================

with tab2:

    st.subheader("Analyze Student")

    with st.form("student_form"):

        c1, c2 = st.columns(2)

        with c1:

            hours = st.number_input(
                "Hours Studied",
                min_value=1,
                max_value=12,
                value=5
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=50,
                max_value=100,
                value=75
            )

            sleep = st.number_input(
                "Sleep Hours",
                min_value=4,
                max_value=10,
                value=7
            )

        with c2:

            previous = st.number_input(
                "Previous Score",
                min_value=0,
                max_value=100,
                value=70
            )

            exam = st.number_input(
                "Exam Score",
                min_value=0,
                max_value=100,
                value=75
            )

        submit = st.form_submit_button(
            "Analyze Student"
        )

    if submit:

        student_data = np.array([
            [
                hours,
                attendance,
                sleep,
                previous,
                exam
            ]
        ])

        scaled_data = scaler.transform(
            student_data
        )

        cluster = model.predict(
            scaled_data
        )[0]

        st.success(
            f"Student Assigned to Cluster {cluster}"
        )

        if cluster == 0:

            st.error("""
### Foundation Learner

**Risk Level:** High

**Recommended Training**
- Aptitude Basics
- Communication Skills
- Python Fundamentals
- Weekly Assessments

**Placement Readiness:** 40%
""")

        elif cluster == 1:

            st.warning("""
### Intermediate Learner

**Risk Level:** Medium

**Recommended Training**
- Data Structures
- SQL
- Problem Solving
- Mini Projects

**Placement Readiness:** 70%
""")

        else:

            st.success("""
### Advanced Learner

**Risk Level:** Low

**Recommended Training**
- Advanced DSA
- System Design
- Mock Interviews
- Industry Projects

**Placement Readiness:** 90%
""")

# ==================================================
# RECOMMENDATIONS TAB
# ==================================================

with tab3:

    st.subheader("Training Programs")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["Cluster"].unique())
    )

    if selected_cluster == 0:

        st.info("""
## Foundation Training Program

### Week 1
- Aptitude Basics

### Week 2
- Python Fundamentals

### Week 3
- Communication Skills

### Week 4
- Assessment Test
""")

    elif selected_cluster == 1:

        st.warning("""
## Intermediate Training Program

### Week 1
- Data Structures

### Week 2
- SQL & DBMS

### Week 3
- Problem Solving

### Week 4
- Mini Project
""")

    else:

        st.success("""
## Advanced Placement Program

### Week 1
- Advanced DSA

### Week 2
- System Design

### Week 3
- Mock Interviews

### Week 4
- Placement Preparation
""")

# ==================================================
# REPORTS TAB
# ==================================================

with tab4:

    st.subheader("Student Database")

    search_index = st.text_input(
        "Search Student Row Number"
    )

    if search_index:

        try:

            idx = int(search_index)

            st.dataframe(
                df.iloc[[idx]],
                use_container_width=True
            )

        except:

            st.error("Invalid Row Number")

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        label="⬇ Download Student Report",
        data=df.to_csv(index=False),
        file_name="student_report.csv",
        mime="text/csv"
    )