import streamlit as st
import pandas as pd
from resume_matcher import get_top_resumes

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.title {
    font-size: 70px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<p class="title">📄 SmartHire AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Resume Screening & Job Matching System</p>',
    unsafe_allow_html=True
)

st.divider()

# Layout
col1, col2 = st.columns([2, 1])

with col1:

    job_description = st.text_area(
        "📝 Enter Job Description",
        height=300,
        placeholder="""
Python
SQL
Machine Learning
Data Analysis
Power BI
Excel
Communication Skills
        """
    )

with col2:

    st.info("""
### How It Works

✅ Reads DOCX Resumes

✅ Extracts Resume Text

✅ TF-IDF Vectorization

✅ Cosine Similarity

✅ Ranks Candidates

✅ Shows Top Matching Resumes
""")

st.divider()

# Button
if st.button(
    "🚀 Find Best Candidates",
    use_container_width=True
):

    if job_description.strip() == "":

        st.warning(
            "Please enter a Job Description"
        )

    else:

        results = get_top_resumes(
            job_description,
            top_n=5
        )

        st.success(
            "Resume Matching Completed Successfully!"
        )

        st.subheader(
            "🏆 Top 5 Matching Resumes"
        )

        df = pd.DataFrame(
            results,
            columns=[
                "Resume Name",
                "Match Score (%)"
            ]
        )

        df.index = range(
            1,
            len(df) + 1
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        best_resume = results[0][0]
        best_score = results[0][1]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="🥇 Best Candidate",
                value=best_resume
            )

        with col2:

            st.metric(
                label="🎯 Highest Match Score",
                value=f"{best_score}%"
            )

        st.success(
            f"Recommended Candidate: {best_resume}"
        )