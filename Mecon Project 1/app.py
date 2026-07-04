import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="MECON Recruitment System",
    layout="wide"
)

# Title
st.title("MECON Recruitment Shortlisting System")

# File Upload
uploaded_file = st.file_uploader(
    "Upload Recruitment Dataset",
    type=["xlsx"]
)

if uploaded_file is not None:

    # Read Excel
    df = pd.read_excel(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Required Columns
    required_columns = [
        "CGPA",
        "Aptitude_Score",
        "Technical_Score",
        "Backlogs"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing Columns: {missing_columns}"
        )

    else:

        # Selection Criteria
        st.subheader("Selection Criteria")

        st.info("""
        ✅ CGPA ≥ 6.5

        ✅ Aptitude Score ≥ 60

        ✅ Technical Score ≥ 60

        ✅ Backlogs = 0
        """)

        # Shortlisting Logic
        shortlisted = df[
            (df["CGPA"] >= 6.5) &
            (df["Aptitude_Score"] >= 60) &
            (df["Technical_Score"] >= 60) &
            (df["Backlogs"] == 0)
        ]

        rejected_count = len(df) - len(shortlisted)

        selection_rate = (
            len(shortlisted) / len(df)
        ) * 100

        # Summary Section
        st.subheader("Recruitment Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Candidates",
                len(df)
            )

        with col2:
            st.metric(
                "Shortlisted",
                len(shortlisted)
            )

        with col3:
            st.metric(
                "Rejected",
                rejected_count
            )

        with col4:
            st.metric(
                "Selection Rate",
                f"{selection_rate:.2f}%"
            )

        # Top Candidates
        st.subheader("Top 10 Shortlisted Candidates")

        top_candidates = shortlisted.sort_values(
            by="Technical_Score",
            ascending=False
        )

        st.dataframe(
            top_candidates.head(10)
        )

        # Download Button
        excel_file = "Top_10_Candidates.xlsx"

        top_candidates.head(10).to_excel(
            excel_file,
            index=False
        )

        with open(excel_file, "rb") as file:
            st.download_button(
                label="📥 Download Top 10 Candidates",
                data=file,
                file_name="Top_10_Candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )