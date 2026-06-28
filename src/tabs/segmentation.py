import streamlit as st
import pandas as pd
import plotly.express as px


REQUIRED_COLUMNS = [
    "Recency",
    "Tenure",
    "Frequency",
    "Monetary",
    "AvgOrderValue"
]


def segmentation_tab(pipeline):

    st.subheader("📊 Customer Segmentation")

    st.write(
        """
        Upload an RFM dataset to classify customers into behavioural segments.
        The segmented dataset will automatically become available to all
        other tabs.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type="csv",
        key="main_upload"
    )

    if uploaded_file is None:

        st.info("Upload a CSV to begin.")
        return

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(e)
        return

    # ----------------------------------------------------
    # Validate columns
    # ----------------------------------------------------

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if len(missing):

        st.error(
            f"Missing required columns: {missing}"
        )

        return

    # ----------------------------------------------------
    # Save original dataset
    # ----------------------------------------------------

    st.session_state["raw_df"] = df.copy()

    st.success("Dataset uploaded successfully.")

    st.markdown("### Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    # ----------------------------------------------------
    # Dataset Summary
    # ----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    c2.metric(
        "Revenue",
        f"£{df['Monetary'].sum():,.0f}"
    )

    c3.metric(
        "Average Frequency",
        round(df["Frequency"].mean(), 2)
    )

    c4.metric(
        "Average Recency",
        round(df["Recency"].mean(), 2)
    )

    st.divider()

    # ----------------------------------------------------
    # Run segmentation
    # ----------------------------------------------------

    if st.button(
        "🚀 Run Customer Segmentation",
        use_container_width=True
    ):

        with st.spinner(
            "Running K-Means clustering..."
        ):

            features = df[
                REQUIRED_COLUMNS
            ].copy()

            processed = pipeline._preprocess(
                features
            )

            clusters = pipeline.model.predict(
                processed
            )

            result = df.copy()

            result["Segment"] = clusters

            segment_map = pipeline._name_segments(
                result
            )

            result["Segment_Name"] = (
                result["Segment"]
                .map(segment_map)
            )

            # -----------------------------------------

            st.session_state["segmented_df"] = result

            st.success(
                "Customer segmentation completed."
            )

    # ----------------------------------------------------
    # Show Results
    # ----------------------------------------------------

    if st.session_state["segmented_df"] is not None:

        result = st.session_state["segmented_df"]

        st.markdown("## Segmentation Results")

        st.dataframe(result)

        st.divider()

        # -----------------------------------------
        # Segment Counts
        # -----------------------------------------

        seg = (
            result["Segment_Name"]
            .value_counts()
            .reset_index()
        )

        seg.columns = [
            "Segment",
            "Customers"
        ]

        fig = px.bar(

            seg,

            x="Segment",

            y="Customers",

            color="Segment",

            text="Customers",

            title="Customer Distribution by Segment"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------------
        # Revenue by Segment
        # -----------------------------------------

        revenue = (

            result

            .groupby(
                "Segment_Name"
            )["Monetary"]

            .sum()

            .reset_index()

        )

        fig2 = px.pie(

            revenue,

            values="Monetary",

            names="Segment_Name",

            hole=.45,

            title="Revenue Contribution by Segment"

        )

        st.plotly_chart(

            fig2,

            use_container_width=True

        )

        # -----------------------------------------
        # Segment Statistics
        # -----------------------------------------

        st.markdown(
            "### Segment Statistics"
        )

        stats = (

            result

            .groupby("Segment_Name")

            .agg(

                Customers=("Segment_Name","count"),

                AvgRevenue=("Monetary","mean"),

                AvgFrequency=("Frequency","mean"),

                AvgRecency=("Recency","mean")

            )

            .round(2)

        )

        st.dataframe(
            stats,
            use_container_width=True
        )

        # -----------------------------------------

        csv = result.to_csv(
            index=False
        ).encode()

        st.download_button(

            "⬇ Download Segmentation Results",

            csv,

            "segmentation_results.csv",

            "text/csv"

        )

        st.success(
            "Segmentation complete. Proceed to the Churn Analysis tab."
        )
