import streamlit as st
import pandas as pd
import plotly.express as px

REQUIRED_COLUMNS = ["Recency", "Tenure", "Frequency", "Monetary", "AvgOrderValue"]


def segmentation_tab(pipeline):
    st.subheader("🎯 Customer Segmentation")
    st.caption(
        "K-Means clustering across RFM features — "
        "Recency, Frequency, Monetary, Tenure, and Average Order Value."
    )

    # ── Check for auto-loaded data from app2.py startup ───────────────────────
    auto_data_ready = (
        st.session_state.get("segmented_df") is not None
        and "Segment_Name" in st.session_state["segmented_df"].columns
    )

    # ── Optional upload override ──────────────────────────────────────────────
    with st.expander("📥 Upload custom RFM dataset (optional — overrides default data)"):
        st.caption(
            f"CSV must contain: {', '.join(REQUIRED_COLUMNS)}"
        )
        uploaded_file = st.file_uploader(
            "Upload Customer RFM CSV",
            type="csv",
            key="seg_upload",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                missing = [c for c in REQUIRED_COLUMNS if c not in df_upload.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    with st.spinner("Running segmentation on custom dataset..."):
                        features  = df_upload[REQUIRED_COLUMNS].copy()
                        processed = pipeline._preprocess(features)
                        clusters  = pipeline.model.predict(processed)

                        df_upload["Segment"]      = clusters
                        seg_map                   = pipeline._name_segments(df_upload)
                        df_upload["Segment_Name"] = df_upload["Segment"].map(seg_map)

                        st.session_state["segmented_df"] = df_upload
                        st.session_state["analysis_df"]  = df_upload
                        st.success("Custom dataset segmented successfully.")
                        auto_data_ready = True

            except Exception as e:
                st.error(f"Error processing file: {e}")

    st.divider()

    # ── Guard: nothing loaded at all ──────────────────────────────────────────
    if not auto_data_ready:
        st.warning("No segmented data available. Upload a CSV above to begin.")
        return

    # ── Use whatever is in session state ─────────────────────────────────────
    result = st.session_state["segmented_df"].copy()

    # ── Summary KPI cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   f"{len(result):,}")
    c2.metric("Total Revenue",     f"£{result['Monetary'].sum():,.0f}")
    c3.metric("Avg Frequency",     f"{result['Frequency'].mean():.1f}")
    c4.metric("Avg Recency",       f"{result['Recency'].mean():.0f} days")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Customer Distribution by Segment")
        seg = result["Segment_Name"].value_counts().reset_index()
        seg.columns = ["Segment", "Customers"]
        fig = px.bar(
            seg, x="Segment", y="Customers",
            color="Segment",
            color_discrete_sequence=["#58A6FF","#3FB950","#E3B341","#F85149"],
            text="Customers",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(color="#E6EDF3"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#30363D"),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Revenue Contribution by Segment")
        revenue = result.groupby("Segment_Name")["Monetary"].sum().reset_index()
        fig2 = px.pie(
            revenue, values="Monetary", names="Segment_Name",
            hole=0.5,
            color_discrete_sequence=["#58A6FF","#3FB950","#E3B341","#F85149"],
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=12)
        fig2.update_layout(
            showlegend=False,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Scatter: RFM landscape ────────────────────────────────────────────────
    st.markdown("#### RFM Landscape — Recency vs Monetary")
    hover_cols = [c for c in ["CustomerID","Frequency","Tenure"] if c in result.columns]
    fig3 = px.scatter(
        result, x="Recency", y="Monetary",
        color="Segment_Name",
        color_discrete_sequence=["#58A6FF","#3FB950","#E3B341","#F85149"],
        hover_data=hover_cols,
        opacity=0.65,
        labels={"Recency":"Recency (days)", "Monetary":"Monetary Value (£)"},
    )
    fig3.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        xaxis=dict(gridcolor="#30363D"),
        yaxis=dict(gridcolor="#30363D"),
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Segment statistics table ──────────────────────────────────────────────
    st.markdown("#### Segment Statistics")
    stats = (
        result.groupby("Segment_Name")
        .agg(
            Customers    =("Segment_Name", "count"),
            AvgRevenue   =("Monetary",     "mean"),
            AvgFrequency =("Frequency",    "mean"),
            AvgRecency   =("Recency",      "mean"),
            AvgTenure    =("Tenure",       "mean"),
        )
        .round(2)
        .reset_index()
    )
    stats["AvgRevenue"] = stats["AvgRevenue"].apply(lambda x: f"£{x:,.0f}")
    st.dataframe(stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Segmentation Results",
        csv,
        "segmentation_results.csv",
        "text/csv",
        use_container_width=True,
    )
