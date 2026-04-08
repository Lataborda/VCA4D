import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="SimaPro CSV Dashboard", layout="wide")


def read_csv_flexible(uploaded_file):
    """Read a CSV trying multiple encodings and separator detection."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    text = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = raw.decode(encoding)
            break
        except Exception:
            continue

    if text is None:
        raise ValueError("Could not decode the uploaded file.")

    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    return pd.read_csv(io.StringIO(text), sep=sep)


def clean_numeric_dataframe(df, id_cols):
    """Convert value columns to numeric, supporting decimal comma and scientific notation."""
    df = df.copy()
    value_cols = [c for c in df.columns if c not in id_cols]

    for col in value_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace({"-": None, "nan": None, "None": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_by_row(df, id_cols):
    """Normalize each row to 100 based on its maximum value."""
    df_norm = df.copy()
    value_cols = [c for c in df_norm.columns if c not in id_cols]

    row_max = df_norm[value_cols].max(axis=1)
    row_max = row_max.replace(0, np.nan)

    df_norm[value_cols] = df_norm[value_cols].div(row_max, axis=0) * 100
    return df_norm


def dataframe_to_long(df, category_col, unit_col, source_name):
    """Convert wide SimaPro table to long format."""
    value_cols = [c for c in df.columns if c not in [category_col, unit_col]]
    df_long = df.melt(
        id_vars=[category_col, unit_col],
        value_vars=value_cols,
        var_name="Entity",
        value_name="Value"
    )
    df_long["Source"] = source_name
    return df_long


def fig_to_png_download(fig):
    """Export matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf


def build_bar_chart(filtered, category_col, unit_col, comparison_label, mode):
    pivot_df = filtered.pivot_table(
        index="Category_label",
        columns=[comparison_label, "Source"],
        values="Value",
        aggfunc="mean"
    )

    pivot_df.columns = [f"{col[0]} | {col[1]}" for col in pivot_df.columns]
    pivot_df = pivot_df.sort_index()

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_df) * 0.45)))
    pivot_df.plot(kind="barh", ax=ax, width=0.85)

    ax.set_ylabel("Impact category")
    ax.set_xlabel("Normalized value (%)" if mode == "Normalized" else "Original value")
    ax.set_title("Comparative impact assessment")
    ax.legend(
        title=f"{comparison_label} | File",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2
    )
    return fig


def build_radar_chart(filtered, comparison_label):
    radar_df = filtered.copy()
    radar_df["Legend"] = radar_df[comparison_label].astype(str) + " | " + radar_df["Source"].astype(str)

    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Category_label",
        color="Legend",
        line_close=True
    )
    fig.update_layout(title="Radar comparison", height=750)
    return fig


def build_heatmap(filtered, comparison_label):
    heat_df = filtered.copy()
    heat_df["Legend"] = heat_df[comparison_label].astype(str) + " | " + heat_df["Source"].astype(str)

    heatmap_df = heat_df.pivot_table(
        index="Category_label",
        columns="Legend",
        values="Value",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        labels={"x": "Comparison", "y": "Impact category", "color": "Value"},
        title="Heatmap of impacts"
    )
    return fig


def show_intro():
    st.title("SimaPro CSV Dashboard")
    st.markdown(
        """
        Upload one or more CSV files exported from SimaPro and compare products, scenarios,
        sites or any other entities. The app can clean values, normalize impacts by row,
        generate comparative charts and export filtered outputs.
        """
    )

    st.info(
        "Recommended CSV structure: first one column for impact category, one for unit, "
        "and the remaining columns as products, scenarios, sites or systems to compare."
    )


def main():
    show_intro()

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one CSV file exported from SimaPro.")
        return

    st.subheader("1. Preview uploaded files")

    raw_dataframes = {}
    common_columns = None

    for file in uploaded_files:
        try:
            df_raw = read_csv_flexible(file)
            raw_dataframes[file.name] = df_raw
            st.markdown(f"**File:** {file.name}")
            st.dataframe(df_raw.head(10), use_container_width=True)

            cols = set(df_raw.columns)
            common_columns = cols if common_columns is None else common_columns.intersection(cols)
        except Exception as e:
            st.error(f"Could not read file {file.name}: {e}")

    if not raw_dataframes:
        st.warning("No readable CSV files were uploaded.")
        return

    common_columns = list(common_columns) if common_columns else []
    if len(common_columns) < 2:
        st.warning(
            "The uploaded files do not share enough common columns. "
            "They should at least share category and unit columns."
        )
        return

    st.subheader("2. Define column roles")
    col1, col2, col3 = st.columns(3)

    with col1:
        category_col = st.selectbox("Impact category column", options=common_columns, index=0)

    with col2:
        unit_col = st.selectbox(
            "Unit column",
            options=common_columns,
            index=1 if len(common_columns) > 1 else 0
        )

    with col3:
        comparison_label = st.text_input(
            "Comparison dimension label",
            value="Entity",
            help="Examples: Product, Scenario, Site"
        )

    id_cols = [category_col, unit_col]

    st.subheader("3. Clean and combine files")

    wide_clean_data = {}
    long_original_list = []
    long_normalized_list = []

    for source_name, df_raw in raw_dataframes.items():
        if category_col not in df_raw.columns or unit_col not in df_raw.columns:
            st.warning(f"Skipping {source_name} because required columns are missing.")
            continue

        df = clean_numeric_dataframe(df_raw, id_cols=id_cols)
        value_cols = [c for c in df.columns if c not in id_cols]
        df = df.dropna(subset=value_cols, how="all")

        if df.empty:
            st.warning(f"Skipping {source_name} because it has no numeric comparison columns after cleaning.")
            continue

        wide_clean_data[source_name] = df

        df_norm = normalize_by_row(df, id_cols=id_cols)
        long_original = dataframe_to_long(df, category_col, unit_col, source_name)
        long_normalized = dataframe_to_long(df_norm, category_col, unit_col, source_name)

        long_original_list.append(long_original)
        long_normalized_list.append(long_normalized)

    if not wide_clean_data:
        st.warning("No valid files remain after cleaning.")
        return

    original_all = pd.concat(long_original_list, ignore_index=True)
    normalized_all = pd.concat(long_normalized_list, ignore_index=True)

    original_all = original_all.rename(columns={"Entity": comparison_label})
    normalized_all = normalized_all.rename(columns={"Entity": comparison_label})

    st.success("Files processed successfully.")

    with st.expander("Show cleaned wide tables"):
        for source_name, df in wide_clean_data.items():
            st.markdown(f"**{source_name}**")
            st.dataframe(df, use_container_width=True)

    st.subheader("4. Filters")
    mode = st.radio("Display mode", ["Normalized", "Original"], horizontal=True)
    working_df = normalized_all.copy() if mode == "Normalized" else original_all.copy()

    available_sources = sorted(working_df["Source"].dropna().unique().tolist())
    available_entities = sorted(working_df[comparison_label].dropna().unique().tolist())
    available_categories = working_df[category_col].dropna().unique().tolist()

    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        selected_sources = st.multiselect("Select files", options=available_sources, default=available_sources)

    with colf2:
        selected_entities = st.multiselect(
            f"Select {comparison_label.lower()}s",
            options=available_entities,
            default=available_entities
        )

    with colf3:
        selected_categories = st.multiselect(
            "Select impact categories",
            options=available_categories,
            default=available_categories
        )

    filtered = working_df[
        (working_df["Source"].isin(selected_sources)) &
        (working_df[comparison_label].isin(selected_entities)) &
        (working_df[category_col].isin(selected_categories))
    ].copy()

    filtered["Category_label"] = (
        filtered[category_col].astype(str) + " (" + filtered[unit_col].astype(str) + ")"
    )

    st.subheader("Filtered data")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("5. Charts")
    chart_type = st.radio(
        "Chart type",
        ["Horizontal bar chart", "Radar chart", "Heatmap"],
        horizontal=True
    )

    if filtered.empty:
        st.warning("No data available with the selected filters.")
        return

    if chart_type == "Horizontal bar chart":
        fig = build_bar_chart(filtered, category_col, unit_col, comparison_label, mode)
        st.pyplot(fig)

        png_data = fig_to_png_download(fig)
        st.download_button(
            "Download bar chart as PNG",
            data=png_data,
            file_name="simapro_bar_chart.png",
            mime="image/png"
        )

    elif chart_type == "Radar chart":
        fig_radar = build_radar_chart(filtered, comparison_label)
        st.plotly_chart(fig_radar, use_container_width=True)

        html_bytes = fig_radar.to_html().encode("utf-8")
        st.download_button(
            "Download radar chart as HTML",
            data=html_bytes,
            file_name="simapro_radar_chart.html",
            mime="text/html"
        )

    elif chart_type == "Heatmap":
        fig_heat = build_heatmap(filtered, comparison_label)
        st.plotly_chart(fig_heat, use_container_width=True)

        html_bytes = fig_heat.to_html().encode("utf-8")
        st.download_button(
            "Download heatmap as HTML",
            data=html_bytes,
            file_name="simapro_heatmap.html",
            mime="text/html"
        )

    st.subheader("6. Export filtered data")
    filtered_csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=filtered_csv,
        file_name="simapro_filtered_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
