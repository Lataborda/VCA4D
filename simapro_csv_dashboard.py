import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="SimaPro CSV Dashboard + PB-LCA", layout="wide")


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


def normalize_category_text(text):
    """Light normalization to improve category matching across CSV files."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    replacements = {
        "é": "e", "è": "e", "ê": "e",
        "á": "a", "à": "a", "â": "a",
        "í": "i", "ï": "i",
        "ó": "o", "ô": "o",
        "ú": "u", "û": "u",
        "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("&", "and")
    text = " ".join(text.split())
    return text


def build_bar_chart(filtered, category_col, unit_col, comparison_label, mode):
    pivot_df = filtered.pivot_table(
        index="Category_label",
        columns=comparison_label,
        values="Value",
        aggfunc="mean"
    )
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


def build_system_limits_chart(
    df_ratio,
    ratio_col="Ratio",
    label_col="Display_label",
    x_max=3.0,
    safe_limit=1.0,
    warning_limit=2.0,
    safe_label="Espace sûr",
    warning_label="Zone d'attention",
    risk_label="Risque élevé",
    title="Impacts par rapport au Safe Operating Space"
):
    """Build a PB-LCA style system-limits chart with background bands."""
    df_plot = df_ratio.copy().sort_values(ratio_col, ascending=True)
    y_positions = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(12, max(5, len(df_plot) * 0.55)))

    ax.axvspan(0, safe_limit, color="#0a8a3a", alpha=1.0, zorder=0)
    ax.axvspan(safe_limit, warning_limit, color="#f2c500", alpha=1.0, zorder=0)
    ax.axvspan(warning_limit, x_max, color="#ef2b0c", alpha=1.0, zorder=0)

    for x in np.arange(0.5, x_max + 0.001, 0.5):
        ax.axvline(x, color="white", lw=1.5, alpha=0.8, zorder=1)
    ax.axvline(safe_limit, color="white", lw=2.5, zorder=2)
    ax.axvline(warning_limit, color="white", lw=2.5, zorder=2)

    ratios = np.clip(df_plot[ratio_col].fillna(0).values, 0, x_max)
    ax.barh(y_positions, ratios, color="black", edgecolor="black", height=0.35, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot[label_col].tolist(), fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Impact / SoSOS ratio", fontsize=13)
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.text(safe_limit / 2, -0.22, safe_label,
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=18, color="#0a8a3a", fontweight="bold")
    ax.text((safe_limit + warning_limit) / 2, -0.22, warning_label,
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=18, color="#c79a00", fontweight="bold")
    ax.text((warning_limit + x_max) / 2, -0.22, risk_label,
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=18, color="#d92c16", fontweight="bold")

    for idx, actual in enumerate(df_plot[ratio_col].fillna(0).values):
        if actual > x_max:
            ax.text(x_max - 0.03, idx, f">{x_max:.1f}",
                    va="center", ha="right", color="white", fontsize=10, fontweight="bold")

    ax.invert_yaxis()
    fig.subplots_adjust(bottom=0.30)
    return fig


def build_system_limits_chart_multi(
    df_ratio,
    ratio_col="Ratio",
    label_col="Display_label",
    entity_col="Entity",
    order_col="plot_order",
    x_max=3.0,
    safe_limit=1.0,
    warning_limit=2.0,
    safe_label="Espace sûr",
    warning_label="Zone d'attention",
    risk_label="Risque élevé",
    title="Comparaison des produits vs limites du système",
    show_values=True,
):
    """PB-LCA style system-limits chart con diseño profesional."""
    df_plot = df_ratio.copy()

    if order_col in df_plot.columns:
        category_order = (
            df_plot[[label_col, order_col]]
            .drop_duplicates()
            .sort_values(order_col)[label_col]
            .tolist()
        )
    else:
        category_order = df_plot[label_col].drop_duplicates().tolist()

    entity_order = df_plot[entity_col].drop_duplicates().tolist()

    product_palette = {
        "amidon":     "#2E5C8A",
        "bobolo":     "#E07B39",
        "industriel": "#2A9D8F",
        "rurale":     "#8B5A3C",
        "gari":       "#6A4C93",
    }

    def get_entity_color(entity):
        name = str(entity).strip().lower()
        for key, color in product_palette.items():
            if key in name:
                return color
        return "#555555"

    n_cat = len(category_order)
    n_ent = len(entity_order)

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.edgecolor": "#333333", "axes.linewidth": 0.8})

    fig, ax = plt.subplots(figsize=(15, max(7, n_cat * 1.4)))
    fig.patch.set_facecolor("white")

    color_safe    = "#7FB069"
    color_warning = "#F4C95D"
    color_risk    = "#D9534F"

    ax.axvspan(0, safe_limit, color=color_safe, alpha=0.85, zorder=0)
    ax.axvspan(safe_limit, warning_limit, color=color_warning, alpha=0.85, zorder=0)
    ax.axvspan(warning_limit, x_max, color=color_risk, alpha=0.85, zorder=0)

    for x in np.arange(0.5, x_max + 0.001, 0.5):
        is_threshold = np.isclose(x, safe_limit) or np.isclose(x, warning_limit)
        ax.axvline(x, color="white",
                   lw=2.2 if is_threshold else 0.8,
                   alpha=0.95 if is_threshold else 0.45, zorder=1)

    y_base = np.arange(n_cat) * 1.35
    separator_y = (y_base[:-1] + y_base[1:]) / 2
    for y in separator_y:
        ax.hlines(y, 0, x_max, colors="white",
                  linestyles=(0, (1, 4)), linewidth=0.8, alpha=0.5, zorder=2)

    group_height = 0.95
    bar_height   = group_height / max(n_ent, 1)

    for i, entity in enumerate(entity_order):
        offsets = y_base - group_height / 2 + (i + 0.5) * bar_height
        ratios = []
        for cat in category_order:
            subset = df_plot[(df_plot[label_col] == cat) & (df_plot[entity_col] == entity)]
            ratios.append(float(subset[ratio_col].iloc[0]) if len(subset) > 0 else 0.0)

        ratios_clipped = np.clip(ratios, 0, x_max)
        color = get_entity_color(entity)
        ax.barh(offsets, ratios_clipped, height=bar_height * 0.88,
                color=color, edgecolor="none", zorder=3, label=entity)

        if show_values:
            for y, val, original in zip(offsets, ratios_clipped, ratios):
                if original > x_max:
                    ax.text(x_max - 0.05, y, f">{x_max:.1f}",
                            va="center", ha="right", color="white",
                            fontsize=8.5, fontweight="600", zorder=4)
                elif val >= 0.05:
                    x_text = val + 0.03
                    if x_text <= safe_limit:
                        text_color = "#1B3A1B"
                    elif x_text <= warning_limit:
                        text_color = "#5C4400"
                    else:
                        text_color = "#5C0000"
                    ax.text(x_text, y, f"{original:.2f}",
                            va="center", ha="left", color=text_color,
                            fontsize=8.5, fontweight="600", zorder=4)

    ax.set_yticks(y_base)
    ax.set_yticklabels(category_order, fontsize=12, color="#222222")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Impact / SoSOS ratio", fontsize=12, color="#333333", labelpad=10)
    ax.tick_params(axis="x", colors="#555555", labelsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.set_title(title, fontsize=17, fontweight="bold", color="#1a1a1a", pad=42, loc="left")

    def spaced(s):
        return " ".join(list(s.upper()))

    y_top = 1.015
    ax.text(safe_limit / 2, y_top, spaced(safe_label),
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=10, color=color_safe, fontweight="bold")
    ax.text((safe_limit + warning_limit) / 2, y_top, spaced(warning_label),
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=10, color="#C19000", fontweight="bold")
    ax.text((warning_limit + x_max) / 2, y_top, spaced(risk_label),
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=10, color=color_risk, fontweight="bold")

    legend = ax.legend(title="Produit", loc="upper center",
                       bbox_to_anchor=(0.5, -0.13), ncol=min(5, n_ent),
                       frameon=False, fontsize=11, title_fontsize=11,
                       handlelength=1.2, handleheight=1.2, columnspacing=2.0)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#333333")

    ax.invert_yaxis()
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.20)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Chaînes de valeur groupées (stacked bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def read_vc_agrupado(uploaded_file):
    """
    Read a SimaPro 'agrupado' CSV (sep=';', decimal=',').
    Strips BOM, renames the category column to 'Catégorie', converts
    Production / Transport / Transformation / Total to float.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = None
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        raise ValueError("Could not decode file.")

    df = pd.read_csv(io.StringIO(text), sep=";", decimal=",")
    df.columns = [c.strip().lstrip("\ufeff").strip() for c in df.columns]

    # Identify category column (first column or by name pattern)
    cat_candidates = [c for c in df.columns if "cat" in c.lower() or "impact" in c.lower()]
    cat_col = cat_candidates[0] if cat_candidates else df.columns[0]
    if cat_col != "Catégorie":
        df = df.rename(columns={cat_col: "Catégorie"})

    for col in ["Total", "Production", "Transport", "Transformation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_vc_grouped_stacked_bar(
    dfs: dict,
    selected_cats: list,
    normalize: bool = True,
    stage_colors: dict = None,
    title: str = "Impacts environnementaux classés par catégorie et par produit",
):
    """
    Horizontal stacked bar chart (100 % or absolute) showing
    Production / Transport / Transformation for each product × impact category.

    Layout (mirrors the reference image):
      • One group of bars per impact category (y-axis)
      • One bar per product inside each group
      • Segments stacked left→right: Production | Transport | Transformation
    """
    stages = ["Production", "Transport", "Transformation"]

    if stage_colors is None:
        stage_colors = {
            "Production":     "#2166ac",   # blue
            "Transport":      "#f4a621",   # amber/orange
            "Transformation": "#1a9641",   # green
        }

    products = list(dfs.keys())
    n_cat  = len(selected_cats)
    n_prod = len(products)

    bar_h   = 0.68
    group_h = n_prod * bar_h + 0.50
    fig_h   = max(5, n_cat * group_h * 0.52 + 1.8)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("white")

    # Y positions: each category group centred on cat_centres[i]
    cat_centres = np.arange(n_cat) * (group_h + 0.25)
    offsets_per_prod = np.linspace(
        -(n_prod - 1) / 2 * bar_h,
         (n_prod - 1) / 2 * bar_h,
        n_prod
    )

    legend_handles = {}   # stage → patch handle

    for p_idx, product in enumerate(products):
        df = dfs[product]
        df_cat = df.set_index("Catégorie")

        for c_idx, cat in enumerate(selected_cats):
            y_pos = cat_centres[c_idx] + offsets_per_prod[p_idx]

            vals = {}
            for stage in stages:
                if cat in df_cat.index and stage in df_cat.columns:
                    v = df_cat.loc[cat, stage]
                    vals[stage] = float(v) if pd.notna(v) else 0.0
                else:
                    vals[stage] = 0.0

            total = sum(vals.values())
            if total == 0:
                continue

            plot_vals = (
                {s: v / total * 100 for s, v in vals.items()}
                if normalize else vals
            )

            left = 0.0
            for stage in stages:
                w = plot_vals[stage]
                bar = ax.barh(
                    y_pos, w,
                    left=left,
                    height=bar_h * 0.86,
                    color=stage_colors[stage],
                    edgecolor="none",
                    zorder=3,
                )
                if stage not in legend_handles:
                    legend_handles[stage] = bar
                left += w

        # Product label to the left of its first category bar
        if n_cat > 0:
            y_label = cat_centres[0] + offsets_per_prod[p_idx]
            ax.text(
                -1.2, y_label, product,
                va="center", ha="right",
                fontsize=8, color="#333333", fontstyle="italic",
            )

    # ── Y-axis: category names centred on group ────────────────────────────
    ax.set_yticks(cat_centres)
    ax.set_yticklabels(selected_cats, fontsize=10, color="#222222")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.invert_yaxis()

    # ── X-axis ────────────────────────────────────────────────────────────
    if normalize:
        ax.set_xlim(0, 100)
        ax.set_xlabel("Contribution (%)", fontsize=11, labelpad=8)
    else:
        ax.set_xlabel("Valeur absolue", fontsize=11, labelpad=8)

    # ── Horizontal separators between category groups ─────────────────────
    for i in range(1, n_cat):
        sep_y = (cat_centres[i - 1] + cat_centres[i]) / 2
        ax.axhline(sep_y, color="#cccccc", linewidth=0.7, zorder=1)

    # ── Stage legend (bottom) ──────────────────────────────────────────────
    handles = [legend_handles[s] for s in stages if s in legend_handles]
    labels  = [s for s in stages if s in legend_handles]
    ax.legend(
        handles, labels,
        title="Étape", title_fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(stages),
        frameon=False, fontsize=10,
        handlelength=1.1,
    )

    # ── Grid and spines ───────────────────────────────────────────────────
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#888888")

    ax.set_title(title, fontsize=13, fontweight="bold",
                 loc="left", pad=10, color="#1a1a1a")

    fig.subplots_adjust(left=0.25, right=0.97, top=0.94, bottom=0.12)
    return fig


def show_vc_grouped_section():
    """Render section 8 — grouped value chains stacked bar chart."""
    st.divider()
    st.subheader("8. Chaînes de valeur groupées — Stacked bar chart")
    st.markdown(
        "Chargez les fichiers CSV **agrupado** (un par chaîne de valeur / produit). "
        "Colonnes attendues : `Catégorie d'impact`, `Unité`, `Total`, "
        "`Production`, `Transport`, `Transformation`."
    )

    vc_files = st.file_uploader(
        "Charger les fichiers CSV agrupado (un par produit)",
        type=["csv"],
        accept_multiple_files=True,
        key="vc_agrupado_files",
    )

    if not vc_files:
        st.info("Chargez au moins un fichier CSV agrupado pour générer le graphique.")
        return

    dfs = {}
    all_cats = None

    for f in vc_files:
        try:
            df = read_vc_agrupado(f)
            # Short display name derived from filename
            name = (
                f.name
                .replace("VC_", "")
                .replace("_Cameroun_impacts_agrupado.csv", "")
                .replace("_Cameroun_impacts_agrupado", "")
                .replace(".csv", "")
                .replace("_", " ")
                .strip()
            )
            dfs[name] = df
            cats = set(df["Catégorie"].dropna().tolist())
            all_cats = cats if all_cats is None else all_cats.intersection(cats)
        except Exception as e:
            st.error(f"Impossible de lire {f.name} : {e}")

    if not dfs:
        return

    all_cats_sorted = sorted(all_cats)

    # Default selection — prioritize key categories similar to the reference image
    priority_defaults = [
        "Global warming",
        "Freshwater eutrophication",
        "Terrestrial ecotoxicity",
        "Land use",
        "Water consumption",
    ]
    default_sel = [c for c in priority_defaults if c in all_cats_sorted]
    if not default_sel:
        default_sel = all_cats_sorted[:6]

    col_a, col_b = st.columns(2)
    with col_a:
        normalize = st.toggle("Normaliser à 100 % par barre", value=True, key="vc_normalize")
    with col_b:
        chart_title = st.text_input(
            "Titre du graphique",
            value="Impacts environnementaux classés par catégorie et par produit",
            key="vc_chart_title",
        )

    selected_cats = st.multiselect(
        "Sélectionner les catégories d'impact",
        options=all_cats_sorted,
        default=default_sel,
        key="vc_selected_cats",
    )

    if not selected_cats:
        st.warning("Sélectionnez au moins une catégorie d'impact.")
        return

    fig = build_vc_grouped_stacked_bar(
        dfs=dfs,
        selected_cats=selected_cats,
        normalize=normalize,
        title=chart_title,
    )
    st.pyplot(fig)

    png_data = fig_to_png_download(fig)
    st.download_button(
        "⬇ Télécharger le graphique (PNG)",
        data=png_data,
        file_name="vc_grouped_stacked_bar.png",
        mime="image/png",
        key="vc_dl_png",
    )

    with st.expander("Afficher le tableau de données combinées"):
        rows = []
        for product, df in dfs.items():
            cols_keep = ["Catégorie"] + [
                c for c in ["Production", "Transport", "Transformation", "Total"]
                if c in df.columns
            ]
            tmp = df[cols_keep].copy()
            tmp.insert(0, "Produit", product)
            rows.append(tmp)
        combined = pd.concat(rows, ignore_index=True)
        st.dataframe(combined, use_container_width=True)
        st.download_button(
            "⬇ Télécharger tableau combiné (CSV)",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="vc_grouped_combined.csv",
            mime="text/csv",
            key="vc_dl_csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def show_intro():
    st.title("SimaPro CSV Dashboard + PB-LCA System Limits")
    st.markdown(
        """
        Upload one or more CSV files exported from SimaPro and compare products, scenarios,
        sites or any other entities. The app can clean values, normalize impacts by row,
        generate comparative charts, export filtered outputs and build a PB-LCA style
        system-limits chart using SoSOS values.
        """
    )
    st.info(
        "Recommended impact CSV structure: one column for impact category, one for unit, "
        "and the remaining columns as products, scenarios, sites or systems to compare."
    )


def main():
    show_intro()

    uploaded_files = st.file_uploader(
        "Upload one or more impact CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one CSV file exported from SimaPro.")
        # Still show section 8 even without files in sections 1-7
        show_vc_grouped_section()
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
        show_vc_grouped_section()
        return

    common_columns = list(common_columns) if common_columns else []

    if len(common_columns) < 2:
        st.warning(
            "The uploaded files do not share enough common columns. "
            "They should at least share category and unit columns."
        )
        show_vc_grouped_section()
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
        long_original   = dataframe_to_long(df,      category_col, unit_col, source_name)
        long_normalized = dataframe_to_long(df_norm, category_col, unit_col, source_name)
        long_original_list.append(long_original)
        long_normalized_list.append(long_normalized)

    if not wide_clean_data:
        st.warning("No valid files remain after cleaning.")
        show_vc_grouped_section()
        return

    original_all   = pd.concat(long_original_list,   ignore_index=True)
    normalized_all = pd.concat(long_normalized_list, ignore_index=True)
    original_all   = original_all.rename(columns={"Entity": comparison_label})
    normalized_all = normalized_all.rename(columns={"Entity": comparison_label})

    st.success("Files processed successfully.")

    with st.expander("Show cleaned wide tables"):
        for source_name, df in wide_clean_data.items():
            st.markdown(f"**{source_name}**")
            st.dataframe(df, use_container_width=True)

    st.subheader("4. Filters")
    mode = st.radio("Display mode", ["Normalized", "Original"], horizontal=True)
    working_df = normalized_all.copy() if mode == "Normalized" else original_all.copy()

    available_sources    = sorted(working_df["Source"].dropna().unique().tolist())
    available_entities   = sorted(working_df[comparison_label].dropna().unique().tolist())
    available_categories = working_df[category_col].dropna().unique().tolist()

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        selected_sources = st.multiselect("Select files", options=available_sources, default=available_sources)
    with colf2:
        selected_entities = st.multiselect(
            f"Select {comparison_label.lower()}s",
            options=available_entities, default=available_entities
        )
    with colf3:
        default_categories = [
            "Global warming", "Freshwater eutrophication",
            "Terrestrial ecotoxicity", "Land use", "Water consumption"
        ]
        default_categories = [c for c in default_categories if c in available_categories]
        selected_categories = st.multiselect(
            "Select impact categories",
            options=available_categories, default=default_categories
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
    else:
        if chart_type == "Horizontal bar chart":
            fig = build_bar_chart(filtered, category_col, unit_col, comparison_label, mode)
            st.pyplot(fig)
            st.download_button("Download bar chart as PNG",
                               data=fig_to_png_download(fig),
                               file_name="simapro_bar_chart.png", mime="image/png")

        elif chart_type == "Radar chart":
            fig_radar = build_radar_chart(filtered, comparison_label)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.download_button("Download radar chart as HTML",
                               data=fig_radar.to_html().encode("utf-8"),
                               file_name="simapro_radar_chart.html", mime="text/html")

        elif chart_type == "Heatmap":
            fig_heat = build_heatmap(filtered, comparison_label)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.download_button("Download heatmap as HTML",
                               data=fig_heat.to_html().encode("utf-8"),
                               file_name="simapro_heatmap.html", mime="text/html")

    st.subheader("6. Export filtered data")
    st.download_button("Download filtered data as CSV",
                       data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="simapro_filtered_data.csv", mime="text/csv")

    st.divider()
    st.subheader("7. PB-LCA system limits chart")
    st.write(
        "Use a second CSV with SoSOS values to calculate the ratio Impact / SoSOS and build "
        "either a single-product or multi-product PB-LCA chart."
    )

    reference_file = st.file_uploader(
        "Upload SoSOS / reference limits CSV", type=["csv"], key="reference_file"
    )
    st.caption(
        "Recommended reference CSV columns: one category column, one SoSOS column, "
        "and optional display label and order columns."
    )

    if reference_file is not None:
        try:
            ref_raw = read_csv_flexible(reference_file)
            st.markdown("**Reference limits preview**")
            st.dataframe(ref_raw.head(10), use_container_width=True)

            ref_col1, ref_col2, ref_col3, ref_col4 = st.columns(4)
            with ref_col1:
                ref_category_col = st.selectbox("Reference category column", ref_raw.columns, key="ref_category")
            with ref_col2:
                ref_sosos_col = st.selectbox("SoSOS column", ref_raw.columns, key="ref_sosos")
            with ref_col3:
                label_options = ["<use impact category>"] + list(ref_raw.columns)
                ref_label_col = st.selectbox("Display label column (optional)", label_options, key="ref_label")
            with ref_col4:
                order_options = ["<keep current order>"] + list(ref_raw.columns)
                ref_order_col = st.selectbox("Order column (optional)", order_options, key="ref_order")

            ref_df = ref_raw.copy()
            ref_df[ref_sosos_col] = pd.to_numeric(ref_df[ref_sosos_col], errors="coerce")
            ref_df = ref_df.dropna(subset=[ref_sosos_col])
            ref_df["category_key"] = ref_df[ref_category_col].apply(normalize_category_text)

            ref_df["Display_label"] = (
                ref_df[ref_category_col].astype(str)
                if ref_label_col == "<use impact category>"
                else ref_df[ref_label_col].astype(str)
            )
            ref_df["plot_order"] = (
                np.arange(len(ref_df))
                if ref_order_col == "<keep current order>"
                else pd.to_numeric(ref_df[ref_order_col], errors="coerce").fillna(np.arange(len(ref_df)))
            )

            pb_base = original_all.copy()
            pb_base["category_key"] = pb_base[category_col].apply(normalize_category_text)
            pb_sources    = sorted(pb_base["Source"].dropna().unique().tolist())
            pb_all_entities = sorted(pb_base[comparison_label].dropna().unique().tolist())

            st.markdown("### PB-LCA chart settings")
            chart_mode = st.radio("PB-LCA chart mode",
                                  ["Single product", "Multi-product comparison"],
                                  horizontal=True, key="pb_chart_mode")

            cfg_a, cfg_b = st.columns(2)
            with cfg_a:
                pb_source = st.selectbox("Select file for PB-LCA chart", pb_sources, key="pb_source")

            if chart_mode == "Single product":
                with cfg_b:
                    pb_entity = st.selectbox(f"Select {comparison_label.lower()} for PB-LCA chart",
                                             pb_all_entities, key="pb_entity_single")
                pb_filtered = pb_base[
                    (pb_base["Source"] == pb_source) &
                    (pb_base[comparison_label] == pb_entity)
                ].copy()
            else:
                with cfg_b:
                    pb_entities = st.multiselect(f"Select {comparison_label.lower()}s for PB-LCA chart",
                                                 pb_all_entities, default=pb_all_entities[:5],
                                                 key="pb_entities_multi")
                pb_filtered = pb_base[
                    (pb_base["Source"] == pb_source) &
                    (pb_base[comparison_label].isin(pb_entities))
                ].copy()

            pb_merged = pb_filtered.merge(
                ref_df[["category_key", ref_sosos_col, "Display_label", "plot_order"]],
                on="category_key", how="inner"
            )

            if pb_merged.empty:
                st.warning("No matching categories were found between the impact table and the SoSOS table.")
            else:
                pb_merged["Ratio"] = pb_merged["Value"] / pb_merged[ref_sosos_col]
                pb_merged = pb_merged.sort_values(["plot_order", comparison_label])

                st.markdown("**PB-LCA merged table**")
                preview_cols = [category_col, unit_col, comparison_label, "Value",
                                ref_sosos_col, "Ratio", "Display_label"]
                st.dataframe(pb_merged[preview_cols], use_container_width=True)

                max_ratio    = float(np.nanmax(pb_merged["Ratio"].values)) if len(pb_merged) > 0 else 3.0
                default_xmax = max(3.0, min(np.ceil(max_ratio), 10.0))

                cfg1, cfg2, cfg3 = st.columns(3)
                with cfg1:
                    x_max = st.number_input("X-axis maximum", min_value=1.5, max_value=20.0,
                                            value=float(default_xmax), step=0.5)
                with cfg2:
                    warning_limit = st.number_input("Second visual threshold", min_value=1.1,
                                                    max_value=20.0, value=2.0, step=0.1)
                with cfg3:
                    default_title = (
                        f"{pb_entity} vs system limits"
                        if chart_mode == "Single product"
                        else "Comparaison des produits vs limites du système"
                    )
                    chart_title = st.text_input("Chart title", value=default_title)

                safe_label    = st.text_input("Safe zone label",      value="Espace sûr")
                warning_label = st.text_input("Middle zone label",    value="Zone d'attention")
                risk_label    = st.text_input("High-risk zone label", value="Risque élevé")

                if chart_mode == "Single product":
                    pb_fig = build_system_limits_chart(
                        pb_merged, ratio_col="Ratio", label_col="Display_label",
                        x_max=float(x_max), safe_limit=1.0,
                        warning_limit=float(warning_limit),
                        safe_label=safe_label, warning_label=warning_label,
                        risk_label=risk_label, title=chart_title
                    )
                else:
                    pb_fig = build_system_limits_chart_multi(
                        pb_merged.rename(columns={comparison_label: "Entity"}),
                        ratio_col="Ratio", label_col="Display_label", entity_col="Entity",
                        order_col="plot_order", x_max=float(x_max), safe_limit=1.0,
                        warning_limit=float(warning_limit),
                        safe_label=safe_label, warning_label=warning_label,
                        risk_label=risk_label, title=chart_title
                    )

                st.pyplot(pb_fig)
                st.download_button("Download system limits chart as PNG",
                                   data=fig_to_png_download(pb_fig),
                                   file_name="pb_lca_system_limits_chart.png", mime="image/png")
                st.download_button("Download PB-LCA merged table as CSV",
                                   data=pb_merged.to_csv(index=False).encode("utf-8"),
                                   file_name="pb_lca_ratios.csv", mime="text/csv")

                st.info(
                    "Methodological note: the threshold at 1.0 corresponds to the PB-LCA interpretation "
                    "Impact / SoSOS = 1. Values above 1 exceed the assigned safe operating space. "
                    "Additional bands above 1 are communication choices unless they are supported "
                    "by a specific uncertainty analysis."
                )

        except Exception as e:
            st.error(f"Could not process the SoSOS reference CSV: {e}")

    # ── Section 8: grouped value chains stacked bar ────────────────────────
    show_vc_grouped_section()


if __name__ == "__main__":
    main()
