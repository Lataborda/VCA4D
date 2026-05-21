import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

st.set_page_config(page_title="SimaPro CSV Dashboard + PB-LCA", layout="wide")


# ─── Generic helpers ────────────────────────────────────────────────────────

def read_csv_flexible(uploaded_file):
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
    df_norm = df.copy()
    value_cols = [c for c in df_norm.columns if c not in id_cols]
    row_max = df_norm[value_cols].max(axis=1).replace(0, np.nan)
    df_norm[value_cols] = df_norm[value_cols].div(row_max, axis=0) * 100
    return df_norm


def dataframe_to_long(df, category_col, unit_col, source_name):
    value_cols = [c for c in df.columns if c not in [category_col, unit_col]]
    df_long = df.melt(id_vars=[category_col, unit_col], value_vars=value_cols,
                      var_name="Entity", value_name="Value")
    df_long["Source"] = source_name
    return df_long


def fig_to_png_download(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf


def normalize_category_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    for old, new in {"é":"e","è":"e","ê":"e","á":"a","à":"a","â":"a",
                     "í":"i","ï":"i","ó":"o","ô":"o","ú":"u","û":"u","ç":"c"}.items():
        text = text.replace(old, new)
    return " ".join(text.replace("&","and").split())


# ─── Chart builders (sections 1-7) ──────────────────────────────────────────

def build_bar_chart(filtered, category_col, unit_col, comparison_label, mode):
    pivot_df = filtered.pivot_table(index="Category_label", columns=comparison_label,
                                    values="Value", aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_df) * 0.45)))
    pivot_df.plot(kind="barh", ax=ax, width=0.85)
    ax.set_ylabel("Impact category")
    ax.set_xlabel("Normalized value (%)" if mode == "Normalized" else "Original value")
    ax.set_title("Comparative impact assessment")
    ax.legend(title=f"{comparison_label} | File", loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2)
    return fig


def build_radar_chart(filtered, comparison_label):
    radar_df = filtered.copy()
    radar_df["Legend"] = radar_df[comparison_label].astype(str) + " | " + radar_df["Source"].astype(str)
    fig = px.line_polar(radar_df, r="Value", theta="Category_label",
                        color="Legend", line_close=True)
    fig.update_layout(title="Radar comparison", height=750)
    return fig


def build_heatmap(filtered, comparison_label):
    heat_df = filtered.copy()
    heat_df["Legend"] = heat_df[comparison_label].astype(str) + " | " + heat_df["Source"].astype(str)
    heatmap_df = heat_df.pivot_table(index="Category_label", columns="Legend",
                                     values="Value", aggfunc="mean")
    fig = px.imshow(heatmap_df, aspect="auto",
                    labels={"x":"Comparison","y":"Impact category","color":"Value"},
                    title="Heatmap of impacts")
    return fig


def build_system_limits_chart(df_ratio, ratio_col="Ratio", label_col="Display_label",
                               x_max=3.0, safe_limit=1.0, warning_limit=2.0,
                               safe_label="Espace sûr", warning_label="Zone d'attention",
                               risk_label="Risque élevé",
                               title="Impacts par rapport au Safe Operating Space"):
    df_plot = df_ratio.copy().sort_values(ratio_col, ascending=True)
    y_positions = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(12, max(5, len(df_plot) * 0.55)))
    ax.axvspan(0, safe_limit, color="#0a8a3a", alpha=1.0, zorder=0)
    ax.axvspan(safe_limit, warning_limit, color="#f2c500", alpha=1.0, zorder=0)
    ax.axvspan(warning_limit, x_max, color="#ef2b0c", alpha=1.0, zorder=0)
    for x in np.arange(0.5, x_max + 0.001, 0.5):
        ax.axvline(x, color="white", lw=1.5, alpha=0.8, zorder=1)
    ax.axvline(safe_limit,   color="white", lw=2.5, zorder=2)
    ax.axvline(warning_limit, color="white", lw=2.5, zorder=2)
    ratios = np.clip(df_plot[ratio_col].fillna(0).values, 0, x_max)
    ax.barh(y_positions, ratios, color="black", edgecolor="black", height=0.35, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot[label_col].tolist(), fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Impact / SoSOS ratio", fontsize=13)
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.set_title(title, fontsize=16, fontweight="bold")
    for zone_x, label, color in [
        (safe_limit/2, safe_label, "#0a8a3a"),
        ((safe_limit+warning_limit)/2, warning_label, "#c79a00"),
        ((warning_limit+x_max)/2, risk_label, "#d92c16"),
    ]:
        ax.text(zone_x, -0.22, label, transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=18, color=color, fontweight="bold")
    for idx, actual in enumerate(df_plot[ratio_col].fillna(0).values):
        if actual > x_max:
            ax.text(x_max-0.03, idx, f">{x_max:.1f}", va="center", ha="right",
                    color="white", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    fig.subplots_adjust(bottom=0.30)
    return fig


def build_system_limits_chart_multi(df_ratio, ratio_col="Ratio", label_col="Display_label",
                                     entity_col="Entity", order_col="plot_order", x_max=3.0,
                                     safe_limit=1.0, warning_limit=2.0,
                                     safe_label="Espace sûr", warning_label="Zone d'attention",
                                     risk_label="Risque élevé",
                                     title="Comparaison des produits vs limites du système",
                                     show_values=True):
    df_plot = df_ratio.copy()
    category_order = (df_plot[[label_col, order_col]].drop_duplicates().sort_values(order_col)[label_col].tolist()
                      if order_col in df_plot.columns
                      else df_plot[label_col].drop_duplicates().tolist())
    entity_order = df_plot[entity_col].drop_duplicates().tolist()
    product_palette = {"amidon":"#2E5C8A","bobolo":"#E07B39","industriel":"#2A9D8F",
                       "rurale":"#8B5A3C","gari":"#6A4C93"}
    def get_color(e):
        n = str(e).strip().lower()
        for k, c in product_palette.items():
            if k in n: return c
        return "#555555"
    n_cat, n_ent = len(category_order), len(entity_order)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#333333","axes.linewidth":0.8})
    fig, ax = plt.subplots(figsize=(15, max(7, n_cat * 1.4)))
    fig.patch.set_facecolor("white")
    for zone, c, a in [(0, safe_limit,"#7FB069"),(safe_limit, warning_limit,"#F4C95D"),
                        (warning_limit, x_max,"#D9534F")]:
        ax.axvspan(zone, c if isinstance(c,float) else c, color=a if isinstance(a,str) else c, alpha=0.85, zorder=0)
    ax.axvspan(0, safe_limit, color="#7FB069", alpha=0.85, zorder=0)
    ax.axvspan(safe_limit, warning_limit, color="#F4C95D", alpha=0.85, zorder=0)
    ax.axvspan(warning_limit, x_max, color="#D9534F", alpha=0.85, zorder=0)
    for x in np.arange(0.5, x_max+0.001, 0.5):
        is_thr = np.isclose(x, safe_limit) or np.isclose(x, warning_limit)
        ax.axvline(x, color="white", lw=2.2 if is_thr else 0.8,
                   alpha=0.95 if is_thr else 0.45, zorder=1)
    y_base = np.arange(n_cat) * 1.35
    group_height, bar_height = 0.95, 0.95 / max(n_ent, 1)
    for i, entity in enumerate(entity_order):
        offsets = y_base - group_height/2 + (i+0.5)*bar_height
        ratios = [float(df_plot[(df_plot[label_col]==cat)&(df_plot[entity_col]==entity)][ratio_col].iloc[0])
                  if len(df_plot[(df_plot[label_col]==cat)&(df_plot[entity_col]==entity)]) > 0 else 0.0
                  for cat in category_order]
        ratios_clipped = np.clip(ratios, 0, x_max)
        ax.barh(offsets, ratios_clipped, height=bar_height*0.88, color=get_color(entity),
                edgecolor="none", zorder=3, label=entity)
        if show_values:
            for y, val, orig in zip(offsets, ratios_clipped, ratios):
                if orig > x_max:
                    ax.text(x_max-0.05, y, f">{x_max:.1f}", va="center", ha="right",
                            color="white", fontsize=8.5, fontweight="600", zorder=4)
                elif val >= 0.05:
                    xt = val+0.03
                    tc = "#1B3A1B" if xt<=safe_limit else "#5C4400" if xt<=warning_limit else "#5C0000"
                    ax.text(xt, y, f"{orig:.2f}", va="center", ha="left",
                            color=tc, fontsize=8.5, fontweight="600", zorder=4)
    ax.set_yticks(y_base)
    ax.set_yticklabels(category_order, fontsize=12, color="#222222")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Impact / SoSOS ratio", fontsize=12, color="#333333", labelpad=10)
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=17, fontweight="bold", color="#1a1a1a", pad=42, loc="left")
    spaced = lambda s: " ".join(list(s.upper()))
    for zone_x, lbl, col in [
        (safe_limit/2, safe_label, "#7FB069"),
        ((safe_limit+warning_limit)/2, warning_label, "#C19000"),
        ((warning_limit+x_max)/2, risk_label, "#D9534F"),
    ]:
        ax.text(zone_x, 1.015, spaced(lbl), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=col, fontweight="bold")
    legend = ax.legend(title="Produit", loc="upper center", bbox_to_anchor=(0.5,-0.13),
                       ncol=min(5,n_ent), frameon=False, fontsize=11, title_fontsize=11)
    legend.get_title().set_fontweight("bold")
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.20)
    return fig


# ─── Section 8: Chaînes de valeur groupées ──────────────────────────────────

def read_vc_combined(uploaded_file):
    """
    Read the combined CSV with all value chains.
    Expected columns: Category of impact, Unit, Value Chain, Total,
                      Production, Transport, Transformation.
    Handles BOM, semicolon or comma separators, decimal comma.
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

    # Detect separator
    first = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    dec = "," if sep == ";" else "."

    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=dec)
    df.columns = [c.strip().lstrip("\ufeff").strip() for c in df.columns]
    # Drop unnamed index column if present
    df = df[[c for c in df.columns if c and not c.startswith("Unnamed")]]

    # Ensure numeric stage columns
    for col in ["Total", "Production", "Transport", "Transformation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_vc_stacked_chart(
    df,
    selected_cats,
    chains_order,
    mode="normalized",       # "normalized" | "relative" | "absolute"
    stages=None,
    stage_colors=None,
    title="Impacts environnementaux classés par catégorie et par produit",
):
    """
    Horizontal stacked bar chart with two display modes:

    'normalized'  → each bar = 100%, shows stage composition.
                    Label on the right: chain name + total value.
    'relative'    → bars scaled to the max total within each category,
                    so you can compare magnitude between chains AND see composition.
                    Label on the right: chain name + total value.
    'absolute'    → actual values (different units per category, use carefully).
    """
    if stages is None:
        stages = ["Production", "Transport", "Transformation"]
    if stage_colors is None:
        stage_colors = {
            "Production":     "#2166ac",
            "Transport":      "#f4a621",
            "Transformation": "#1a9641",
        }

    n_cat = len(selected_cats)
    n_ch  = len(chains_order)
    bar_h = 0.55
    gap   = 0.35

    # ── Compute y positions ────────────────────────────────────────────────
    y_pos      = []   # (cat, chain, y_centre)
    cat_label_y = []
    y = 0.0
    for cat in selected_cats:
        centre = y + (n_ch - 1) * (bar_h + 0.06) / 2
        cat_label_y.append(centre)
        for chain in chains_order:
            y_pos.append((cat, chain, y))
            y += bar_h + 0.06
        y += gap

    fig_h = max(5, y * 0.60 + 1.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Per-category maximum (for 'relative' mode) ─────────────────────────
    cat_max = {}
    for cat in selected_cats:
        sub = df[df["Category of impact"] == cat]["Total"]
        cat_max[cat] = float(sub.max()) if not sub.empty else 1.0

    x_label_positions = []   # collect right-end x for each bar (for label)

    for cat, chain, y0 in y_pos:
        row = df[(df["Category of impact"] == cat) & (df["Value Chain"] == chain)]
        if row.empty:
            x_label_positions.append((y0, 0, "", chain, cat))
            continue

        total = float(row["Total"].values[0]) if "Total" in row.columns else np.nan
        unit  = str(row["Unit"].values[0]) if "Unit" in row.columns else ""
        vals  = {}
        for s in stages:
            vals[s] = float(row[s].values[0]) if (s in row.columns and pd.notna(row[s].values[0])) else 0.0

        raw_total = sum(vals.values())
        if raw_total == 0:
            x_label_positions.append((y0, 0, unit, chain, cat))
            continue

        # Determine scale factor
        if mode == "normalized":
            scale = 100.0 / raw_total
        elif mode == "relative":
            scale = 100.0 / cat_max[cat] if cat_max[cat] > 0 else 0
        else:  # absolute
            scale = 1.0

        left = 0.0
        for stage in stages:
            w = vals[stage] * scale
            ax.barh(y0, w, left=left, height=bar_h * 0.88,
                    color=stage_colors[stage], edgecolor="none", zorder=3)
            left += w

        x_label_positions.append((y0, left, unit, chain, cat))

    # ── Right-side labels ─────────────────────────────────────────────────
    # Format: "Chain name  |  total value + unit"
    max_x = max((x for _, x, *_ in x_label_positions), default=100)
    offset = max_x * 0.015 + 0.5

    for y0, bar_end, unit, chain, cat in x_label_positions:
        row = df[(df["Category of impact"] == cat) & (df["Value Chain"] == chain)]
        total = float(row["Total"].values[0]) if not row.empty and "Total" in row.columns else None

        short = (chain.replace("VC ", "").replace("Farine P. ", "F.P. ").strip())
        if total is not None:
            lbl = f"{short}   {total:.3g} {unit}"
        else:
            lbl = short

        ax.text(bar_end + offset, y0, lbl,
                va="center", ha="left", fontsize=8, color="#333333")

    # ── Category labels (left, bold, centred on group) ─────────────────────
    for label, cy in zip(selected_cats, cat_label_y):
        ax.text(-max_x * 0.01 - 0.5, cy, label,
                va="center", ha="right", fontsize=10, color="#222222", fontweight="bold")

    # ── Horizontal separators ─────────────────────────────────────────────
    y_cur = 0.0
    for c_idx in range(n_cat - 1):
        y_cur += n_ch * (bar_h + 0.06)
        sep = y_cur + gap / 2
        ax.axhline(sep, color="#cccccc", linewidth=0.7, zorder=1)
        y_cur += gap

    # ── Axes ─────────────────────────────────────────────────────────────
    x_right = max_x * 1.02
    ax.set_xlim(0, x_right)
    ax.set_ylim(-bar_h, y_pos[-1][2] + bar_h * 1.5)
    ax.invert_yaxis()
    ax.set_yticks([])

    xlabel_map = {
        "normalized": "Contribution (%) — composition par étape",
        "relative":   "Impact relatif (% du max. de la catégorie) — composition + magnitude",
        "absolute":   "Valeur absolue (unités propres à chaque catégorie)",
    }
    ax.set_xlabel(xlabel_map.get(mode, ""), fontsize=11, labelpad=8)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888888")

    # ── Legend ────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=stage_colors[s], label=s) for s in stages]
    ax.legend(handles=patches, title="Étape", ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.35, -0.08), fontsize=10,
              title_fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10, color="#1a1a1a")

    # Right margin large enough for labels
    fig.subplots_adjust(left=0.17, right=0.62, top=0.95, bottom=0.10)
    return fig


def show_vc_grouped_section():
    st.divider()
    st.subheader("8. Chaînes de valeur groupées — Stacked bar chart")
    st.markdown(
        "Chargez le fichier CSV combiné contenant toutes les chaînes de valeur. "
        "Colonnes attendues : **Category of impact**, **Unit**, **Value Chain**, "
        "**Total**, **Production**, **Transport**, **Transformation**."
    )

    vc_file = st.file_uploader(
        "Charger le fichier CSV combiné (toutes chaînes)",
        type=["csv"],
        key="vc_combined_file",
    )

    if vc_file is None:
        st.info("Chargez le fichier CSV combiné pour générer le graphique.")
        return

    try:
        df = read_vc_combined(vc_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    # Detect column names
    cat_col   = next((c for c in df.columns if "category" in c.lower() or "catégorie" in c.lower()), None)
    chain_col = next((c for c in df.columns if "chain" in c.lower() or "chaine" in c.lower()), None)

    if cat_col is None or chain_col is None:
        st.error(f"Colonnes 'Category of impact' ou 'Value Chain' non trouvées. Colonnes : {df.columns.tolist()}")
        return

    all_cats   = sorted(df[cat_col].dropna().unique().tolist())
    all_chains = df[chain_col].dropna().unique().tolist()

    # ── Controls ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        mode = st.radio(
            "Mode d'affichage",
            options=["normalized", "relative", "absolute"],
            format_func=lambda m: {
                "normalized": "✦ Normalisé (100%) — composition",
                "relative":   "✦ Relatif (% du max) — magnitude + composition",
                "absolute":   "✦ Valeur absolue",
            }[m],
            key="vc_mode",
        )
    with col_b:
        st.markdown(
            """
            **Guide des modes :**
            - **Normalisé** : chaque barre = 100 %. Montre la *composition* par étape. Toutes les chaînes semblent "égales" en longueur.
            - **Relatif** : les barres sont mises à l'échelle par rapport à la chaîne avec le plus grand impact de chaque catégorie.
              Permet de voir **à la fois la composition et la magnitude** entre chaînes.
            - **Absolu** : longueur = valeur brute (unités différentes selon la catégorie — interpréter avec prudence).
            """
        )

    chart_title = st.text_input(
        "Titre du graphique",
        value="Impacts environnementaux classés par catégorie et par produit — Cameroun",
        key="vc_chart_title",
    )

    priority_defaults = [
        "Global warming", "Freshwater eutrophication",
        "Terrestrial ecotoxicity", "Land use", "Water consumption",
    ]
    default_sel = [c for c in priority_defaults if c in all_cats] or all_cats[:6]

    selected_cats = st.multiselect(
        "Catégories d'impact à afficher",
        options=all_cats,
        default=default_sel,
        key="vc_selected_cats",
    )

    selected_chains = st.multiselect(
        "Chaînes de valeur à afficher",
        options=all_chains,
        default=all_chains,
        key="vc_selected_chains",
    )

    if not selected_cats or not selected_chains:
        st.warning("Sélectionnez au moins une catégorie et une chaîne.")
        return

    df_filtered = df[
        df[cat_col].isin(selected_cats) &
        df[chain_col].isin(selected_chains)
    ].copy()

    # Rename to standard names if needed
    rename_map = {}
    if cat_col != "Category of impact":
        rename_map[cat_col] = "Category of impact"
    if chain_col != "Value Chain":
        rename_map[chain_col] = "Value Chain"
    if rename_map:
        df_filtered = df_filtered.rename(columns=rename_map)

    fig = build_vc_stacked_chart(
        df=df_filtered,
        selected_cats=selected_cats,
        chains_order=selected_chains,
        mode=mode,
        title=chart_title,
    )

    st.pyplot(fig)

    png_data = fig_to_png_download(fig)
    st.download_button(
        "⬇ Télécharger le graphique (PNG, 300 dpi)",
        data=png_data,
        file_name=f"vc_grouped_{mode}.png",
        mime="image/png",
        key="vc_dl_png",
    )

    with st.expander("Afficher les données filtrées"):
        st.dataframe(df_filtered, use_container_width=True)
        st.download_button(
            "⬇ Télécharger les données (CSV)",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="vc_grouped_data.csv",
            mime="text/csv",
            key="vc_dl_csv",
        )


# ─── Main app ────────────────────────────────────────────────────────────────

def show_intro():
    st.title("SimaPro CSV Dashboard + PB-LCA System Limits")
    st.markdown(
        "Upload one or more CSV files exported from SimaPro to compare products, "
        "scenarios or sites. The app cleans, normalizes, visualizes, and exports. "
        "Section 8 provides a dedicated stacked bar chart for grouped value-chain CSVs."
    )
    st.info(
        "Standard CSV structure: impact category column, unit column, "
        "then one column per product / scenario / site."
    )


def main():
    show_intro()

    uploaded_files = st.file_uploader(
        "Upload one or more impact CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="main_impact_files",
    )

    if not uploaded_files:
        st.info("Please upload at least one CSV file exported from SimaPro.")
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
        st.warning("The uploaded files do not share enough common columns.")
        show_vc_grouped_section()
        return

    st.subheader("2. Define column roles")
    col1, col2, col3 = st.columns(3)
    with col1:
        category_col = st.selectbox("Impact category column", options=common_columns, index=0, key="main_cat_col")
    with col2:
        unit_col = st.selectbox("Unit column", options=common_columns,
                                index=1 if len(common_columns) > 1 else 0, key="main_unit_col")
    with col3:
        comparison_label = st.text_input("Comparison dimension label", value="Entity",
                                         help="Examples: Product, Scenario, Site",
                                         key="main_comparison_label")

    id_cols = [category_col, unit_col]

    st.subheader("3. Clean and combine files")
    wide_clean_data, long_original_list, long_normalized_list = {}, [], []

    for source_name, df_raw in raw_dataframes.items():
        if category_col not in df_raw.columns or unit_col not in df_raw.columns:
            st.warning(f"Skipping {source_name} — required columns missing.")
            continue
        df = clean_numeric_dataframe(df_raw, id_cols=id_cols)
        value_cols = [c for c in df.columns if c not in id_cols]
        df = df.dropna(subset=value_cols, how="all")
        if df.empty:
            st.warning(f"Skipping {source_name} — no numeric data after cleaning.")
            continue
        wide_clean_data[source_name] = df
        df_norm = normalize_by_row(df, id_cols=id_cols)
        long_original_list.append(dataframe_to_long(df, category_col, unit_col, source_name))
        long_normalized_list.append(dataframe_to_long(df_norm, category_col, unit_col, source_name))

    if not wide_clean_data:
        st.warning("No valid files remain after cleaning.")
        show_vc_grouped_section()
        return

    original_all   = pd.concat(long_original_list,   ignore_index=True).rename(columns={"Entity": comparison_label})
    normalized_all = pd.concat(long_normalized_list, ignore_index=True).rename(columns={"Entity": comparison_label})
    st.success("Files processed successfully.")

    with st.expander("Show cleaned wide tables"):
        for sn, df in wide_clean_data.items():
            st.markdown(f"**{sn}**")
            st.dataframe(df, use_container_width=True)

    st.subheader("4. Filters")
    mode = st.radio("Display mode", ["Normalized", "Original"], horizontal=True, key="main_display_mode")
    working_df = normalized_all.copy() if mode == "Normalized" else original_all.copy()

    available_sources    = sorted(working_df["Source"].dropna().unique())
    available_entities   = sorted(working_df[comparison_label].dropna().unique())
    available_categories = working_df[category_col].dropna().unique().tolist()

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        selected_sources   = st.multiselect("Select files",   available_sources,   default=available_sources, key="main_sel_sources")
    with colf2:
        selected_entities  = st.multiselect(f"Select {comparison_label.lower()}s",
                                            available_entities, default=available_entities, key="main_sel_entities")
    with colf3:
        default_categories = [c for c in ["Global warming","Freshwater eutrophication",
                                           "Terrestrial ecotoxicity","Land use","Water consumption"]
                               if c in available_categories]
        selected_categories = st.multiselect("Select impact categories",
                                             available_categories, default=default_categories, key="main_sel_categories")

    filtered = working_df[
        working_df["Source"].isin(selected_sources) &
        working_df[comparison_label].isin(selected_entities) &
        working_df[category_col].isin(selected_categories)
    ].copy()
    filtered["Category_label"] = (
        filtered[category_col].astype(str) + " (" + filtered[unit_col].astype(str) + ")"
    )

    st.subheader("Filtered data")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("5. Charts")
    chart_type = st.radio("Chart type", ["Horizontal bar chart", "Radar chart", "Heatmap"], horizontal=True, key="main_chart_type")

    if not filtered.empty:
        if chart_type == "Horizontal bar chart":
            fig = build_bar_chart(filtered, category_col, unit_col, comparison_label, mode)
            st.pyplot(fig)
            st.download_button("Download bar chart as PNG", data=fig_to_png_download(fig),
                               file_name="simapro_bar_chart.png", mime="image/png")
        elif chart_type == "Radar chart":
            fig_r = build_radar_chart(filtered, comparison_label)
            st.plotly_chart(fig_r, use_container_width=True)
            st.download_button("Download radar chart as HTML", data=fig_r.to_html().encode("utf-8"),
                               file_name="simapro_radar_chart.html", mime="text/html")
        elif chart_type == "Heatmap":
            fig_h = build_heatmap(filtered, comparison_label)
            st.plotly_chart(fig_h, use_container_width=True)
            st.download_button("Download heatmap as HTML", data=fig_h.to_html().encode("utf-8"),
                               file_name="simapro_heatmap.html", mime="text/html")
    else:
        st.warning("No data available with the selected filters.")

    st.subheader("6. Export filtered data")
    st.download_button("Download filtered data as CSV",
                       data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="simapro_filtered_data.csv", mime="text/csv")

    st.divider()
    st.subheader("7. PB-LCA system limits chart")
    reference_file = st.file_uploader("Upload SoSOS / reference limits CSV",
                                      type=["csv"], key="reference_file")
    st.caption("Reference CSV columns: category, SoSOS value, optional display label and order.")

    if reference_file is not None:
        try:
            ref_raw = read_csv_flexible(reference_file)
            st.dataframe(ref_raw.head(10), use_container_width=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1: ref_category_col = st.selectbox("Reference category column", ref_raw.columns, key="ref_category")
            with r2: ref_sosos_col    = st.selectbox("SoSOS column",              ref_raw.columns, key="ref_sosos")
            with r3:
                lo = ["<use impact category>"] + list(ref_raw.columns)
                ref_label_col = st.selectbox("Display label column (optional)", lo, key="ref_label")
            with r4:
                oo = ["<keep current order>"] + list(ref_raw.columns)
                ref_order_col = st.selectbox("Order column (optional)", oo, key="ref_order")

            ref_df = ref_raw.copy()
            ref_df[ref_sosos_col] = pd.to_numeric(ref_df[ref_sosos_col], errors="coerce")
            ref_df = ref_df.dropna(subset=[ref_sosos_col])
            ref_df["category_key"] = ref_df[ref_category_col].apply(normalize_category_text)
            ref_df["Display_label"] = (ref_df[ref_category_col].astype(str)
                                       if ref_label_col == "<use impact category>"
                                       else ref_df[ref_label_col].astype(str))
            ref_df["plot_order"] = (np.arange(len(ref_df))
                                    if ref_order_col == "<keep current order>"
                                    else pd.to_numeric(ref_df[ref_order_col], errors="coerce")
                                        .fillna(np.arange(len(ref_df))))

            pb_base = original_all.copy()
            pb_base["category_key"] = pb_base[category_col].apply(normalize_category_text)
            pb_sources = sorted(pb_base["Source"].dropna().unique())
            pb_all_entities = sorted(pb_base[comparison_label].dropna().unique())

            st.markdown("### PB-LCA chart settings")
            chart_mode = st.radio("PB-LCA chart mode",
                                  ["Single product", "Multi-product comparison"],
                                  horizontal=True, key="pb_chart_mode")
            ca, cb = st.columns(2)
            with ca:
                pb_source = st.selectbox("Select file for PB-LCA", pb_sources, key="pb_source")

            if chart_mode == "Single product":
                with cb:
                    pb_entity = st.selectbox(f"Select {comparison_label.lower()}",
                                             pb_all_entities, key="pb_entity_single")
                pb_filtered = pb_base[(pb_base["Source"]==pb_source) &
                                       (pb_base[comparison_label]==pb_entity)].copy()
            else:
                with cb:
                    pb_entities = st.multiselect(f"Select {comparison_label.lower()}s",
                                                 pb_all_entities, default=pb_all_entities[:5],
                                                 key="pb_entities_multi")
                pb_filtered = pb_base[(pb_base["Source"]==pb_source) &
                                       (pb_base[comparison_label].isin(pb_entities))].copy()

            pb_merged = pb_filtered.merge(
                ref_df[["category_key", ref_sosos_col, "Display_label", "plot_order"]],
                on="category_key", how="inner")

            if pb_merged.empty:
                st.warning("No matching categories between impact table and SoSOS table.")
            else:
                pb_merged["Ratio"] = pb_merged["Value"] / pb_merged[ref_sosos_col]
                pb_merged = pb_merged.sort_values(["plot_order", comparison_label])
                st.dataframe(pb_merged[[category_col, unit_col, comparison_label,
                                        "Value", ref_sosos_col, "Ratio", "Display_label"]],
                             use_container_width=True)

                max_ratio = float(np.nanmax(pb_merged["Ratio"].values)) if len(pb_merged) > 0 else 3.0
                default_xmax = max(3.0, min(np.ceil(max_ratio), 10.0))
                c1, c2, c3 = st.columns(3)
                with c1: x_max = st.number_input("X-axis maximum", 1.5, 20.0, float(default_xmax), 0.5, key="pb_x_max")
                with c2: warning_limit = st.number_input("Second visual threshold", 1.1, 20.0, 2.0, 0.1, key="pb_warn_limit")
                with c3:
                    dt = (f"{pb_entity} vs system limits" if chart_mode == "Single product"
                          else "Comparaison des produits vs limites du système")
                    chart_title = st.text_input("Chart title", value=dt, key="pb_chart_title_input")

                safe_label    = st.text_input("Safe zone label",      value="Espace sûr",    key="pb_safe_label")
                warning_label = st.text_input("Middle zone label",    value="Zone d'attention", key="pb_warn_label")
                risk_label    = st.text_input("High-risk zone label", value="Risque élevé",  key="pb_risk_label")

                pb_fig = (
                    build_system_limits_chart(pb_merged, ratio_col="Ratio", label_col="Display_label",
                                              x_max=float(x_max), safe_limit=1.0,
                                              warning_limit=float(warning_limit),
                                              safe_label=safe_label, warning_label=warning_label,
                                              risk_label=risk_label, title=chart_title)
                    if chart_mode == "Single product" else
                    build_system_limits_chart_multi(
                        pb_merged.rename(columns={comparison_label:"Entity"}),
                        ratio_col="Ratio", label_col="Display_label", entity_col="Entity",
                        order_col="plot_order", x_max=float(x_max), safe_limit=1.0,
                        warning_limit=float(warning_limit),
                        safe_label=safe_label, warning_label=warning_label,
                        risk_label=risk_label, title=chart_title)
                )
                st.pyplot(pb_fig)
                st.download_button("Download system limits chart as PNG",
                                   data=fig_to_png_download(pb_fig),
                                   file_name="pb_lca_system_limits_chart.png", mime="image/png")
                st.download_button("Download PB-LCA merged table as CSV",
                                   data=pb_merged.to_csv(index=False).encode("utf-8"),
                                   file_name="pb_lca_ratios.csv", mime="text/csv")
                st.info(
                    "Methodological note: threshold at 1.0 → Impact / SoSOS = 1. "
                    "Values above 1 exceed the assigned safe operating space."
                )
        except Exception as e:
            st.error(f"Could not process the SoSOS reference CSV: {e}")

    show_vc_grouped_section()


    show_gw_comparison_section()

if __name__ == "__main__":
    main()


# ─── Section 9: Global Warming comparison vs reference chains ────────────────

# Default reference values (from image — editable in the UI)
DEFAULT_REFERENCES = {
    "R.D. Congo": {"Agriculture": 107.7, "Transport": 59.5,  "Transformation": 305.2},
    "Thailand":   {"Agriculture": 571.2, "Transport": 53.8,  "Transformation": 341.7},
}

GW_PHASES  = ["Agriculture", "Transport", "Transformation"]
GW_COLORS  = {
    "Agriculture":    "#4C72B0",
    "Transport":      "#DD8452",
    "Transformation": "#55A868",
}


def build_gw_comparison_chart(
    cameroon_data: dict,   # {chain_label: {phase: value}}
    references: dict,      # {country_label: {phase: value}}
    title: str = "Émissions de CO₂eq par phase du cycle de vie",
    x_label: str = "Émissions de CO2eq (kg / tonne de produit)",
):
    """
    Horizontal stacked bar chart comparing Global Warming (CO2eq) across
    reference countries and Cameroon value chains.

    Parameters
    ----------
    cameroon_data : {label: {Agriculture, Transport, Transformation}}
    references    : {label: {Agriculture, Transport, Transformation}}
    """
    phases  = GW_PHASES
    colors  = GW_COLORS

    # Order: references on top, Cameroon below, separated by dashed line
    rows   = list(references.items()) + list(cameroon_data.items())
    labels = [r[0] for r in rows]
    n_ref  = len(references)
    n      = len(rows)

    bar_h  = 0.62
    gap    = 0.28
    y_pos  = np.arange(n) * (bar_h + gap)

    fig_h  = max(4.5, n * (bar_h + gap) * 0.80 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Subtle background for reference rows
    for i in range(n_ref):
        ax.axhspan(y_pos[i] - bar_h * 0.65, y_pos[i] + bar_h * 0.65,
                   color="#eef2f7", zorder=0)

    # Draw bars
    all_totals = []
    for i, (label, data) in enumerate(rows):
        left = 0.0
        for phase in phases:
            val = float(data.get(phase, 0.0))
            ax.barh(y_pos[i], val, left=left, height=bar_h,
                    color=colors[phase], edgecolor="white", linewidth=0.6, zorder=3)
            if val > 12:
                ax.text(left + val / 2, y_pos[i], f"{val:.1f}",
                        va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
            left += val

        total = left
        all_totals.append(total)
        ax.text(total + 5, y_pos[i], f"Total: {total:.0f}",
                va="center", ha="left", fontsize=8.5, color="#333333")

    # Dashed separator between references and Cameroon
    if n_ref > 0 and n_ref < n:
        sep_y = (y_pos[n_ref - 1] + y_pos[n_ref]) / 2
        ax.axhline(sep_y, color="#999999", linewidth=1.0, linestyle="--", zorder=2)
        ax.text(2, sep_y - 0.22, "← Références",
                fontsize=8, color="#666666", style="italic")
        ax.text(2, sep_y + 0.14, "Cameroun ↓",
                fontsize=8, color="#666666", style="italic")

    # Axes
    x_max = max(all_totals) * 1.18 if all_totals else 100
    ax.set_xlim(0, x_max)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.45, color="#cccccc", zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="y", length=0)

    # Legend
    patches = [mpatches.Patch(color=colors[p], label=p) for p in phases]
    ax.legend(handles=patches, title="Phase", frameon=True, framealpha=0.92,
              loc="lower right", fontsize=9, title_fontsize=9, edgecolor="#cccccc")

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)

    fig.subplots_adjust(left=0.22, right=0.88, top=0.92, bottom=0.12)
    return fig


def show_gw_comparison_section():
    st.divider()
    st.subheader("9. Émissions CO₂eq — Comparaison avec pays de référence")
    st.markdown(
        "Compare les émissions de **Global Warming** (kg CO₂eq) par phase "
        "(Agriculture/Production · Transport · Transformation) entre les chaînes "
        "de valeur Cameroun et deux pays de référence."
    )

    # ── Reference values (editable) ──────────────────────────────────────
    with st.expander("✏️ Modifier les valeurs de référence", expanded=False):
        ref_cols = st.columns(2)
        references = {}
        for col_idx, (country, defaults) in enumerate(DEFAULT_REFERENCES.items()):
            with ref_cols[col_idx]:
                st.markdown(f"**{country}**")
                ag   = st.number_input(f"Agriculture – {country}",   value=defaults["Agriculture"],   key=f"ref_ag_{country}",   step=0.1)
                tr   = st.number_input(f"Transport – {country}",     value=defaults["Transport"],     key=f"ref_tr_{country}",   step=0.1)
                transf = st.number_input(f"Transformation – {country}", value=defaults["Transformation"], key=f"ref_tf_{country}", step=0.1)
                references[country] = {"Agriculture": ag, "Transport": tr, "Transformation": transf}

    # ── Cameroon data from combined CSV ──────────────────────────────────
    vc_file = st.file_uploader(
        "Charger le fichier CSV combiné Cameroun (même fichier que Section 8)",
        type=["csv"],
        key="gw_vc_file",
    )

    if vc_file is None:
        st.info("Chargez le fichier CSV pour ajouter les chaînes Cameroun.")
        # Show chart with references only
        cameroon_data = {}
    else:
        try:
            df = read_vc_combined(vc_file)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            return

        cat_col   = next((c for c in df.columns if "category" in c.lower() or "catégorie" in c.lower()), None)
        chain_col = next((c for c in df.columns if "chain" in c.lower() or "chaine" in c.lower()), None)

        if cat_col is None or chain_col is None:
            st.error(f"Colonnes requises introuvables. Colonnes : {df.columns.tolist()}")
            return

        gw_df = df[df[cat_col] == "Global warming"].copy()
        if gw_df.empty:
            st.warning("Aucune ligne 'Global warming' trouvée dans le fichier.")
            cameroon_data = {}
        else:
            all_chains = gw_df[chain_col].dropna().unique().tolist()
            selected_chains = st.multiselect(
                "Chaînes Cameroun à inclure",
                options=all_chains,
                default=all_chains,
                key="gw_selected_chains",
            )

            cameroon_data = {}
            for chain in selected_chains:
                row = gw_df[gw_df[chain_col] == chain]
                if row.empty:
                    continue
                cameroon_data[chain] = {
                    "Agriculture":    float(row["Production"].values[0])       if "Production"     in row.columns else 0.0,
                    "Transport":      float(row["Transport"].values[0])        if "Transport"      in row.columns else 0.0,
                    "Transformation": float(row["Transformation"].values[0])   if "Transformation" in row.columns else 0.0,
                }

    if not references and not cameroon_data:
        st.warning("Aucune donnée à afficher.")
        return

    # ── Chart controls ───────────────────────────────────────────────────
    chart_title = st.text_input(
        "Titre du graphique",
        value="Émissions de CO₂eq par phase du cycle de vie",
        key="gw_chart_title",
    )
    x_label = st.text_input(
        "Libellé axe X",
        value="Émissions de CO2eq (kg / tonne de produit)",
        key="gw_x_label",
    )

    fig = build_gw_comparison_chart(
        cameroon_data=cameroon_data,
        references=references,
        title=chart_title,
        x_label=x_label,
    )
    st.pyplot(fig)

    st.download_button(
        "⬇ Télécharger le graphique (PNG, 300 dpi)",
        data=fig_to_png_download(fig),
        file_name="gw_comparison_references.png",
        mime="image/png",
        key="gw_dl_png",
    )

    # Data table
    with st.expander("Voir les données du graphique"):
        table_rows = []
        for label, data in list(references.items()) + list(cameroon_data.items()):
            row = {"Chaîne / Pays": label, "Type": "Référence" if label in references else "Cameroun"}
            row.update(data)
            row["Total"] = sum(data.values())
            table_rows.append(row)
        tbl = pd.DataFrame(table_rows)
        st.dataframe(tbl, use_container_width=True)
        st.download_button(
            "⬇ Télécharger tableau (CSV)",
            data=tbl.to_csv(index=False).encode("utf-8"),
            file_name="gw_comparison_data.csv",
            mime="text/csv",
            key="gw_dl_csv",
        )
