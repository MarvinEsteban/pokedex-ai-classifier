"""Pokédex AI Classifier — interfaz Streamlit para el Random Forest de Legendarios."""

import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "modelo.joblib"
CSV_PATH = "pokemon.csv"
FEATURES = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
STATS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

st.set_page_config(page_title="Pokédex AI Classifier", page_icon="⚡", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


modelo = cargar_modelo()
IDX_LEGENDARIO = list(modelo.classes_).index(True)


@st.cache_resource
def cargar_pokedex():
    """Dataset de referencia y la hoja en la que cae cada Pokémon en cada árbol."""
    df = pd.read_csv(CSV_PATH).dropna(subset=["Name"]).reset_index(drop=True)
    hojas = modelo.apply(df[FEATURES])  # shape (n_pokemon, n_arboles)
    return df, hojas


def pokemon_similares(entrada, n=5):
    """Proximidad del Random Forest: % de árboles en que comparten hoja."""
    df, hojas = cargar_pokedex()
    hojas_entrada = modelo.apply(entrada)[0]
    proximidad = (hojas == hojas_entrada).mean(axis=1)
    # Desempate: a igual proximidad, el de estadísticas más cercanas
    distancia = ((df[STATS].to_numpy() - entrada[STATS].to_numpy()) ** 2).sum(axis=1)
    top = np.lexsort((distancia, -proximidad))[:n]
    res = df.loc[top, ["Name", "Legendary"] + STATS].copy()
    res["Total"] = res[STATS].sum(axis=1)
    res["Proximidad"] = proximidad[top]
    return res


# ---------------------------------------------------------------------------
# Estilos CSS personalizados (Pokédex / dashboard gaming oscuro)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0d0f17;
        --card: #161a26;
        --card-2: #1c2132;
        --border: #262c40;
        --crimson: #dc143c;
        --crimson-dark: #8b0a25;
        --gold: #ffcb05;
        --gold-soft: #f5b700;
        --blue: #4a90d9;
        --text: #e8eaf2;
        --muted: #8a91a8;
    }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a1020 0%, var(--bg) 45%) fixed;
        color: var(--text);
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1300px; }

    /* Cabecera tipo Pokédex */
    .dex-header {
        display: flex; align-items: center; gap: 1.2rem;
        background: linear-gradient(135deg, var(--crimson) 0%, var(--crimson-dark) 100%);
        border-radius: 20px; padding: 1.4rem 1.8rem; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(220, 20, 60, 0.25);
        border-bottom: 4px solid #000;
    }
    .dex-lens {
        width: 64px; height: 64px; border-radius: 50%; flex-shrink: 0;
        background: radial-gradient(circle at 35% 35%, #9fe3ff 0%, #2a8fd8 45%, #0b3d6b 100%);
        border: 5px solid #f2f2f2; box-shadow: 0 0 18px rgba(120, 210, 255, 0.7);
    }
    .dex-dots span {
        display: inline-block; width: 14px; height: 14px; border-radius: 50%;
        margin-right: 6px; border: 2px solid #000;
    }
    .dex-header h1 { margin: 0; font-size: 2rem; color: #fff; letter-spacing: .5px; }
    .dex-header p  { margin: .2rem 0 0; color: #ffd6de; }

    /* Tarjetas */
    .card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }
    .card h3 { margin-top: 0; color: var(--gold); font-size: 1.1rem; }
    .card-title {
        color: var(--gold); font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; font-size: .85rem; margin-bottom: .6rem;
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
    .stTabs [data-baseweb="tab"] {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 12px 12px 0 0; padding: .6rem 1.2rem; color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--card-2); color: var(--gold) !important;
        border-bottom: 3px solid var(--crimson);
    }

    /* Botón de predicción */
    div.stButton > button, div.stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--crimson) 0%, #ff4d6d 50%, var(--gold-soft) 100%);
        background-size: 200% 200%; background-position: 0% 50%;
        color: #fff; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
        border: none; border-radius: 14px; padding: .8rem 1rem;
        box-shadow: 0 6px 18px rgba(220, 20, 60, 0.35);
        transition: all .35s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-position: 100% 50%; color: #1a1a1a;
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 26px rgba(255, 203, 5, 0.45);
    }
    div.stButton > button:active { transform: translateY(0); }

    /* Tarjetas de resultado */
    .result {
        border-radius: 18px; padding: 1.5rem 1.6rem; margin-top: .3rem;
        text-align: center;
    }
    .result .label { font-size: .85rem; letter-spacing: 2px; text-transform: uppercase; }
    .result .verdict { font-size: 2.2rem; font-weight: 900; margin: .3rem 0; }
    .result-legend {
        background: linear-gradient(145deg, #2b2210 0%, #1a1508 100%);
        border: 3px solid var(--gold);
        box-shadow: 0 0 25px rgba(255, 203, 5, 0.55), inset 0 0 25px rgba(255, 203, 5, 0.12);
        animation: glow 2.2s ease-in-out infinite alternate;
    }
    .result-legend .label  { color: #ffe27a; }
    .result-legend .verdict { color: var(--gold); text-shadow: 0 0 14px rgba(255, 203, 5, .8); }
    .result-common {
        background: linear-gradient(145deg, #182233 0%, #141a26 100%);
        border: 2px solid #3a4a66;
        box-shadow: 0 6px 18px rgba(74, 144, 217, 0.15);
    }
    .result-common .label  { color: #9fb3d1; }
    .result-common .verdict { color: #cfe0f7; }
    @keyframes glow {
        from { box-shadow: 0 0 15px rgba(255, 203, 5, .35), inset 0 0 15px rgba(255, 203, 5, .08); }
        to   { box-shadow: 0 0 35px rgba(255, 203, 5, .75), inset 0 0 30px rgba(255, 203, 5, .18); }
    }

    /* Barra de votos */
    .vote-bar {
        background: #0f121b; border: 1px solid var(--border); border-radius: 999px;
        height: 22px; overflow: hidden; margin: .4rem 0 .2rem;
    }
    .vote-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }
    .vote-legend { background: linear-gradient(90deg, var(--gold-soft), var(--gold)); }
    .vote-common { background: linear-gradient(90deg, #2d5f9a, var(--blue)); }
    .vote-caption { display: flex; justify-content: space-between; color: var(--muted); font-size: .85rem; }

    /* Insignias de confianza del bosque */
    .badges { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-top: .9rem; }
    .badge {
        display: inline-block; padding: .35rem .85rem; border-radius: 999px;
        font-size: .8rem; font-weight: 700; letter-spacing: .5px;
    }
    .badge-sure  { background: rgba(61, 220, 132, .12); border: 1px solid #3ddc84; color: #7ef0b0; }
    .badge-doubt { background: rgba(255, 120, 40, .14); border: 1px solid #ff7828; color: #ffa36b; }
    .badge-600   { background: rgba(160, 110, 255, .14); border: 1px solid #a06eff; color: #c9aaff; }
    .badge-note  { color: var(--muted); font-size: .8rem; margin-top: .5rem; }

    /* Pokémon más parecidos */
    .sim-row {
        display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(40px, 1fr) 42px;
        align-items: center; gap: .7rem; padding: .45rem .7rem;
        margin: .3rem 0; border-radius: 10px; background: var(--card-2);
        border-left: 3px solid #3a4a66; font-size: .9rem;
    }
    .sim-legend { border-left-color: var(--gold); }
    .sim-name  { display: block; font-weight: 700; overflow-wrap: anywhere; line-height: 1.2; }
    .sim-meta  { display: block; font-size: .75rem; color: var(--muted); }
    .sim-tag   { color: #9fb3d1; }
    .sim-legend .sim-tag { color: var(--gold); }
    .sim-bar   { height: 8px; background: #0f121b; border-radius: 999px; overflow: hidden; }
    .sim-bar div { height: 100%; background: linear-gradient(90deg, #2d5f9a, var(--blue)); }
    .sim-legend .sim-bar div { background: linear-gradient(90deg, var(--gold-soft), var(--gold)); }
    .sim-pct   { width: 40px; text-align: right; font-weight: 800; }

    /* Resumen de stats */
    .stat-row { display: flex; align-items: center; gap: .6rem; margin: .25rem 0; font-size: .9rem; }
    .stat-row .name { width: 70px; color: var(--muted); }
    .stat-row .val  { width: 36px; text-align: right; font-weight: 700; }
    .stat-row .bar  { flex: 1; height: 8px; background: #0f121b; border-radius: 999px; overflow: hidden; }
    .stat-row .bar div { height: 100%; background: linear-gradient(90deg, var(--crimson), #ff6b81); }
    .total-badge {
        display: inline-block; margin-top: .6rem; padding: .35rem .9rem; border-radius: 999px;
        background: rgba(255, 203, 5, .12); border: 1px solid var(--gold); color: var(--gold); font-weight: 800;
    }

    /* st.metric como tarjetas */
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--border);
        border-left: 4px solid var(--crimson); border-radius: 14px;
        padding: 1rem 1.2rem; box-shadow: 0 6px 18px rgba(0, 0, 0, .3);
    }
    div[data-testid="stMetricLabel"] p { color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-size: .8rem; }
    div[data-testid="stMetricValue"] { color: var(--gold); }

    /* Uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background: var(--card-2); border: 2px dashed var(--crimson); border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dex-header">
        <div class="dex-lens"></div>
        <div>
            <div class="dex-dots">
                <span style="background:#ff3b3b"></span><span style="background:#ffcb05"></span><span style="background:#3ddc84"></span>
            </div>
            <h1>Pokédex AI Classifier</h1>
            <p>Random Forest de 100 árboles que detecta Pokémon <b>Legendarios</b> a partir de sus estadísticas base.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ind, tab_csv, tab_rf = st.tabs(
    ["⚡ Predicción Individual", "📂 Inferencia Masiva (CSV)", "🌲 Métricas del Bosque"]
)

# ---------------------------------------------------------------------------
# a) Predicción individual
# ---------------------------------------------------------------------------
with tab_ind:
    col_form, col_res = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="card-title">Estadísticas base</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hp = st.slider("HP", 1, 255, 70)
            sp_atk = st.slider("Sp. Atk", 1, 200, 70)
        with c2:
            attack = st.slider("Attack", 1, 200, 80)
            sp_def = st.slider("Sp. Def", 1, 230, 70)
        with c3:
            defense = st.slider("Defense", 1, 230, 70)
            speed = st.slider("Speed", 1, 180, 70)
        generation = st.selectbox("Generation", [1, 2, 3, 4, 5, 6], index=0)
        predecir = st.button("⚡ Analizar Pokémon")

    with col_res:
        st.markdown('<div class="card-title">Veredicto del Random Forest</div>', unsafe_allow_html=True)
        valores = {
            "HP": hp, "Attack": attack, "Defense": defense,
            "Sp. Atk": sp_atk, "Sp. Def": sp_def, "Speed": speed,
            "Generation": generation,
        }
        if predecir:
            entrada = pd.DataFrame([valores], columns=FEATURES)
            es_legendario = bool(modelo.predict(entrada)[0])
            prob_leg = float(modelo.predict_proba(entrada)[0][IDX_LEGENDARIO])
            prob_mostrar = prob_leg if es_legendario else 1 - prob_leg
            clase = "legend" if es_legendario else "common"
            texto = "✨ LEGENDARIO ✨" if es_legendario else "Común"

            # Confianza del bosque: complementa al veredicto, no lo sustituye
            if prob_leg > 0.70 or prob_leg < 0.35:
                badge = '<span class="badge badge-sure">✔ Alta confianza del bosque</span>'
                nota = ""
            else:
                badge = '<span class="badge badge-doubt">🔥 Zona de duda · Élite</span>'
                nota = (
                    '<div class="badge-note">Los árboles están divididos: estadísticas de élite '
                    "(Total ≈ 580–600), donde se mezclan legendarios, Megas y pseudolegendarios.</div>"
                )
            total_stats = sum(valores[s] for s in STATS)
            if total_stats == 600:
                badge += '<span class="badge badge-600">💎 Total 600 · perfil pseudo-legendario</span>'

            st.markdown(
                f"""
                <div class="result result-{clase}">
                    <div class="label">Clasificación</div>
                    <div class="verdict">{texto}</div>
                    <div class="label">{prob_mostrar:.0%} de los árboles coinciden</div>
                    <div class="badges">{badge}</div>
                    {nota}
                </div>
                """,
                unsafe_allow_html=True,
            )

            arboles_leg = round(prob_leg * modelo.n_estimators)
            st.markdown(
                f"""
                <div class="card" style="margin-top:1rem">
                    <div class="card-title">Votos de los árboles (predict_proba)</div>
                    <div class="vote-bar"><div class="vote-fill vote-{clase}" style="width:{prob_leg * 100:.1f}%"></div></div>
                    <div class="vote-caption">
                        <span>Legendario: <b>{prob_leg:.1%}</b> (~{arboles_leg} árboles)</span>
                        <span>Común: <b>{1 - prob_leg:.1%}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prob_leg, text=f"Probabilidad de ser Legendario: {prob_leg:.1%}")

            similares = pokemon_similares(entrada)
            filas_sim = "".join(
                f'<div class="sim-row sim-{"legend" if r.Legendary else "common"}">'
                f'<span><span class="sim-name">{html.escape(r.Name)}</span>'
                f'<span class="sim-meta"><span class="sim-tag">{"Legendario" if r.Legendary else "Común"}</span>'
                f" · Total {r.Total}</span></span>"
                f'<span class="sim-bar"><div style="width:{r.Proximidad * 100:.0f}%"></div></span>'
                f'<span class="sim-pct">{r.Proximidad:.0%}</span></div>'
                for r in similares.itertuples()
            )
            st.markdown(
                f"""
                <div class="card" style="margin-top:1rem">
                    <div class="card-title">Pokémon más parecidos (proximidad del bosque)</div>
                    {filas_sim}
                    <div class="badge-note">% de los {modelo.n_estimators} árboles en los que
                    comparte hoja con cada Pokémon del dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="card" style="text-align:center;color:#8a91a8">'
                "Ajusta las estadísticas y pulsa <b>Analizar Pokémon</b>.</div>",
                unsafe_allow_html=True,
            )

        total = sum(valores[s] for s in STATS)
        filas = "".join(
            f'<div class="stat-row"><span class="name">{s}</span><span class="val">{valores[s]}</span>'
            f'<span class="bar"><div style="width:{valores[s] / 255 * 100:.0f}%"></div></span></div>'
            for s in STATS
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Resumen de estadísticas</div>
                {filas}
                <span class="total-badge">TOTAL: {total}</span>
                <span style="color:#8a91a8;margin-left:.6rem">Generación {generation}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# b) Inferencia masiva (CSV)
# ---------------------------------------------------------------------------
with tab_csv:
    st.markdown(
        '<div class="card"><div class="card-title">Carga por lotes</div>'
        f"El CSV debe contener las columnas: <code>{', '.join(FEATURES)}</code>. "
        "El resto de columnas se conservan en el resultado.</div>",
        unsafe_allow_html=True,
    )
    archivo = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        faltan = [c for c in FEATURES if c not in df.columns]
        if faltan:
            st.error(f"Faltan columnas obligatorias: {', '.join(faltan)}")
        else:
            X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
            validas = X.notna().all(axis=1)
            resultado = df.copy()
            resultado["Predicción"] = pd.NA
            resultado["Prob. Legendario"] = pd.NA
            if validas.any():
                resultado.loc[validas, "Predicción"] = [
                    "Legendario" if p else "Común" for p in modelo.predict(X[validas])
                ]
                resultado.loc[validas, "Prob. Legendario"] = (
                    modelo.predict_proba(X[validas])[:, IDX_LEGENDARIO].round(3)
                )

            n_leg = int((resultado["Predicción"] == "Legendario").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Filas procesadas", int(validas.sum()))
            m2.metric("Legendarios detectados", n_leg)
            m3.metric("Filas inválidas", int((~validas).sum()))

            st.dataframe(resultado, width="stretch", height=450)
            st.download_button(
                "⬇ Descargar predicciones",
                resultado.to_csv(index=False).encode("utf-8"),
                file_name="predicciones.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# c) Métricas del bosque
# ---------------------------------------------------------------------------
with tab_rf:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Árboles (n_estimators)", modelo.n_estimators)
    k2.metric("Criterio de división", modelo.criterion.capitalize())
    k3.metric("Precisión OOB", f"{modelo.oob_score_:.2%}")
    k4.metric("Profundidad máx.", modelo.max_depth)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("max_features", str(modelo.max_features))
    k6.metric("min_samples_split", modelo.min_samples_split)
    k7.metric("Validación", "Out-of-Bag")
    k8.metric("Variables", modelo.n_features_in_)

    st.markdown(
        '<div class="card" style="margin-top:1rem"><div class="card-title">'
        "Importancia de las variables (feature_importances_)</div>"
        "Reducción media de impureza Gini aportada por cada variable en los 100 árboles.</div>",
        unsafe_allow_html=True,
    )
    importancias = pd.DataFrame(
        {"Importancia": modelo.feature_importances_}, index=FEATURES
    ).sort_values("Importancia", ascending=False)
    st.bar_chart(importancias, color="#dc143c", height=380)
