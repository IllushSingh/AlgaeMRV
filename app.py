from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.model import (
    compute_ndci,
    fit_biomass_model,
    calculate_carbon,
    summarize_carbon,
    data_quality_score,
)
from modules.farm_intelligence import load_bundle, forecast_week, model_summary

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "algaemrv_test_data.csv"
SPATIAL_DATA = ROOT / "data" / "algaemrv_spatial_pixels.csv"
MODEL_PATH = ROOT / "models" / "atp3_forecast.joblib"

ACCENT = "#2de3a7"
ACCENT_2 = "#38bdf8"
DANGER = "#fb7185"
TEXT = "#e8f6f2"
MUTED = "#8aa6b1"
GRID = "rgba(138,166,177,0.13)"
NDCI_SCALE = [
    [0.00, "#07202a"],
    [0.25, "#0d5a63"],
    [0.50, "#12a382"],
    [0.75, "#79e3a2"],
    [1.00, "#eaffd4"],
]

st.set_page_config(
    page_title="AlgaeMRV",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --surface:rgba(18,48,62,.45);
  --border:rgba(45,227,167,.16);
  --border-hi:rgba(45,227,167,.42);
  --accent:#2de3a7;
  --accent2:#38bdf8;
  --text:#e8f6f2;
  --muted:#8aa6b1;
}

.stApp{
  background:
    radial-gradient(1100px 620px at 10% -12%, rgba(45,227,167,.13), transparent 60%),
    radial-gradient(900px 520px at 90% -4%, rgba(56,189,248,.10), transparent 58%),
    linear-gradient(180deg,#050e14 0%,#04121a 46%,#030c12 100%);
  background-attachment:fixed;
  font-family:'Inter',system-ui,-apple-system,sans-serif;
}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stDecoration"], .stAppDeployButton, #MainMenu{display:none;}
.block-container{padding-top:2rem;padding-bottom:3.5rem;max-width:1440px;}

.hero{
  position:relative;overflow:hidden;
  border:1px solid var(--border);border-radius:22px;
  padding:34px 38px 36px;margin-bottom:8px;
  background:linear-gradient(135deg,rgba(45,227,167,.11),rgba(56,189,248,.05) 48%,rgba(5,14,20,.25));
}
.hero:before{
  content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(720px 270px at 4% 0%,rgba(45,227,167,.22),transparent 72%);
}
.hero-top{display:flex;align-items:center;gap:16px;position:relative;}
.brand{font-size:.78rem;font-weight:700;letter-spacing:.24em;text-transform:uppercase;color:var(--muted);}
.brand b{color:var(--accent);font-weight:800;}
.hero h1{
  position:relative;margin:20px 0 0;
  font-size:clamp(2.2rem,3.8vw,3.4rem);
  line-height:1.05;font-weight:800;letter-spacing:-.04em;
  color:#f4fcf9;text-wrap:balance;
}
/* the hand-placed break only reads well once the card is wide enough */
@media (max-width:1150px){ .hero h1 br{display:none;} }
.hero h1 em{
  font-style:normal;
  background:linear-gradient(96deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  -webkit-box-decoration-break:clone;box-decoration-break:clone;
}
.hero-sub{position:relative;margin:18px 0 0;color:var(--muted);font-size:.95rem;font-weight:500;}
.hero-sub span{color:var(--accent);font-weight:700;margin:0 9px;}

.chip-row{display:flex;flex-wrap:wrap;gap:9px;margin:4px 0 6px;}
.chip{
  display:inline-flex;align-items:center;gap:7px;
  padding:6px 13px;border-radius:999px;
  font-size:.75rem;font-weight:600;letter-spacing:.01em;
  border:1px solid transparent;white-space:nowrap;
}
.chip .dot{width:7px;height:7px;border-radius:50%;background:currentColor;flex:none;}
.chip.ok{color:#7ef0c6;background:rgba(45,227,167,.10);border-color:rgba(45,227,167,.30);}
.chip.warn{color:#fcd66b;background:rgba(251,191,36,.10);border-color:rgba(251,191,36,.30);}
.chip.info{color:#8fd3f7;background:rgba(56,189,248,.10);border-color:rgba(56,189,248,.28);}
.chip.mute{color:var(--muted);background:rgba(138,166,177,.09);border-color:rgba(138,166,177,.22);}

.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px;margin:8px 0 6px;}
.kpi{
  position:relative;overflow:hidden;
  border:1px solid var(--border);border-radius:16px;
  padding:17px 18px 15px;
  background:linear-gradient(160deg,rgba(18,48,62,.70),rgba(8,24,32,.48));
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
}
.kpi:hover{
  transform:translateY(-3px);border-color:var(--border-hi);
  box-shadow:0 14px 32px -14px rgba(45,227,167,.5);
}
.kpi:before{
  content:"";position:absolute;left:18px;right:18px;top:0;height:2px;border-radius:0 0 3px 3px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));opacity:.9;
}
.kpi-label{font-size:.67rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.kpi-value{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:1.7rem;font-weight:700;color:var(--text);line-height:1.15;margin-top:9px;
}
.kpi-unit{font-family:'Inter',sans-serif;font-size:.78rem;color:var(--muted);font-weight:500;margin-left:5px;}
.kpi-delta{
  display:inline-flex;align-items:center;gap:5px;margin-top:10px;
  padding:3px 10px;border-radius:999px;font-size:.74rem;font-weight:700;
}
.kpi-delta.up{color:#7ef0c6;background:rgba(45,227,167,.13);}
.kpi-delta.down{color:#fda4af;background:rgba(251,113,133,.13);}
.kpi-delta.flat{color:var(--muted);background:rgba(138,166,177,.13);}
.kpi-sub{margin-top:9px;font-size:.73rem;color:var(--muted);line-height:1.4;}

.sec{display:flex;align-items:center;gap:11px;margin:30px 0 12px;}
.sec-bar{width:3px;height:19px;border-radius:3px;background:linear-gradient(180deg,var(--accent),var(--accent2));flex:none;}
.sec h3{margin:0;font-size:1.04rem;font-weight:700;letter-spacing:-.012em;color:var(--text);}
.sec-note{margin-left:auto;font-size:.75rem;color:var(--muted);text-align:right;}

div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(160deg,rgba(18,48,62,.38),rgba(8,24,32,.22));
  border-color:var(--border) !important;
  border-radius:16px;
}

.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid rgba(255,255,255,.06);}
.stTabs [data-baseweb="tab"]{
  height:46px;padding:0 20px;border-radius:11px 11px 0 0;
  background:transparent;color:var(--muted);font-weight:600;font-size:.92rem;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--text);background:rgba(255,255,255,.03);}
.stTabs [aria-selected="true"]{
  color:var(--accent) !important;
  background:linear-gradient(180deg,rgba(45,227,167,.15),transparent);
}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent);height:2px;}
.stTabs [data-baseweb="tab-border"]{display:none;}

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#071a23,#050e14 60%);
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{padding-top:1.6rem;}
.side-title{
  font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);font-weight:700;margin:2px 0 2px;
}

.stDownloadButton button,.stButton button{
  border:1px solid var(--border-hi);background:rgba(45,227,167,.08);
  color:var(--accent);font-weight:600;border-radius:11px;transition:.18s ease;
}
.stDownloadButton button:hover,.stButton button:hover{
  background:rgba(45,227,167,.18);border-color:var(--accent);color:#eafff7;
  transform:translateY(-1px);
}

[data-testid="stExpander"]{border:1px solid var(--border);border-radius:13px;background:rgba(8,24,32,.35);}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted) !important;}
hr{border-color:rgba(255,255,255,.06);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def chip(text, tone="ok"):
    return f'<span class="chip {tone}"><span class="dot"></span>{text}</span>'


def chip_row(*chips):
    st.markdown('<div class="chip-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def kpi(label, value, unit="", delta=None, tone="flat", sub=None):
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    delta_html = f'<div class="kpi-delta {tone}">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}{unit_html}</div>{delta_html}{sub_html}</div>'
    )


def kpi_row(*cards):
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section(title, note=""):
    note_html = f'<div class="sec-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="sec"><span class="sec-bar"></span><h3>{title}</h3>{note_html}</div>',
        unsafe_allow_html=True,
    )


def trend(value, good_up=True):
    if abs(value) < 1e-9:
        return "■ ", "flat"
    rising = value > 0
    return ("▲ " if rising else "▼ "), ("up" if rising == good_up else "down")


def style_fig(fig, height=360, title=None, showlegend=True):
    # Legend sits above the plot, so a titled chart needs room for both rows.
    if title and showlegend:
        top_margin = 86
    elif title:
        top_margin = 52
    elif showlegend:
        top_margin = 46
    else:
        top_margin = 22
    fig.update_layout(
        height=height,
        margin={"l": 12, "r": 12, "t": top_margin, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "size": 13, "color": TEXT},
        showlegend=showlegend,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#0b2733",
            "bordercolor": "rgba(45,227,167,.42)",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": TEXT},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": MUTED},
        },
    )
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 15, "color": TEXT, "family": "Inter, sans-serif"},
                "x": 0,
                "xanchor": "left",
                "y": 0.99,
                "yanchor": "top",
            }
        )
    axis = {
        "gridcolor": GRID,
        "zerolinecolor": "rgba(138,166,177,.25)",
        "linecolor": GRID,
        "tickfont": {"color": MUTED, "size": 11},
        "title_font": {"color": MUTED, "size": 11},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


with st.sidebar:
    st.markdown('<div class="side-title">Carbon assumptions</div>', unsafe_allow_html=True)
    st.caption("Applied to all carbon calculations below.")

    carbon_fraction = st.slider("Dry-biomass carbon fraction", 0.30, 0.60, 0.48, 0.01)
    grid_ef = st.number_input("Electricity factor (kg CO₂e/kWh)", 0.0, 2.0, 0.70, 0.01)
    transport_ef = st.number_input("Transport factor (kg CO₂e/km)", 0.0, 2.0, 0.18, 0.01)
    uncertainty = st.slider("Conservative uncertainty deduction", 0.0, 0.50, 0.10, 0.01)
    fate = st.selectbox(
        "Biomass fate",
        ["Durable material", "Long-term storage", "Bioplastic", "Feed", "Biofuel"],
    )
    permanence_lookup = {
        "Durable material": 0.75,
        "Long-term storage": 0.90,
        "Bioplastic": 0.50,
        "Feed": 0.10,
        "Biofuel": 0.05,
    }
    permanence = permanence_lookup[fate]
    st.markdown(
        chip(f"Permanence factor {permanence:.2f}", "info"),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="side-title">Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Optional monitoring CSV", type="csv")


def build_forecast_carbon(forecast, origin_row, recent_history):
    """Convert predicted biomass concentration into a simple 7-day carbon outlook."""
    out = forecast.copy()
    pond_volume = float(origin_row["pond_area_m2"] * origin_row["water_depth_m"])
    previous_total = float(origin_row["estimated_biomass_kg_m3"] * pond_volume)

    recent = recent_history.tail(3)
    electricity = float(recent["electricity_kWh_day"].mean())
    transport = float(recent["transport_km_day"].mean())
    daily_operational = electricity * grid_ef + transport * transport_ef

    gross_daily = []
    for predicted_concentration in out["predicted_biomass_g_l"]:
        total = float(predicted_concentration) * pond_volume  # 1 g/L == 1 kg/m3
        gain = max(total - previous_total, 0.0)
        gross_daily.append(gain * carbon_fraction * (44.0 / 12.0))
        previous_total = total

    out["projected_gross_CO2_kg_day"] = gross_daily
    out["projected_operational_emissions_kgCO2e_day"] = daily_operational
    out["projected_cumulative_gross_CO2_kg"] = out["projected_gross_CO2_kg_day"].cumsum()
    out["projected_cumulative_operational_kgCO2e"] = out[
        "projected_operational_emissions_kgCO2e_day"
    ].cumsum()
    out["projected_cumulative_net_kgCO2e"] = (
        out["projected_cumulative_gross_CO2_kg"]
        - out["projected_cumulative_operational_kgCO2e"]
    )
    out["projected_durable_removal_kgCO2e"] = (
        out["projected_cumulative_net_kgCO2e"].clip(lower=0)
        * (1.0 - uncertainty)
        * permanence
    )
    return out


df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DEFAULT_DATA)
df["date"] = pd.to_datetime(df["date"])
df["ndci"] = compute_ndci(df)
biomass_model, calibration, calibration_r2, calibration_rmse = fit_biomass_model(df)
results = calculate_carbon(
    df,
    biomass_model,
    carbon_fraction=carbon_fraction,
    grid_ef=grid_ef,
    transport_ef=transport_ef,
    uncertainty_deduction=uncertainty,
    permanence_factor=permanence,
)
results = results.sort_values(["pond_id", "date"]).reset_index(drop=True)
dq = data_quality_score(results, calibration_r2)

st.markdown(
    '<div class="hero">'
    '<div class="hero-top">'
    '<div class="brand">🌿 Algae<b>MRV</b></div>'
    "</div>"
    "<h1><em>Satellite-verified</em> carbon accounting <br/>for algae farms</h1>"
    '<p class="hero-sub">Monitor<span>→</span>predict<span>→</span>account</p>'
    "</div>",
    unsafe_allow_html=True,
)

window = f'{results["date"].min():%d %b} - {results["date"].max():%d %b %Y}'
chip_row(
    chip(f'Pond {results["pond_id"].iloc[0]}', "info"),
    chip(f"Monitoring window {window}", "mute"),
    chip(f"{len(results)} daily observations", "mute"),
    chip(f"Calibration R² {calibration_r2:.3f}", "ok" if calibration_r2 > 0.7 else "warn"),
)

dates = results["date"].dt.strftime("%Y-%m-%d").tolist()
default_origin_index = max(1, len(dates) - 8)

section("Forecast origin", "Pick any day in the monitoring window")
with st.container(border=True):
    origin_label = st.select_slider(
        "Forecast origin",
        options=dates,
        value=dates[default_origin_index],
        label_visibility="collapsed",
        help="The ML forecast uses the farm state at this date. Later demo observations are shown only for comparison.",
    )

origin_date = pd.Timestamp(origin_label)
origin_index = int(results.index[results["date"].eq(origin_date)][0])
origin = results.loc[origin_index]
history = results.iloc[: origin_index + 1].copy()
current_carbon = summarize_carbon(history)

previous_row = results.iloc[origin_index - 1] if origin_index > 0 else None
biomass_change = (
    float(origin["estimated_biomass_kg_m3"] - previous_row["estimated_biomass_kg_m3"])
    if previous_row is not None
    else 0.0
)
ndci_change = float(origin["ndci"] - previous_row["ndci"]) if previous_row is not None else 0.0

arrow, tone = trend(biomass_change)
ndci_arrow, ndci_tone = trend(ndci_change)
net_balance = current_carbon["net_biological_balance_kgCO2e"]

section("Farm state at origin", origin_date.strftime("%A, %d %B %Y"))
kpi_row(
    kpi(
        "Calculated biomass",
        f'{origin["estimated_biomass_kg_m3"]:.3f}',
        "kg/m³",
        f"{arrow}{abs(biomass_change):.3f} vs previous day",
        tone,
    ),
    kpi(
        "NDCI",
        f'{origin["ndci"]:.3f}',
        "",
        f"{ndci_arrow}{abs(ndci_change):.4f} vs previous day",
        ndci_tone,
        sub="Normalised Difference Chlorophyll Index",
    ),
    kpi(
        "Net carbon balance to date",
        f"{net_balance:+.1f}",
        "kg CO₂e",
        "▲ net removal" if net_balance > 0 else "▼ net emitter",
        "up" if net_balance > 0 else "down",
    ),
    kpi(
        "Monitoring data quality",
        f"{dq:.1f}",
        "/ 100",
        "Above threshold" if dq >= 80 else "Needs review",
        "up" if dq >= 80 else "flat",
    ),
)

health_col, gauge_col = st.columns([1.75, 1], gap="medium")

with health_col:
    section("Pond conditions")
    with st.container(border=True):
        st.markdown(
            '<div class="chip-row" style="margin:6px 0 2px">'
            + chip(
                f'Temperature {origin["temperature_C"]:.1f} °C',
                "ok" if 20 <= origin["temperature_C"] <= 32 else "warn",
            )
            + chip(f'pH {origin["pH"]:.2f}', "ok" if 6.5 <= origin["pH"] <= 8.6 else "warn")
            + chip(
                f'Dissolved O₂ {origin["dissolved_oxygen_mg_L"]:.2f} mg/L',
                "ok" if 4 <= origin["dissolved_oxygen_mg_L"] <= 15 else "warn",
            )
            + chip(
                f'Depth {origin["water_depth_m"]:.2f} m',
                "ok" if 0.15 <= origin["water_depth_m"] <= 0.6 else "warn",
            )
            + chip(f'EC {origin["EC_mS_cm"]:.2f} mS/cm', "mute")
            + chip(f'CO₂ feed {origin["CO2_feed_kg_day"]:.1f} kg/day', "info")
            + "</div>",
            unsafe_allow_html=True,
        )
        recent_window = history.tail(14)
        spark = go.Figure()
        spark.add_trace(
            go.Scatter(
                x=recent_window["date"],
                y=recent_window["estimated_biomass_kg_m3"],
                mode="lines",
                line={"color": ACCENT, "width": 2.5, "shape": "spline"},
                fill="tozeroy",
                fillcolor="rgba(45,227,167,.10)",
                name="Biomass",
                hovertemplate="%{x|%d %b}<br>%{y:.3f} kg/m³<extra></extra>",
            )
        )
        style_fig(spark, height=150, showlegend=False)
        spark.update_yaxes(title=None, showgrid=False)
        spark.update_xaxes(title=None)
        st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False})

with gauge_col:
    section("Data quality")
    with st.container(border=True):
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=dq,
                number={
                    "suffix": "<span style='font-size:.9rem;color:#8aa6b1'> / 100</span>",
                    "font": {"size": 36, "color": TEXT, "family": "JetBrains Mono, monospace"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": MUTED,
                        "tickfont": {"size": 10, "color": MUTED},
                    },
                    "bar": {"color": ACCENT, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(251,113,133,.16)"},
                        {"range": [50, 80], "color": "rgba(251,191,36,.16)"},
                        {"range": [80, 100], "color": "rgba(45,227,167,.16)"},
                    ],
                    "threshold": {
                        "line": {"color": ACCENT_2, "width": 3},
                        "thickness": 0.8,
                        "value": 80,
                    },
                },
            )
        )
        style_fig(gauge, height=218, showlegend=False)
        gauge.update_layout(margin={"l": 24, "r": 24, "t": 16, "b": 4})
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.caption("Completeness, plausibility, NDCI validity, calibration fit")

forecast = None
forecast_warnings = []
model_bundle = None
if MODEL_PATH.exists():
    try:
        model_bundle = load_bundle(MODEL_PATH)
        if previous_row is not None:
            gap = max(1, int((origin["date"] - previous_row["date"]).days))
            recent_growth = (
                float(origin["estimated_biomass_kg_m3"])
                - float(previous_row["estimated_biomass_kg_m3"])
            ) / gap
        else:
            recent_growth = np.nan

        state = {
            "biomass_g_l": float(origin["estimated_biomass_kg_m3"]),
            "recent_growth_g_l_day": recent_growth,
            "ph": float(origin["pH"]),
            "temperature_C": float(origin["temperature_C"]),
            "dissolved_oxygen_mg_L": float(origin["dissolved_oxygen_mg_L"]),
            "water_depth_m": float(origin["water_depth_m"]),
            "culture_age_days": float((origin["date"] - results["date"].min()).days),
        }
        forecast, forecast_warnings = forecast_week(model_bundle, state, origin["date"])
        forecast = build_forecast_carbon(forecast, origin, history)
    except (ValueError, KeyError, OSError, EOFError, ImportError, pickle.UnpicklingError) as exc:
        st.error(f"Forecast model could not be loaded: {exc}")
else:
    st.warning("Forecast model is not trained yet. Run `python train_atp3_model.py`, then reload this page.")

if forecast is not None:
    final = forecast.iloc[-1]
    growth_pct = 100.0 * (
        final["predicted_biomass_g_l"] / origin["estimated_biomass_kg_m3"] - 1.0
    )
    growth_arrow, growth_tone = trend(growth_pct)
    net_projected = final["projected_cumulative_net_kgCO2e"]

    section(
        "7-day ATP³ forecast",
        f'{model_bundle.get("model_name")}, trained on {model_bundle.get("training_pairs"):,} ATP³ pairs',
    )
    kpi_row(
        kpi(
            "Predicted biomass · Day +7",
            f'{final["predicted_biomass_g_l"]:.3f}',
            "kg/m³",
            f'± {final["validation_mae_g_l"]:.3f} validation MAE',
            "flat",
        ),
        kpi("Predicted 7-day growth", f"{growth_pct:+.1f}", "%", f"{growth_arrow}vs origin", growth_tone),
        kpi(
            "Projected 7-day net CO₂",
            f"{net_projected:+.1f}",
            "kg CO₂e",
            "▲ removal" if net_projected > 0 else "▼ emission",
            "up" if net_projected > 0 else "down",
        ),
        kpi(
            "Projected durable removal",
            f'{final["projected_durable_removal_kgCO2e"]:.1f}',
            "kg CO₂e",
            f"after {uncertainty:.0%} deduction, {fate.lower()}",
            "flat",
        ),
    )

    actual_future = results[
        (results["date"] > origin_date)
        & (results["date"] <= origin_date + pd.Timedelta(days=7))
    ][["date", "estimated_biomass_kg_m3", "gross_CO2_fixed_kg", "operational_emissions_kgCO2e"]].copy()
    actual_future["actual_cumulative_net_kgCO2e"] = (
        actual_future["gross_CO2_fixed_kg"]
        - actual_future["operational_emissions_kgCO2e"]
    ).cumsum()

    band_dates = pd.concat([pd.Series([origin_date]), forecast["date"]], ignore_index=True)
    band_mid = pd.concat(
        [
            pd.Series([float(origin["estimated_biomass_kg_m3"])]),
            forecast["predicted_biomass_g_l"],
        ],
        ignore_index=True,
    )
    band_error = pd.concat(
        [pd.Series([0.0]), forecast["validation_mae_g_l"].astype(float)], ignore_index=True
    )

    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pd.concat([band_dates, band_dates[::-1]], ignore_index=True),
                y=pd.concat(
                    [band_mid + band_error, (band_mid - band_error).clip(lower=0)[::-1]],
                    ignore_index=True,
                ),
                fill="toself",
                fillcolor="rgba(56,189,248,.13)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name="Forecast uncertainty (± validation MAE)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=results["date"],
                y=results["estimated_biomass_kg_m3"],
                mode="lines+markers",
                name="Calculated from monitoring data",
                line={"color": ACCENT, "width": 2.6},
                marker={"size": 6, "color": ACCENT, "line": {"width": 0}},
                hovertemplate="%{y:.3f} kg/m³<extra>Monitored</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_dates,
                y=band_mid,
                mode="lines+markers",
                name="ATP³ ML forecast",
                line={"color": ACCENT_2, "width": 2.6, "dash": "dash"},
                marker={"size": 7, "color": ACCENT_2, "symbol": "diamond"},
                hovertemplate="%{y:.3f} kg/m³<extra>Forecast</extra>",
            )
        )
        fig.add_vline(
            x=origin_date.to_pydatetime(),
            line_dash="dot",
            line_color="rgba(232,246,242,.45)",
            annotation_text="forecast origin",
            annotation_position="top",
            annotation_font={"color": MUTED, "size": 11},
        )
        style_fig(fig, height=470, title="Calculated biomass vs 7-day forecast")
        fig.update_yaxes(title_text="Dry biomass (kg/m³ = g/L)")
        fig.update_xaxes(title_text=None)
        st.plotly_chart(fig, use_container_width=True)

    comparison = forecast[
        [
            "date",
            "horizon_days",
            "predicted_biomass_g_l",
            "projected_cumulative_net_kgCO2e",
        ]
    ].merge(
        actual_future[["date", "estimated_biomass_kg_m3", "actual_cumulative_net_kgCO2e"]],
        on="date",
        how="left",
    )
    comparison["error"] = (
        comparison["predicted_biomass_g_l"] - comparison["estimated_biomass_kg_m3"]
    ).abs()
    comparison = comparison.rename(
        columns={
            "horizon_days": "Day",
            "predicted_biomass_g_l": "Predicted biomass (kg/m³)",
            "estimated_biomass_kg_m3": "Calculated biomass (kg/m³)",
            "projected_cumulative_net_kgCO2e": "Predicted cumulative net CO₂e (kg)",
            "actual_cumulative_net_kgCO2e": "Calculated cumulative net CO₂e (kg)",
            "error": "Absolute error (kg/m³)",
        }
    )
    comparison["Day"] = comparison["Day"].map(lambda x: f"+{int(x)}")
    comparison["date"] = comparison["date"].dt.strftime("%d %b")

    section("Predicted vs calculated", "Against bundled demo observations")
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn("Date", width="small"),
            "Day": st.column_config.TextColumn("Horizon", width="small"),
            "Predicted biomass (kg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "Calculated biomass (kg/m³)": st.column_config.NumberColumn(format="%.3f"),
            "Predicted cumulative net CO₂e (kg)": st.column_config.NumberColumn(format="%.1f"),
            "Calculated cumulative net CO₂e (kg)": st.column_config.NumberColumn(format="%.1f"),
            "Absolute error (kg/m³)": st.column_config.ProgressColumn(
                "Absolute error", format="%.3f", min_value=0.0, max_value=0.2
            ),
        },
    )

    if model_bundle is not None:
        with st.expander("Model validation details"):
            st.json(model_summary(model_bundle))
            if forecast_warnings:
                st.write("Input notes:")
                for warning in forecast_warnings:
                    st.write("- " + warning)

st.caption(
    "Forecast values are predictions from real historical ATP³ cultivation data. "
    "The bundled monitoring run is synthetic and is used here as a replay/demo to compare predicted vs calculated values."
)

full_summary = summarize_carbon(results)

tab1, tab2, tab3 = st.tabs(["🛰  Remote sensing", "📊  Carbon MRV", "🔒  Audit"])

with tab1:
    spatial = pd.read_csv(SPATIAL_DATA)
    spatial["date"] = pd.to_datetime(spatial["date"])
    spatial["ndci"] = (
        spatial["B5_rededge_reflectance"] - spatial["B4_red_reflectance"]
    ) / (
        spatial["B5_rededge_reflectance"] + spatial["B4_red_reflectance"]
    )
    spatial["label"] = spatial["date"].dt.strftime("%Y-%m-%d")
    capture_labels = sorted(spatial["label"].unique())
    capture_cube = np.stack(
        [
            spatial[spatial["label"].eq(label)]
            .pivot(index="y", columns="x", values="ndci")
            .to_numpy()
            for label in capture_labels
        ]
    )

    # Satellite only revisits every 7 days, so interpolate each pixel's deviation
    # from its capture mean and re-centre it on the NDCI measured that day.
    capture_times = pd.to_datetime(capture_labels).astype("int64").to_numpy()
    anomaly = capture_cube - capture_cube.mean(axis=(1, 2), keepdims=True)
    flat_anomaly = anomaly.reshape(len(capture_labels), -1)

    daily = (
        results[["date", "ndci"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["biomass"] = biomass_model.predict(daily[["ndci"]].to_numpy())
    daily_times = daily["date"].astype("int64").to_numpy()
    daily_labels = daily["date"].dt.strftime("%d %b").tolist()

    cube = np.column_stack(
        [
            np.interp(daily_times, capture_times, flat_anomaly[:, pixel])
            for pixel in range(flat_anomaly.shape[1])
        ]
    ).reshape(len(daily), *capture_cube.shape[1:])
    cube += daily["ndci"].to_numpy()[:, None, None]

    first_day = daily.iloc[0]
    last_day = daily.iloc[-1]
    latest_frame = cube[-1]
    window_ndci_change = float(last_day["ndci"] - first_day["ndci"])
    biomass_change_window = float(last_day["biomass"] - first_day["biomass"])
    change_symbol, change_tone = trend(window_ndci_change)

    section(
        "Multispectral observations",
        f"{len(daily)} daily NDCI surfaces, 24 × 24 pixels, interpolated between {len(capture_labels)} satellite captures",
    )
    kpi_row(
        kpi(
            "First day",
            f'{first_day["ndci"]:.3f}',
            "mean NDCI",
            first_day["date"].strftime("%d %b %Y"),
            "flat",
            sub=f'Calibrated biomass {first_day["biomass"]:.3f} kg/m³',
        ),
        kpi(
            "Last day",
            f'{last_day["ndci"]:.3f}',
            "mean NDCI",
            last_day["date"].strftime("%d %b %Y"),
            "flat",
            sub=f'Calibrated biomass {last_day["biomass"]:.3f} kg/m³',
        ),
        kpi(
            "Change over window",
            f"{window_ndci_change:+.3f}",
            "NDCI",
            f"{change_symbol}{abs(biomass_change_window):.3f} kg/m³",
            change_tone,
            sub=f"Across {len(daily)} monitored days",
        ),
        kpi(
            "Pixel spread",
            f"{latest_frame.std():.3f}",
            "σ NDCI",
            f"range {latest_frame.min():.3f} - {latest_frame.max():.3f}",
            "flat",
            sub="Std dev across latest surface",
        ),
    )

    map_col, dist_col = st.columns([1.25, 1], gap="medium")

    with map_col, st.container(border=True):
        surface = px.imshow(
            cube,
            animation_frame=0,
            origin="lower",
            aspect="equal",
            color_continuous_scale=NDCI_SCALE,
            zmin=float(np.nanmin(cube)),
            zmax=float(np.nanmax(cube)),
            labels={"color": "NDCI"},
        )
        surface.update_traces(hovertemplate="x %{x}, y %{y}<br>NDCI %{z:.3f}<extra></extra>")
        style_fig(surface, height=470, title="NDCI surface (press play)", showlegend=False)
        surface.update_layout(hovermode="closest")
        surface.update_xaxes(showgrid=False, title=None)
        surface.update_yaxes(showgrid=False, title=None)
        surface.update_coloraxes(
            colorbar={
                "title": {"text": "NDCI", "font": {"color": MUTED, "size": 11}},
                "tickfont": {"color": MUTED, "size": 10},
                "outlinewidth": 0,
                "thickness": 12,
            }
        )

        menu = surface.layout.updatemenus[0]
        menu.bgcolor = "rgba(45,227,167,.12)"
        menu.bordercolor = "rgba(45,227,167,.42)"
        menu.borderwidth = 1
        menu.font = {"color": ACCENT, "size": 12, "family": "Inter, sans-serif"}
        play_args = menu.buttons[0].args[1]
        play_args["frame"] = {"duration": 260, "redraw": True}
        play_args["transition"] = {"duration": 240, "easing": "cubic-in-out"}

        slider = surface.layout.sliders[0]
        slider.currentvalue = {
            "prefix": "Date:  ",
            "font": {"color": TEXT, "size": 13, "family": "Inter, sans-serif"},
        }
        slider.font = {"color": MUTED, "size": 11}
        slider.bgcolor = "rgba(138,166,177,.18)"
        slider.activebgcolor = ACCENT
        slider.bordercolor = "rgba(0,0,0,0)"
        slider.transition = {"duration": 240, "easing": "cubic-in-out"}
        for step, label in zip(slider.steps, daily_labels):
            step.label = label

        st.plotly_chart(surface, use_container_width=True)

    with dist_col, st.container(border=True):
        bin_edges = np.linspace(float(cube.min()), float(cube.max()), 41)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        pixel_counts = np.stack(
            [np.histogram(frame.ravel(), bins=bin_edges)[0] for frame in cube]
        )

        density = go.Figure()
        density.add_trace(
            go.Heatmap(
                z=pixel_counts.T,
                x=daily_labels,
                y=bin_centres,
                colorscale=[
                    [0.00, "rgba(7,32,42,0)"],
                    [0.20, "#0d5a63"],
                    [0.60, "#12a382"],
                    [1.00, "#8bf0b4"],
                ],
                showscale=False,
                hovertemplate="%{x}<br>NDCI %{y:.3f}<br>%{z} pixels<extra></extra>",
                name="Pixel density",
                showlegend=False,
            )
        )
        density.add_trace(
            go.Scatter(
                x=daily_labels,
                y=daily["ndci"],
                mode="lines+markers",
                name="Daily mean NDCI",
                line={"color": "#eaffd4", "width": 2.2},
                marker={"size": 5, "color": "#eaffd4"},
                hovertemplate="%{x}<br>mean %{y:.3f}<extra></extra>",
            )
        )
        style_fig(density, height=470, title="Pixel NDCI distribution by day")
        density.update_layout(hovermode="closest")
        density.update_xaxes(title_text=None)
        density.update_yaxes(title_text="NDCI")
        st.plotly_chart(density, use_container_width=True)

    section("Calibration", f"R² {calibration_r2:.3f} · RMSE {calibration_rmse:.3f} kg/m³ · {len(calibration)} lab samples")
    with st.container(border=True):
        plot = calibration.copy()
        xline = np.linspace(results["ndci"].min(), results["ndci"].max(), 100)
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=xline,
                y=biomass_model.predict(xline.reshape(-1, 1)),
                mode="lines",
                name="Fitted calibration",
                line={"color": ACCENT_2, "width": 2.4},
                hovertemplate="NDCI %{x:.3f}<br>%{y:.3f} kg/m³<extra></extra>",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=plot["ndci"],
                y=plot["measured_dry_biomass_kg_m3"],
                mode="markers",
                name="Lab-measured samples",
                marker={
                    "size": 11,
                    "color": ACCENT,
                    "line": {"width": 1.5, "color": "rgba(5,14,20,.85)"},
                    "opacity": 0.9,
                },
                hovertemplate="NDCI %{x:.3f}<br>%{y:.3f} kg/m³<extra></extra>",
            )
        )
        style_fig(fig2, height=430, title="NDCI → dry biomass calibration")
        fig2.update_layout(hovermode="closest")
        fig2.update_xaxes(title_text="NDCI")
        fig2.update_yaxes(title_text="Measured dry biomass (kg/m³)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    gross = full_summary["gross_CO2_fixed_kg"]
    operational = full_summary["operational_emissions_kgCO2e"]
    net_total = full_summary["net_biological_balance_kgCO2e"]
    durable = full_summary["potential_durable_removal_kgCO2e"]

    section("Period carbon ledger", f"{window}, whole monitoring window")
    kpi_row(
        kpi("Gross CO₂ fixed", f"{gross:.1f}", "kg", "biomass gain", "up"),
        kpi("Operational emissions", f"-{operational:.1f}", "kg CO₂e", "electricity + transport", "down"),
        kpi(
            "Net biological balance",
            f"{net_total:+.1f}",
            "kg CO₂e",
            "▲ net removal" if net_total > 0 else "▼ net emitter",
            "up" if net_total > 0 else "down",
        ),
        kpi(
            "Potential durable removal",
            f"{durable:.1f}",
            "kg CO₂e",
            f"{fate.lower()}, permanence {permanence:.2f}",
            "flat",
        ),
    )

    left, right = st.columns([1, 1], gap="medium")

    with left:
        with st.container(border=True):
            waterfall = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative", "relative", "total"],
                    x=["Gross fixation", "Operational", "Net balance"],
                    y=[gross, -operational, 0],
                    text=[f"+{gross:.1f}", f"-{operational:.1f}", f"{net_total:+.1f}"],
                    textposition="outside",
                    textfont={"color": TEXT, "size": 12, "family": "JetBrains Mono, monospace"},
                    increasing={"marker": {"color": ACCENT}},
                    decreasing={"marker": {"color": DANGER}},
                    totals={"marker": {"color": ACCENT_2}},
                    connector={"line": {"color": "rgba(138,166,177,.35)", "dash": "dot"}},
                    hovertemplate="%{x}<br>%{y:.2f} kg CO₂e<extra></extra>",
                )
            )
            style_fig(waterfall, height=430, title="Net balance breakdown", showlegend=False)
            waterfall.update_layout(hovermode="closest")
            waterfall.update_yaxes(title_text="kg CO₂e")
            st.plotly_chart(waterfall, use_container_width=True)

    with right:
        with st.container(border=True):
            flux_fig = go.Figure()
            flux_fig.add_trace(
                go.Bar(
                    x=results["date"],
                    y=results["gross_CO2_fixed_kg"],
                    name="Gross fixation",
                    marker={"color": "rgba(45,227,167,.75)"},
                    hovertemplate="+%{y:.2f} kg<extra>Fixed</extra>",
                )
            )
            flux_fig.add_trace(
                go.Bar(
                    x=results["date"],
                    y=-results["operational_emissions_kgCO2e"],
                    name="Operational emissions",
                    marker={"color": "rgba(251,113,133,.70)"},
                    hovertemplate="%{y:.2f} kg<extra>Emitted</extra>",
                )
            )
            flux_fig.add_trace(
                go.Scatter(
                    x=results["date"],
                    y=results["cumulative_net_biological_balance_kgCO2e"],
                    name="Cumulative net",
                    mode="lines",
                    yaxis="y2",
                    line={"color": ACCENT_2, "width": 2.6, "shape": "spline"},
                    hovertemplate="%{y:+.1f} kg<extra>Cumulative</extra>",
                )
            )
            style_fig(flux_fig, height=430, title="Daily flux and cumulative balance")
            flux_fig.update_layout(
                barmode="relative",
                bargap=0.25,
                yaxis={"title": {"text": "Daily kg CO₂e", "font": {"color": MUTED, "size": 11}}},
                yaxis2={
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "zeroline": False,
                    "tickfont": {"color": ACCENT_2, "size": 11},
                    "title": {"text": "Cumulative kg CO₂e", "font": {"color": ACCENT_2, "size": 11}},
                },
            )
            st.plotly_chart(flux_fig, use_container_width=True)

    chip_row(
        chip(f"Carbon fraction {carbon_fraction:.2f}", "info"),
        chip(f"Uncertainty deduction {uncertainty:.0%}", "info"),
        chip(f"Biomass fate {fate}", "info"),
        chip(f"Permanence {permanence:.2f}", "info"),
        chip(f"Grid {grid_ef:.2f} kg/kWh", "mute"),
        chip(f"Transport {transport_ef:.2f} kg/km", "mute"),
    )

with tab3:
    record = {
        "project": "AlgaeMRV HackOut'26 prototype",
        "monitoring_start": results["date"].min().strftime("%Y-%m-%d"),
        "monitoring_end": results["date"].max().strftime("%Y-%m-%d"),
        "calibration_samples": int(len(calibration)),
        "calibration_r2": round(float(calibration_r2), 4),
        "gross_CO2_fixed_kg": round(full_summary["gross_CO2_fixed_kg"], 3),
        "operational_emissions_kgCO2e": round(full_summary["operational_emissions_kgCO2e"], 3),
        "net_biological_balance_kgCO2e": round(full_summary["net_biological_balance_kgCO2e"], 3),
        "potential_durable_removal_kgCO2e": round(full_summary["potential_durable_removal_kgCO2e"], 3),
        "data_quality_score": dq,
        "forecast_model": None if model_bundle is None else model_bundle.get("model_name"),
        "forecast_origin": origin_date.strftime("%Y-%m-%d"),
        "predicted_biomass_day7_g_l": None if forecast is None else round(float(forecast.iloc[-1]["predicted_biomass_g_l"]), 4),
        "status": "DEMONSTRATION / NOT A CERTIFIED CARBON CREDIT",
    }

    section("Verification record", "Inputs and results for this run")
    chip_row(
        chip("Demonstration only - not a certified carbon credit", "warn"),
        chip(f"Data quality {dq:.1f}/100", "ok" if dq >= 80 else "warn"),
        chip(f'Model {record["forecast_model"]}', "info"),
        chip(f'Origin {record["forecast_origin"]}', "mute"),
    )

    kpi_row(
        kpi("Monitoring window", f'{record["monitoring_start"]}', "", f'to {record["monitoring_end"]}', "flat"),
        kpi("Calibration samples", f'{record["calibration_samples"]}', "lab", f'R² {record["calibration_r2"]:.3f}', "flat"),
        kpi("Net balance", f'{record["net_biological_balance_kgCO2e"]:+.1f}', "kg CO₂e", "audited period total", "flat"),
        kpi("Durable removal", f'{record["potential_durable_removal_kgCO2e"]:.1f}', "kg CO₂e", f"permanence {permanence:.2f}", "flat"),
    )

    with st.expander("Full machine-readable MRV record", expanded=False):
        st.json(record)

    section("Export")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇  MRV record (JSON)",
            data=json.dumps(record, indent=2),
            file_name="algaemrv_mrv_record.json",
            mime="application/json",
            width="stretch",
        )
    with d2:
        st.download_button(
            "⬇  Monitoring table (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="algaemrv_calculated_results.csv",
            mime="text/csv",
            width="stretch",
        )
    with d3:
        if forecast is not None:
            st.download_button(
                "⬇  7-day forecast (CSV)",
                data=forecast.to_csv(index=False).encode("utf-8"),
                file_name="algaemrv_7_day_forecast.csv",
                mime="text/csv",
                width="stretch",
            )
