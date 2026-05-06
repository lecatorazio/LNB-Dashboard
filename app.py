import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
# =========================
# CONFIG
# =========================
st.set_page_config(page_title="LNB Dashboard", layout="wide")
 
# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_players():
    try:
        conn = sqlite3.connect("basket.db")
        df = pd.read_sql_query("SELECT * FROM boxscores", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erreur chargement boxscores : {e}")
        st.stop()
 
@st.cache_data
def load_teamstats():
    try:
        return pd.read_csv("lnb_all_teamstats.csv")
    except:
        return pd.DataFrame()
 
df_raw    = load_players()
ts_raw    = load_teamstats()
 
# =========================
# HELPERS
# =========================
def parse_minutes(m):
    try:
        if pd.isna(m): return 0.0
        t = str(m)
        if "PT" in t:
            t = t.replace("PT", "")
            minutes, seconds = 0.0, 0.0
            if "M" in t:
                parts = t.split("M")
                minutes = float(parts[0]) if parts[0] else 0.0
                if len(parts) > 1 and "S" in parts[1]:
                    seconds = float(parts[1].replace("S", ""))
            elif "S" in t:
                seconds = float(t.replace("S", ""))
            return minutes + seconds / 60
        if ":" in t:
            m_, s_ = t.split(":")
            return float(m_) + float(s_) / 60
        return float(t)
    except:
        return 0.0
 
def safe_div(a, b, default=0.0):
    result = a / b

    # Cas scalar (float)
    if np.isscalar(result):
        if np.isinf(result) or np.isnan(result):
            return default
        return result

    # Cas Series / DataFrame
    return (
        result.replace([np.inf, -np.inf], np.nan)
               .fillna(default)
    )
 
# =========================
# PRÉPARATION JOUEURS
# =========================
df = df_raw.copy()
df["MIN"]  = df["stat_minutes"].apply(parse_minutes)
df["FGM"]  = df["stat_pointsTwoMade"].fillna(0)   + df["stat_pointsThreeMade"].fillna(0)
df["FGA"]  = df["stat_pointsTwoAttempted"].fillna(0) + df["stat_pointsThreeAttempted"].fillna(0)
df["PTS"]  = df["stat_points"].fillna(0)
df["AST"]  = df["stat_assists"].fillna(0)
df["REB"]  = df["stat_rebounds"].fillna(0)
df["OREB"] = df["stat_reboundsOffensive"].fillna(0)
df["DREB"] = df["stat_reboundsDefensive"].fillna(0)
df["STL"]  = df["stat_steals"].fillna(0)
df["BLK"]  = df["stat_blocks"].fillna(0)
df["TOV"]  = df["stat_turnovers"].fillna(0)
df["FTA"]  = df["stat_freeThrowsAttempted"].fillna(0)
df["FTM"]  = df["stat_freeThrowsMade"].fillna(0)
df["3PA"]  = df["stat_pointsThreeAttempted"].fillna(0)
df["3PM"]  = df["stat_pointsThreeMade"].fillna(0)
df["2PA"]  = df["stat_pointsTwoAttempted"].fillna(0)
df["2PM"]  = df["stat_pointsTwoMade"].fillna(0)
df["FOUL"] = df["stat_foulsTotal"].fillna(0)
df["PM"]   = df["stat_plusMinus"].fillna(0)
 
# =========================
# HOME / AWAY
# =========================
if "side" in df.columns:
    fixture_teams = (
        df[df["side"].isin(["home", "away"])]
        .groupby(["fixture_id", "side"])["team_name"]
        .first().unstack()
        .rename(columns={"home": "home_team", "away": "away_team"})
        .reset_index()
    )
else:
    fixture_teams = (
        df.groupby("fixture_id")["team_name"]
        .apply(lambda x: sorted(x.dropna().unique())).reset_index()
    )
    fixture_teams["home_team"] = fixture_teams["team_name"].apply(lambda x: x[0] if len(x) > 0 else "—")
    fixture_teams["away_team"] = fixture_teams["team_name"].apply(lambda x: x[1] if len(x) > 1 else "—")
    fixture_teams = fixture_teams[["fixture_id", "home_team", "away_team"]]
 
df = df.merge(fixture_teams, on="fixture_id", how="left")
 
# =========================
# PRÉPARATION STATS ÉQUIPE (depuis lnb_all_teamstats.csv)
# =========================
ts = ts_raw[ts_raw["team_name"] != "UNKNOWN"].copy()
 
ts["tm_FGM"]  = ts["team_pointsTwoMade"].fillna(0)   + ts["team_pointsThreeMade"].fillna(0)
ts["tm_FGA"]  = ts["team_pointsTwoAttempted"].fillna(0) + ts["team_pointsThreeAttempted"].fillna(0)
ts["tm_FTA"]  = ts["team_freeThrowsAttempted"].fillna(0)
ts["tm_FTM"]  = ts["team_freeThrowsMade"].fillna(0)
ts["tm_OREB"] = ts["team_reboundsOffensive"].fillna(0)
ts["tm_DREB"] = ts["team_reboundsDefensive"].fillna(0)
ts["tm_REB"]  = ts["team_rebounds"].fillna(0)
ts["tm_TOV"]  = ts["team_turnovers"].fillna(0)
ts["tm_PTS"]  = ts["team_points"].fillna(0)
ts["tm_AST"]  = ts["team_assists"].fillna(0)
ts["tm_STL"]  = ts["team_steals"].fillna(0)
ts["tm_BLK"]  = ts["team_blocks"].fillna(0)
ts["tm_3PA"]  = ts["team_pointsThreeAttempted"].fillna(0)
ts["tm_3PM"]  = ts["team_pointsThreeMade"].fillna(0)
 
# Possessions équipe (Dean Oliver)
ts["tm_POSS"] = ts["tm_FGA"] - ts["tm_OREB"] + ts["tm_TOV"] + 0.44 * ts["tm_FTA"]
 
# Minutes équipe (somme 5 joueurs × durée match ≈ 200 min)
ts["tm_MIN"]  = ts["team_minutes"].apply(parse_minutes) if "team_minutes" in ts.columns else 200.0
 
# Stats adversaire : même fixture_id, side opposé
opp = ts[["fixture_id", "side", "team_name",
          "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM",
          "tm_OREB", "tm_DREB", "tm_REB", "tm_TOV",
          "tm_PTS", "tm_POSS", "tm_3PA", "tm_AST", "tm_MIN"]].copy()
 
opp["opp_side"] = opp["side"].map({"home": "away", "away": "home"})
opp = opp.rename(columns={c: c.replace("tm_", "opp_") for c in opp.columns if c.startswith("tm_")})
opp = opp.rename(columns={"team_name": "opp_team_name"})
opp["side"] = opp["opp_side"]
opp = opp.drop(columns=["opp_side"])
 
# Merge tm + opp sur fixture+side
ts_full = ts.merge(
    opp[["fixture_id", "side", "opp_team_name",
         "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM",
         "opp_OREB", "opp_DREB", "opp_REB", "opp_TOV",
         "opp_PTS", "opp_POSS", "opp_3PA", "opp_AST", "opp_MIN"]],
    on=["fixture_id", "side"], how="left"
)
 
# =========================
# MERGE JOUEURS ← STATS ÉQUIPE
# =========================
# On merge via fixture_id + team_name
tm_cols = ["fixture_id", "team_name",
           "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM",
           "tm_OREB", "tm_DREB", "tm_REB", "tm_TOV",
           "tm_PTS", "tm_POSS", "tm_3PA", "tm_AST", "tm_MIN",
           "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM",
           "opp_OREB", "opp_DREB", "opp_REB", "opp_TOV",
           "opp_PTS", "opp_POSS", "opp_3PA", "opp_AST", "opp_MIN",
           "opp_team_name"]
 
df = df.merge(ts_full[tm_cols], on=["fixture_id", "team_name"], how="left")
 
# Supprimer doublons éventuels
df = df.drop_duplicates(
    subset=["fixture_id", "player_id"] if "player_id" in df.columns else ["fixture_id", "player_name"]
)
 
# =========================
# STATS AVANCÉES — FORMULES DEAN OLIVER
# =========================
 
# --- Tirs ---
df["eFG%"]   = safe_div(df["FGM"] + 0.5 * df["3PM"], df["FGA"]) * 100
df["TS%"]    = safe_div(df["PTS"], 2 * (df["FGA"] + 0.44 * df["FTA"])) * 100
df["FTrate"] = safe_div(df["FTA"], df["FGA"]) * 100
df["3Prate"] = safe_div(df["3PA"], df["FGA"]) * 100
 
# --- Part du joueur sur les possessions équipe ---
# qAST : proportion des tirs de l'équipe assistés par le joueur
df["qAST"] = (
    safe_div(df["MIN"], df["tm_MIN"] / 5) *
    safe_div(df["tm_AST"], df["tm_FGM"])
).clip(0, 1)
 
# FG_part
df["FG_part"] = df["FGM"] * (1 - 0.5 * safe_div(df["PTS"] - df["FTM"], 2 * df["FGM"])) * df["qAST"]
 
# AST_part
df["AST_part"] = 0.5 * (
    safe_div(df["tm_PTS"] - df["tm_FTM"], 2 * df["tm_FGM"])
) * df["AST"]
 
# FT_part
df["FT_part"] = (1 - (1 - safe_div(df["FTM"], df["FTA"])) ** 2) * 0.4 * df["FTA"]
 
# OREB_part
df["OREB_part"] = df["OREB"] * safe_div(
    df["tm_PTS"],
    df["tm_FGA"] - df["tm_OREB"] + df["tm_TOV"] + 0.44 * df["tm_FTA"]
)
 
# PProd (production de points Dean Oliver)
df["PProd"] = (df["FG_part"] + df["AST_part"]) * 2 + df["FTM"] + df["OREB_part"]
 
# --- Possessions individuelles ---
df["ScPoss"] = np.nan_to_num(
    safe_div(df["PProd"], df["PTS"].replace(0, np.nan)),
    nan=0.0
) * (
    df["FGA"]
    - df["OREB"] * safe_div(df["tm_OREB"], df["tm_OREB"] + df["opp_DREB"])
    + 0.44 * df["FTA"]
    + df["TOV"]
)
df["TotPoss"] = df["FGA"] - df["OREB"] * safe_div(df["tm_OREB"], df["tm_OREB"] + df["opp_DREB"]) + \
                0.44 * df["FTA"] + df["TOV"]
df["Floor%"]  = safe_div(df["ScPoss"], df["TotPoss"]) * 100
df["POSS"]    = df["TotPoss"]
df["PPP"]     = safe_div(df["PTS"], df["POSS"])
 
# --- ORtg (Dean Oliver, per 100 poss) ---
df["ORtg"] = safe_div(df["PProd"] * 100, df["TotPoss"])
 
# --- DRtg (Dean Oliver) ---
# Stops individuels
df["DORpct"]   = safe_div(df["opp_OREB"], df["opp_OREB"] + df["tm_DREB"])
df["DFGpct"]   = safe_div(df["opp_FGM"], df["opp_FGA"])
df["FMwt"]     = safe_div(df["DFGpct"] * (1 - df["DORpct"]),
                           df["DFGpct"] * (1 - df["DORpct"]) + (1 - df["DFGpct"]) * df["DORpct"])
df["Stops1"]   = (df["STL"] +
                  df["BLK"] * df["FMwt"] * (1 - 1.07 * df["DORpct"]) +
                  df["DREB"] * (1 - df["FMwt"]))
df["Stops2"]   = (df["FOUL"] / df["tm_FOUL"].replace(0, np.nan)).fillna(0) * \
                  df["opp_FTM"] * (1 - (1 - safe_div(df["opp_FTM"], df["opp_FTA"])) ** 2) \
                  if "tm_FOUL" in df.columns else 0.0
df["Stops"]    = df["Stops1"]
df["Stop%"]    = safe_div(df["Stops"] * df["tm_POSS"],
                           df["MIN"] / (df["tm_MIN"] / 5) * df["opp_POSS"]) * 100
 
# DPts : points accordés par le joueur
df["DPts"]     = df["opp_PTS"] * safe_div(df["MIN"], df["tm_MIN"] / 5) * \
                  safe_div(df["Stops"], df["opp_POSS"])
df["DRtg"]     = safe_div(df["DPts"] * 100, df["Stops"].replace(0, np.nan)).fillna(
    safe_div(df["opp_PTS"] * 100, df["opp_POSS"])
)
df["NETRtg"]   = df["ORtg"] - df["DRtg"]
 
# --- Usage ---
df["USG%"] = safe_div(
    df["FGA"] + 0.44 * df["FTA"] + df["TOV"],
    safe_div(df["MIN"], df["tm_MIN"] / 5) * (df["tm_FGA"] + 0.44 * df["tm_FTA"] + df["tm_TOV"])
) * 100
 
# --- Playmaking ---
df["AST%"]    = safe_div(
    df["AST"],
    safe_div(df["MIN"], df["tm_MIN"] / 5) * df["tm_FGM"] - df["FGM"]
) * 100
df["AST/BP"]  = safe_div(df["AST"], df["TOV"])
df["Ratio_PM"]= safe_div(df["AST"], df["AST"] + df["TOV"])
df["qAST_pct"]= df["qAST"] * 100
 
# --- Rebonds ---
df["OREB%"] = safe_div(
    df["OREB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_OREB"] + df["opp_DREB"])
) * 100
df["DREB%"] = safe_div(
    df["DREB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_DREB"] + df["opp_OREB"])
) * 100
df["TRB%"]  = safe_div(
    df["REB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_REB"] + df["opp_REB"])
) * 100
 
# --- Défense ---
df["STL%"]     = safe_div(df["STL"] * (df["tm_MIN"] / 5), df["MIN"] * df["opp_POSS"]) * 100
df["BLK%"]     = safe_div(
    df["BLK"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["opp_FGA"] - df["opp_3PA"])
) * 100
df["STOCKS"]   = df["STL"] + df["BLK"]
df["BLK_FOUL"] = safe_div(df["BLK"], df["FOUL"])
df["TOV%"]     = safe_div(df["TOV"], df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) * 100
 
# --- Efficacité globale ---
# Gamescore (Hollinger)
df["Gamescore"] = (df["PTS"] + 0.4*df["FGM"] - 0.7*df["FGA"] -
                   0.4*(df["FTA"]-df["FTM"]) + 0.7*df["OREB"] +
                   0.3*df["DREB"] + df["STL"] + 0.7*df["AST"] +
                   0.7*df["BLK"] - 0.4*df["FOUL"] - df["TOV"])
 
# PIE (NBA Player Impact Estimate)
df["PIE_num"]    = (df["PTS"] + df["FGM"] + df["FTM"] - df["FGA"] - df["FTA"] +
                    df["DREB"] + 0.5*df["OREB"] + df["AST"] + df["STL"] +
                    0.5*df["BLK"] - df["FOUL"] - df["TOV"])
df["PIE_den"]    = (df["tm_PTS"] + df["tm_FGM"] + df["tm_FTM"] +
                    df["opp_PTS"] + df["opp_FGM"] + df["opp_FTM"])
df["PIE"]        = safe_div(df["PIE_num"], df["PIE_den"]) * 100
df["PIE_MIN"]    = safe_div(df["PIE"], df["MIN"])
 
# PER simplifié (Hollinger)
df["PER"] = safe_div(1, df["MIN"]) * (
    df["3PM"] + (2/3)*df["AST"] + 2*df["FGM"] - 0.5*df["FGA"] +
    (1/2)*df["FTM"] - (3/4)*df["FTA"] + df["DREB"] + (3/4)*df["OREB"] +
    df["BLK"] + df["STL"] - df["TOV"] - df["FOUL"]
)
 
# =========================
# AGRÉGATION SAISON JOUEURS
# =========================
agg_map = {
    "fixture_id": "count",
    "MIN": "sum", "PTS": "sum", "AST": "sum", "REB": "sum",
    "OREB": "sum", "DREB": "sum", "STL": "sum", "BLK": "sum",
    "TOV": "sum", "FGM": "sum", "FGA": "sum", "FTM": "sum",
    "FTA": "sum", "3PM": "sum", "3PA": "sum", "FOUL": "sum",
    "PM": "sum", "STOCKS": "sum", "PProd": "sum",
    "ScPoss": "sum", "TotPoss": "sum",
    "eFG%": "mean", "TS%": "mean", "FTrate": "mean", "3Prate": "mean",
    "USG%": "mean", "AST%": "mean", "AST/BP": "mean", "Ratio_PM": "mean",
    "qAST_pct": "mean",
    "OREB%": "mean", "DREB%": "mean", "TRB%": "mean",
    "STL%": "mean", "BLK%": "mean", "BLK_FOUL": "mean", "TOV%": "mean",
    "Floor%": "mean", "PPP": "mean",
    "ORtg": "mean", "DRtg": "mean", "NETRtg": "mean",
    "Gamescore": "mean", "PIE": "mean", "PIE_MIN": "mean", "PER": "mean",
    "Stop%": "mean", "Stops": "sum",
    "FG_part": "mean", "AST_part": "mean", "FT_part": "mean", "OREB_part": "sum",
}
agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
 
player_season = df.groupby(["player_name", "team_name"]).agg(agg_map).reset_index()
player_season = player_season.rename(columns={"fixture_id": "GP"})
player_season["MIN_PG"] = safe_div(player_season["MIN"], player_season["GP"])
player_season["POSS"]   = player_season["TotPoss"]
 
# Per 36 min
for col in ["PTS", "AST", "REB", "STL", "BLK", "TOV", "STOCKS"]:
    if col in player_season.columns:
        player_season[f"{col}_P36"] = safe_div(player_season[col], player_season["MIN"]) * 36
 
MIN_MINUTES = 50
 
# =========================
# AGRÉGATION SAISON ÉQUIPES (depuis ts_full)
# =========================
team_agg = ts_full.groupby("team_name").agg(
    GP=("fixture_id", "nunique"),
    tm_PTS=("tm_PTS", "sum"), tm_FGM=("tm_FGM", "sum"), tm_FGA=("tm_FGA", "sum"),
    tm_FTA=("tm_FTA", "sum"), tm_FTM=("tm_FTM", "sum"),
    tm_OREB=("tm_OREB", "sum"), tm_DREB=("tm_DREB", "sum"),
    tm_REB=("tm_REB", "sum"), tm_TOV=("tm_TOV", "sum"),
    tm_AST=("tm_AST", "sum"), tm_STL=("tm_STL", "sum"),
    tm_BLK=("tm_BLK", "sum"), tm_3PA=("tm_3PA", "sum"),
    tm_POSS=("tm_POSS", "sum"),
    opp_PTS=("opp_PTS", "sum"), opp_POSS=("opp_POSS", "sum"),
    opp_OREB=("opp_OREB", "sum"), opp_TOV=("opp_TOV", "sum"),
).reset_index()
 
team_agg["eFG%"]   = safe_div(team_agg["tm_FGM"], team_agg["tm_FGA"]) * 100
team_agg["TS%"]    = safe_div(team_agg["tm_PTS"], 2*(team_agg["tm_FGA"] + 0.44*team_agg["tm_FTA"])) * 100
team_agg["TOV%"]   = safe_div(team_agg["tm_TOV"], team_agg["tm_FGA"] + 0.44*team_agg["tm_FTA"] + team_agg["tm_TOV"]) * 100
team_agg["OREB%"]  = safe_div(team_agg["tm_OREB"], team_agg["tm_OREB"] + team_agg["opp_OREB"]) * 100
team_agg["ORtg"]   = safe_div(team_agg["tm_PTS"]  * 100, team_agg["tm_POSS"])
team_agg["DRtg"]   = safe_div(team_agg["opp_PTS"] * 100, team_agg["opp_POSS"])
team_agg["NETRtg"] = team_agg["ORtg"] - team_agg["DRtg"]
team_agg["PPP"]    = safe_div(team_agg["tm_PTS"], team_agg["tm_POSS"])
team_agg["PTS_PG"] = safe_div(team_agg["tm_PTS"],  team_agg["GP"])
team_agg["AST_PG"] = safe_div(team_agg["tm_AST"],  team_agg["GP"])
team_agg["REB_PG"] = safe_div(team_agg["tm_REB"],  team_agg["GP"])
team_agg["TOV_PG"] = safe_div(team_agg["tm_TOV"],  team_agg["GP"])
team_agg["POSS_PG"]= safe_div(team_agg["tm_POSS"], team_agg["GP"])
 
team_season = team_agg.rename(columns={
    "tm_PTS": "PTS", "tm_FGM": "FGM", "tm_FGA": "FGA",
    "tm_FTA": "FTA", "tm_FTM": "FTM", "tm_AST": "AST",
    "tm_REB": "REB", "tm_TOV": "TOV", "tm_STL": "STL",
    "tm_BLK": "BLK", "tm_POSS": "POSS",
})
 
# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🏀 LNB Dashboard")
    st.markdown("---")
    st.subheader("Filtres")
 
    teams_list = ["Toutes"] + sorted(df["team_name"].dropna().unique().tolist())
    team_filter = st.selectbox("Équipe", teams_list)
 
    pos_list = ["Toutes"] + sorted(df["position"].dropna().unique().tolist()) \
               if "position" in df.columns else ["Toutes"]
    pos_filter = st.selectbox("Position", pos_list)
 
    st.markdown("---")
    st.subheader("Top joueurs")
    raw_stat_cols = [c for c in df.select_dtypes(include="number").columns
                     if c not in ["player_id", "fixture_id", "team_id"]]
    default_stat = next((c for c in raw_stat_cols if "point" in c.lower()),
                        raw_stat_cols[0] if raw_stat_cols else None)
    if raw_stat_cols:
        stat_col = st.selectbox("Stat", raw_stat_cols,
                                index=raw_stat_cols.index(default_stat) if default_stat in raw_stat_cols else 0)
        top_n = st.slider("Nombre de joueurs", 3, 20, 10)
    else:
        stat_col, top_n = None, 10
 
    st.markdown("---")
    st.subheader("Stats avancées")
    min_min = st.slider("Minutes min. (saison)", 0, 500, MIN_MINUTES, step=10)
 
# =========================
# FILTRAGE
# =========================
df_filtered = df.copy()
if team_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["team_name"] == team_filter]
if pos_filter != "Toutes" and "position" in df.columns:
    df_filtered = df_filtered[df_filtered["position"] == pos_filter]
 
adv_filtered = player_season[player_season["MIN"] >= min_min].copy()
if team_filter != "Toutes":
    adv_filtered = adv_filtered[adv_filtered["team_name"] == team_filter]
 
# =========================
# ONGLETS PRINCIPAUX
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Boxscores",
    "📊 Stats avancées joueurs",
    "🏟️ Stats avancées équipes",
    "⚔️ Comparaison joueurs",
])
 
# ══════════════════════════════════════════
# ONGLET 1 — BOXSCORES
# ══════════════════════════════════════════
with tab1:
    st.title("🏀 Dashboard Boxscores")
 
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Joueurs uniques", df_filtered["player_name"].nunique())
    k2.metric("Matchs", df_filtered["fixture_id"].nunique())
    k3.metric("Lignes filtrées", len(df_filtered))
    k4.metric("Lignes totales", len(df))
 
    st.markdown("---")
 
    if stat_col and "player_name" in df_filtered.columns:
        st.subheader(f"🔥 Top {top_n} joueurs — {stat_col}")
        top = (df_filtered.groupby("player_name")[stat_col]
               .sum().sort_values(ascending=False).head(top_n).reset_index())
        top.columns = ["Joueur", stat_col]
        fig = px.bar(top, x=stat_col, y="Joueur", orientation="h",
                     color=stat_col, color_continuous_scale="Blues", text_auto=".0f")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), height=400)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("---")
    st.subheader("📋 Données filtrées")
 
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"{len(df_filtered)} lignes · {df_filtered.shape[1]} colonnes")
    with c2:
        st.download_button("⬇️ CSV", df_filtered.to_csv(index=False).encode("utf-8"),
                           "boxscores_filtres.csv", "text/csv")
 
    cols_to_hide = ["fixture_id", "player_id", "team_id", "side",
                    "opp_team_name", "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM",
                    "tm_OREB", "tm_DREB", "tm_REB", "tm_TOV", "tm_PTS",
                    "tm_POSS", "tm_3PA", "tm_AST", "tm_MIN",
                    "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM",
                    "opp_OREB", "opp_DREB", "opp_REB", "opp_TOV",
                    "opp_PTS", "opp_POSS", "opp_3PA", "opp_AST", "opp_MIN",
                    "DORpct", "DFGpct", "FMwt", "Stops1", "Stops2",
                    "PIE_num", "PIE_den", "PProd_FG", "PProd_AST", "PProd_OREB",
                    "ScPoss", "FGxPoss", "FTxPoss", "DPts"]
    display_cols = [c for c in df_filtered.columns if c not in cols_to_hide]
    priority = ["player_name", "team_name", "home_team", "away_team", "position"]
    front = [c for c in priority if c in display_cols]
    rest  = [c for c in display_cols if c not in front]
    display_cols = front + rest
 
    col_rename = {"bib": "numéro"}
    st.dataframe(df_filtered[display_cols].rename(columns=col_rename),
                 use_container_width=True, height=500)
 
    with st.expander("🧾 Aperçu données brutes (5 premières lignes)"):
        st.dataframe(df[display_cols].rename(columns=col_rename).head(), use_container_width=True)
        st.caption(f"Dataset complet : {df.shape[0]} lignes · {df.shape[1]} colonnes")
 
 
# ══════════════════════════════════════════
# ONGLET 2 — STATS AVANCÉES JOUEURS
# ══════════════════════════════════════════
with tab2:
    st.title("📊 Stats avancées — Joueurs")
    st.caption(f"Saison complète · Minimum {min_min} min · {len(adv_filtered)} joueurs · Formules Dean Oliver")
 
    def round_df(d, dec=2):
        return d.round(dec)
 
    base = ["player_name", "team_name", "GP", "MIN", "MIN_PG"]
    s1, s2, s3, s4, s5, s6 = st.tabs([
        "🎯 Tirs & scoring", "🎩 Playmaking",
        "💪 Rebonds", "🛡️ Défense",
        "⚡ Efficacité", "🔁 Possessions"
    ])
 
    with s1:
        cols = base + ["PTS", "eFG%", "TS%", "FTrate", "3Prate",
                       "FG_part", "FT_part", "PProd", "Gamescore"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("TS%", ascending=False),
                     use_container_width=True, height=500)
        st.caption("eFG% / TS% = efficacité tir · PProd = points produits (Dean Oliver)")
 
    with s2:
        cols = base + ["AST", "AST%", "AST/BP", "qAST_pct", "Ratio_PM", "USG%", "TOV", "TOV%"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("AST%", ascending=False),
                     use_container_width=True, height=500)
        st.caption("AST% = % tirs équipe assistés · USG% = % possessions utilisées · qAST = qualité des passes")
 
    with s3:
        cols = base + ["REB", "OREB", "DREB", "OREB%", "DREB%", "TRB%",
                       "REB_P36", "OREB_P36", "DREB_P36"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("TRB%", ascending=False),
                     use_container_width=True, height=500)
        st.caption("OREB%/DREB%/TRB% = % rebonds disponibles captés · _P36 = per 36 min")
 
    with s4:
        cols = base + ["STL", "BLK", "STOCKS", "STL%", "BLK%",
                       "BLK_FOUL", "Stops", "Stop%", "STL_P36", "BLK_P36"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("Stop%", ascending=False),
                     use_container_width=True, height=500)
        st.caption("STOCKS = STL+BLK · Stop% = % possessions adverses stoppées (Dean Oliver)")
 
    with s5:
        cols = base + ["PER", "PIE", "PIE_MIN", "Gamescore", "ORtg", "DRtg", "NETRtg", "PM"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("PER", ascending=False),
                     use_container_width=True, height=400)
 
        st.markdown("---")
        if "ORtg" in adv_filtered.columns and "DRtg" in adv_filtered.columns:
            st.subheader("ORtg vs DRtg")
            fig2 = px.scatter(
                adv_filtered, x="ORtg", y="DRtg", text="player_name",
                color="team_name", size="MIN",
                hover_data=["GP", "PER", "NETRtg"],
                labels={"ORtg": "Rating offensif (per 100 poss)",
                        "DRtg": "Rating défensif (per 100 poss)"},
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        st.caption("PER = Player Efficiency Rating · PIE = Player Impact Estimate · ORtg/DRtg per 100 possessions (Dean Oliver)")
 
    with s6:
        cols = base + ["POSS", "PPP", "TotPoss", "ScPoss", "Floor%", "USG%", "TOV%"]
        av = [c for c in cols if c in adv_filtered.columns]
        st.dataframe(round_df(adv_filtered[av]).sort_values("Floor%", ascending=False),
                     use_container_width=True, height=500)
        st.caption("PPP = points/possession · Floor% = % possessions productrices · TotPoss = total possessions utilisées")
 
    st.markdown("---")
    st.download_button("⬇️ Télécharger stats avancées joueurs",
                       adv_filtered.round(3).to_csv(index=False).encode("utf-8"),
                       "stats_avancees_joueurs.csv", "text/csv")
 
 
# ══════════════════════════════════════════
# ONGLET 3 — STATS AVANCÉES ÉQUIPES
# ══════════════════════════════════════════
with tab3:
    st.title("🏟️ Stats avancées — Équipes")
    st.caption(f"Saison complète · {len(team_season)} équipes · Formules Dean Oliver")
 
    if len(team_season) > 0:
        best_net = team_season.loc[team_season["NETRtg"].idxmax()]
        best_off = team_season.loc[team_season["ORtg"].idxmax()]
        best_def = team_season.loc[team_season["DRtg"].idxmin()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Meilleur NETRtg", best_net["team_name"], f"{best_net['NETRtg']:.1f}")
        k2.metric("Meilleure attaque", best_off["team_name"], f"{best_off['ORtg']:.1f} ORtg")
        k3.metric("Meilleure défense", best_def["team_name"], f"{best_def['DRtg']:.1f} DRtg")
 
    st.markdown("---")
    te1, te2 = st.tabs(["📈 Ratings & efficacité", "📦 Stats générales"])
 
    with te1:
        cols = ["team_name", "GP", "ORtg", "DRtg", "NETRtg", "PPP",
                "eFG%", "TS%", "TOV%", "OREB%", "POSS_PG"]
        av = [c for c in cols if c in team_season.columns]
        st.dataframe(team_season[av].round(2).sort_values("NETRtg", ascending=False),
                     use_container_width=True, height=400)
        st.markdown("---")
        st.subheader("ORtg vs DRtg — équipes")
        fig3 = px.scatter(
            team_season, x="ORtg", y="DRtg", text="team_name",
            color="NETRtg", color_continuous_scale="RdYlGn", size="GP",
            labels={"ORtg": "Rating offensif (per 100 poss)",
                    "DRtg": "Rating défensif (per 100 poss)"},
        )
        fig3.update_traces(textposition="top center", textfont_size=10)
        fig3.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Vert = meilleur NETRtg · ORtg/DRtg per 100 possessions (Dean Oliver)")
 
    with te2:
        cols = ["team_name", "GP", "PTS_PG", "AST_PG", "REB_PG", "TOV_PG",
                "PTS", "AST", "REB", "TOV", "STL", "BLK", "FGM", "FGA", "POSS"]
        av = [c for c in cols if c in team_season.columns]
        st.dataframe(team_season[av].round(2).sort_values("PTS_PG", ascending=False),
                     use_container_width=True, height=400)
 
    st.markdown("---")
    st.download_button("⬇️ Télécharger stats avancées équipes",
                       team_season.round(3).to_csv(index=False).encode("utf-8"),
                       "stats_avancees_equipes.csv", "text/csv")
 
 
# ══════════════════════════════════════════
# ONGLET 4 — COMPARAISON JOUEURS
# ══════════════════════════════════════════
with tab4:
    st.title("⚔️ Comparaison joueurs")
 
    all_players = sorted(player_season["player_name"].unique().tolist())
 
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Joueur 1", all_players, index=0)
    with c2:
        p2 = st.selectbox("Joueur 2", all_players,
                          index=1 if len(all_players) > 1 else 0)
 
    if p1 == p2:
        st.warning("Sélectionnez deux joueurs différents.")
    else:
        d1 = player_season[player_season["player_name"] == p1].iloc[0]
        d2 = player_season[player_season["player_name"] == p2].iloc[0]
 
        # KPIs côte à côte
        st.markdown("---")
        stats_kpi = ["GP", "MIN_PG", "PTS", "AST", "REB", "STL", "BLK",
                     "ORtg", "DRtg", "NETRtg", "PER", "PIE", "TS%", "USG%"]
        stats_kpi = [s for s in stats_kpi if s in player_season.columns]
 
        cols_kpi = st.columns(len(stats_kpi))
        for i, stat in enumerate(stats_kpi):
            v1 = d1[stat]
            v2 = d2[stat]
            # Pour DRtg, moins = mieux
            better = p1 if (v1 > v2 if stat != "DRtg" else v1 < v2) else p2
            delta = f"{v1 - v2:+.1f}"
            cols_kpi[i].metric(
                stat,
                f"{v1:.1f}",
                delta=delta,
                delta_color="normal" if stat != "DRtg" else "inverse"
            )
 
        st.markdown("---")
 
        # Radar chart
        radar_stats = ["ORtg", "DRtg", "TS%", "USG%", "TRB%", "AST%",
                       "STL%", "BLK%", "Floor%", "PER"]
        radar_stats = [s for s in radar_stats if s in player_season.columns]
 
        # Normalisation 0-100 par rapport au max de la saison
        fig_radar = go.Figure()
        for player, data, color in [(p1, d1, "#185FA5"), (p2, d2, "#D85A30")]:
            vals = []
            for s in radar_stats:
                col_max = player_season[s].replace([np.inf, -np.inf], np.nan).max()
                col_min = player_season[s].replace([np.inf, -np.inf], np.nan).min()
                norm = safe_div(data[s] - col_min, col_max - col_min) * 100
                # Pour DRtg, inverser (moins = mieux)
                if s == "DRtg":
                    norm = 100 - norm
                vals.append(float(norm))
 
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_stats + [radar_stats[0]],
                fill="toself", name=player,
                line_color=color, opacity=0.7,
            ))
 
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Valeurs normalisées 0-100 par rapport au reste de la ligue · DRtg inversé (plus haut = meilleure défense)")
 
        st.markdown("---")
 
        # Tableau comparatif complet
        st.subheader("Tableau comparatif complet")
        all_stats = [c for c in player_season.columns
                     if c not in ["player_name", "team_name"] and
                     player_season[c].dtype in [np.float64, np.int64]]
 
        compare_df = pd.DataFrame({
            "Stat": all_stats,
            p1: [round(d1[s], 2) if s in d1 else "—" for s in all_stats],
            p2: [round(d2[s], 2) if s in d2 else "—" for s in all_stats],
        })
        st.dataframe(compare_df, use_container_width=True, height=500)
 
        st.download_button(
            "⬇️ Télécharger comparaison",
            compare_df.to_csv(index=False).encode("utf-8"),
            f"comparaison_{p1}_vs_{p2}.csv", "text/csv"
        )