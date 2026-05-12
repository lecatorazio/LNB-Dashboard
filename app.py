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

df_raw = load_players()
ts_raw = load_teamstats()

# =========================
# COMPETITION BUILDING
# =========================
fixture_comp = (
    df_raw.groupby("fixture_id")["team_name"]
    .apply(lambda x: list(x.dropna().unique()))
    .reset_index(name="teams")
)

elite2_teams = set([
    "Blois", "Châlons-Reims", "Roanne", "Rouen", "Vichy", "Quimper", "Antibes", "Orléans",
    "Pau-Lacq-Orthez", "Evreux", "Denain", "Challans", "Saint-Chamond", "Caen", "Poitiers",
    "Gries-Souffel", "Hyères-Toulon", "Nantes", "La Rochelle", "Aix-Maurienne"
])

def classify_competition(teams):
    if len(set(teams).intersection(elite2_teams)) > 0:
        return "Elite 2"
    return "Elite"

fixture_comp["competition"] = fixture_comp["teams"].apply(classify_competition)

df_raw = df_raw.merge(fixture_comp[["fixture_id", "competition"]], on="fixture_id", how="left")
ts_raw = ts_raw.merge(fixture_comp[["fixture_id", "competition"]], on="fixture_id", how="left")

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
    if np.isscalar(result):
        return default if (np.isinf(result) or np.isnan(result)) else result
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)

# =========================
# PRÉPARATION JOUEURS
# =========================
df = df_raw.copy()
df["MIN"]  = df["stat_minutes"].apply(parse_minutes)
df["FGM"]  = df["stat_pointsTwoMade"].fillna(0)      + df["stat_pointsThreeMade"].fillna(0)
df["FGA"]  = df["stat_pointsTwoAttempted"].fillna(0)  + df["stat_pointsThreeAttempted"].fillna(0)
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
# NORMALISATION DES POSITIONS
# =========================
POSITION_MAP = {
    "PG": "Guard", "G": "Guard", "SG": "Guard", "GF": "Guard",
    "SF": "Forward", "FG": "Forward", "F": "Forward", "PF": "Forward",
    "FC": "Big", "C": "Big",
}
if "position" in df.columns:
    df["position"] = df["position"].map(POSITION_MAP).fillna(df["position"])

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
# PRÉPARATION STATS ÉQUIPE
# =========================
ts = ts_raw[ts_raw["team_name"] != "UNKNOWN"].copy()

ts["tm_FGM"]  = ts["team_pointsTwoMade"].fillna(0)      + ts["team_pointsThreeMade"].fillna(0)
ts["tm_FGA"]  = ts["team_pointsTwoAttempted"].fillna(0)  + ts["team_pointsThreeAttempted"].fillna(0)
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
ts["tm_3PM"]  = ts["team_pointsThreeMade"].fillna(0)
ts["tm_3PA"]  = ts["team_pointsThreeAttempted"].fillna(0)
ts["tm_POSS"] = ts["tm_FGA"] - ts["tm_OREB"] + ts["tm_TOV"] + 0.44 * ts["tm_FTA"]
ts["tm_MIN"]  = ts["team_minutes"].apply(parse_minutes) if "team_minutes" in ts.columns else 200.0

opp = ts[["fixture_id", "side", "team_name",
          "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM",
          "tm_OREB", "tm_DREB", "tm_REB", "tm_TOV",
          "tm_PTS", "tm_POSS", "tm_3PM", "tm_3PA", "tm_AST", "tm_STL", "tm_BLK", "tm_MIN"]].copy()
opp["opp_side"] = opp["side"].map({"home": "away", "away": "home"})
opp = opp.rename(columns={c: c.replace("tm_", "opp_") for c in opp.columns if c.startswith("tm_")})
opp = opp.rename(columns={"team_name": "opp_team_name"})
opp["side"] = opp["opp_side"]
opp = opp.drop(columns=["opp_side"])

ts_full = ts.merge(
    opp[["fixture_id", "side", "opp_team_name",
         "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM",
         "opp_OREB", "opp_DREB", "opp_REB", "opp_TOV",
         "opp_PTS", "opp_POSS", "opp_3PM", "opp_3PA", "opp_AST", "opp_STL", "opp_BLK", "opp_MIN"]],
    on=["fixture_id", "side"], how="left"
)

tm_cols = ["fixture_id", "team_name",
           "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM",
           "tm_OREB", "tm_DREB", "tm_REB", "tm_TOV",
           "tm_PTS", "tm_POSS", "tm_3PM", "tm_3PA", "tm_AST", "tm_STL", "tm_BLK", "tm_MIN",
           "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM",
           "opp_OREB", "opp_DREB", "opp_REB", "opp_TOV",
           "opp_PTS", "opp_POSS", "opp_3PM", "opp_3PA", "opp_AST", "opp_STL", "opp_BLK", "opp_MIN",
           "opp_team_name"]

df = df.merge(ts_full[tm_cols], on=["fixture_id", "team_name"], how="left")
df = df.drop_duplicates(
    subset=["fixture_id", "player_id"] if "player_id" in df.columns else ["fixture_id", "player_name"]
)

# =========================
# STATS AVANCÉES PAR MATCH
# =========================
df["eFG%"]   = safe_div(df["FGM"] + 0.5 * df["3PM"], df["FGA"]) * 100
df["TS%"]    = safe_div(df["PTS"], 2 * (df["FGA"] + 0.44 * df["FTA"])) * 100
df["FTrate"] = safe_div(df["FTA"], df["FGA"]) * 100
df["3Prate"] = safe_div(df["3PA"], (df["FGA"] + df["3PA"])) * 100

df["qAST"] = (
    safe_div(df["MIN"], df["tm_MIN"] / 5) *
    safe_div(df["tm_AST"], df["tm_FGM"])
).clip(0, 1)

df["FG_part"]   = df["FGM"] * (1 - 0.5 * safe_div(df["PTS"] - df["FTM"], 2 * df["FGM"])) * df["qAST"]
df["AST_part"]  = 0.5 * safe_div(df["tm_PTS"] - df["tm_FTM"], 2 * df["tm_FGM"]) * df["AST"]
df["FT_part"]   = (1 - (1 - safe_div(df["FTM"], df["FTA"])) ** 2) * 0.4 * df["FTA"]
df["OREB_part"] = df["OREB"] * safe_div(
    df["tm_PTS"], df["tm_FGA"] - df["tm_OREB"] + df["tm_TOV"] + 0.44 * df["tm_FTA"]
)
df["PProd"] = (df["FG_part"] + df["AST_part"]) * 2 + df["FTM"] + df["OREB_part"]

df["TotPoss"] = (
    df["FGA"]
    - df["OREB"] * safe_div(df["tm_OREB"], df["tm_OREB"] + df["opp_DREB"])
    + 0.44 * df["FTA"]
    + df["TOV"]
)
df["ScPoss"]  = safe_div(df["PProd"], df["PTS"].replace(0, np.nan)).fillna(0) * df["TotPoss"]
df["Floor%"]  = safe_div(df["ScPoss"], df["TotPoss"]) * 100
df["PPP"]     = safe_div(df["PTS"], df["TotPoss"])
df["ORtg"]    = safe_div(df["PProd"] * 100, df["TotPoss"])

df["DORpct"] = safe_div(df["opp_OREB"], df["opp_OREB"] + df["tm_DREB"])
df["DFGpct"] = safe_div(df["opp_FGM"], df["opp_FGA"])
df["FMwt"]   = safe_div(
    df["DFGpct"] * (1 - df["DORpct"]),
    df["DFGpct"] * (1 - df["DORpct"]) + (1 - df["DFGpct"]) * df["DORpct"]
)
df["Stops"] = (
    df["STL"]
    + df["BLK"] * df["FMwt"] * (1 - 1.07 * df["DORpct"])
    + df["DREB"] * (1 - df["FMwt"])
)
df["Stop%"] = safe_div(
    df["Stops"] * df["tm_POSS"],
    df["MIN"] / (df["tm_MIN"] / 5) * df["opp_POSS"]
) * 100
df["DPts"] = (
    df["opp_PTS"]
    * safe_div(df["MIN"], df["tm_MIN"] / 5)
    * safe_div(df["Stops"], df["opp_POSS"])
)
df["DRtg"] = safe_div(df["DPts"] * 100, df["Stops"].replace(0, np.nan)).fillna(
    safe_div(df["opp_PTS"] * 100, df["opp_POSS"])
)
df["NETRtg"] = df["ORtg"] - df["DRtg"]

df["USG%"] = safe_div(
    df["FGA"] + 0.44 * df["FTA"] + df["TOV"],
    safe_div(df["MIN"], df["tm_MIN"] / 5) * (df["tm_FGA"] + 0.44 * df["tm_FTA"] + df["tm_TOV"])
) * 100

df["AST%"]     = safe_div(
    df["AST"],
    safe_div(df["MIN"], df["tm_MIN"] / 5) * df["tm_FGM"] - df["FGM"]
) * 100
df["AST/TO"]   = safe_div(df["AST"], df["TOV"])
df["qAST_pct"] = df["qAST"] * 100

df["OREB%"] = safe_div(
    df["OREB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_OREB"] + df["opp_DREB"])
) * 100
df["DREB%"] = safe_div(
    df["DREB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_DREB"] + df["opp_OREB"])
) * 100
df["TRB%"] = safe_div(
    df["REB"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["tm_OREB"] + df["tm_DREB"] + df["opp_OREB"] + df["opp_DREB"])
) * 100

df["STL%"]   = safe_div(df["STL"] * (df["tm_MIN"] / 5), df["MIN"] * df["opp_POSS"]) * 100
df["BLK%"]   = safe_div(
    df["BLK"] * (df["tm_MIN"] / 5),
    df["MIN"] * (df["opp_FGA"] - df["opp_3PA"])
) * 100
df["STOCKS"] = df["STL"] + df["BLK"]
df["TOV%"]   = safe_div(df["TOV"], df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) * 100

df["GmSc"] = (
    df["PTS"] + 0.4 * df["FGM"] - 0.7 * df["FGA"]
    - 0.4 * (df["FTA"] - df["FTM"]) + 0.7 * df["OREB"]
    + 0.3 * df["DREB"] + df["STL"] + 0.7 * df["AST"]
    + 0.7 * df["BLK"] - 0.4 * df["FOUL"] - df["TOV"]
)

df["PIE_num"] = (
    df["PTS"] + df["FGM"] + df["FTM"] - df["FGA"] - df["FTA"]
    + df["DREB"] + 0.5 * df["OREB"]
    + df["AST"] + df["STL"] + 0.5 * df["BLK"]
    - df["FOUL"] - df["TOV"]
)
df["PIE_den"] = (
      (df["tm_PTS"]  + df["opp_PTS"])
    + (df["tm_FGM"]  + df["opp_FGM"])
    + (df["tm_FTM"]  + df["opp_FTM"])
    - (df["tm_FGA"]  + df["opp_FGA"])
    - (df["tm_FTA"]  + df["opp_FTA"])
    + (df["tm_OREB"] + df["tm_DREB"] + df["opp_OREB"] + df["opp_DREB"])
    + (df["tm_AST"]  + df["opp_AST"])
    + (df["tm_STL"]  + df["opp_STL"])
    + 0.5 * (df["tm_BLK"] + df["opp_BLK"])
)
df["PIE"] = safe_div(df["PIE_num"], df["PIE_den"]) * 100

df["PER"] = safe_div(1, df["MIN"]) * (
    df["3PM"] + (2 / 3) * df["AST"] + 2 * df["FGM"] - 0.5 * df["FGA"]
    + 0.5 * df["FTM"] - 0.75 * df["FTA"]
    + df["DREB"] + 0.75 * df["OREB"]
    + df["BLK"] + df["STL"] - df["TOV"] - df["FOUL"]
)

# =========================
# SIDEBAR
# ← competition_filter défini ici en premier
# =========================
with st.sidebar:
    st.title("🏀 LNB Dashboard")
    st.markdown("---")
    st.subheader("Filtres")

    competition_filter = st.selectbox(
        "Compétition",
        ["Elite", "Elite 2"],
        index=0
    )

    # ── Filtre équipes dynamique : on ne propose que les équipes
    #    présentes dans la compétition sélectionnée ──────────────
    teams_in_comp = sorted(
        df[df["competition"] == competition_filter]["team_name"]
        .dropna().unique().tolist()
    )
    teams_list = ["Toutes"] + teams_in_comp
    team_filter = st.selectbox("Équipe", teams_list)

    pos_list = ["Toutes"] + sorted(df["position"].dropna().unique().tolist()) \
               if "position" in df.columns else ["Toutes"]
    pos_filter = st.selectbox("Position", pos_list)

    st.markdown("---")
    st.subheader("Filtres volume")
    min_min = st.slider("Minutes min. (saison)", 0, 500, 50, step=10)

# =========================
# AGRÉGATION SAISON JOUEURS
# ← après la sidebar, filtrée par compétition
# =========================
sum_cols = ["MIN", "PTS", "AST", "REB", "OREB", "DREB", "STL", "BLK",
            "TOV", "FGM", "FGA", "FTM", "FTA", "3PM", "3PA", "FOUL",
            "PM", "STOCKS", "PProd", "ScPoss", "TotPoss", "Stops"]

mean_cols = ["eFG%", "TS%", "FTrate", "3Prate",
             "USG%", "AST%", "AST/TO", "qAST_pct",
             "OREB%", "DREB%", "TRB%",
             "STL%", "BLK%", "TOV%",
             "Floor%", "PPP",
             "ORtg", "DRtg", "NETRtg",
             "GmSc", "PIE", "PER", "Stop%"]

agg_map = {"fixture_id": "count"}
for c in sum_cols:
    if c in df.columns:
        agg_map[c] = "sum"
for c in mean_cols:
    if c in df.columns:
        agg_map[c] = "mean"

df_comp = df[df["competition"] == competition_filter].copy()

# On groupe uniquement sur player_name + team_name pour éviter les doublons
# causés par des valeurs de position différentes selon les matchs
player_season = df_comp.groupby(["player_name", "team_name"]).agg(agg_map).reset_index()

# On rattache la position la plus fréquente pour chaque joueur/équipe
if "position" in df_comp.columns:
    pos_mode = (
        df_comp.groupby(["player_name", "team_name"])["position"]
        .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None)
        .reset_index()
    )
    player_season = player_season.merge(pos_mode, on=["player_name", "team_name"], how="left")

player_season = player_season.rename(columns={"fixture_id": "GP"})

for col in ["MIN", "PTS", "AST", "REB", "OREB", "DREB",
            "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA",
            "3PM", "3PA", "FOUL", "STOCKS", "PM"]:
    if col in player_season.columns:
        player_season[f"{col}_PG"] = safe_div(player_season[col], player_season["GP"])

player_season["FG%"] = safe_div(player_season["FGM"], player_season["FGA"]) * 100
player_season["3P%"] = safe_div(player_season["3PM"], player_season["3PA"]) * 100
player_season["FT%"] = safe_div(player_season["FTM"], player_season["FTA"]) * 100

player_season["EVAL"] = (
    player_season["PTS"] + player_season["REB"] + player_season["AST"]
    + player_season["STL"] + player_season["BLK"]
    - (player_season["FGA"] - player_season["FGM"])
    - (player_season["FTA"] - player_season["FTM"])
    - player_season["TOV"]
)
player_season["EVAL_PG"] = safe_div(player_season["EVAL"], player_season["GP"])

# =========================
# AGRÉGATION SAISON ÉQUIPES
# ← après la sidebar, filtrée par compétition
# =========================
ts_comp = ts_full[ts_full["competition"] == competition_filter].copy()

team_agg = ts_comp.groupby("team_name").agg(
    GP       = ("fixture_id", "nunique"),
    tm_MIN   = ("tm_MIN",  "sum"),
    tm_PTS   = ("tm_PTS",  "sum"),
    tm_FGM   = ("tm_FGM",  "sum"),
    tm_FGA   = ("tm_FGA",  "sum"),
    tm_FTA   = ("tm_FTA",  "sum"),
    tm_FTM   = ("tm_FTM",  "sum"),
    tm_OREB  = ("tm_OREB", "sum"),
    tm_DREB  = ("tm_DREB", "sum"),
    tm_REB   = ("tm_REB",  "sum"),
    tm_TOV   = ("tm_TOV",  "sum"),
    tm_AST   = ("tm_AST",  "sum"),
    tm_STL   = ("tm_STL",  "sum"),
    tm_BLK   = ("tm_BLK",  "sum"),
    tm_3PA   = ("tm_3PA",  "sum"),
    tm_3PM   = ("tm_3PM",  "sum"),
    tm_POSS  = ("tm_POSS", "sum"),
    opp_PTS  = ("opp_PTS",  "sum"),
    opp_POSS = ("opp_POSS", "sum"),
    opp_FGM  = ("opp_FGM",  "sum"),
    opp_FGA  = ("opp_FGA",  "sum"),
    opp_FTM  = ("opp_FTM",  "sum"),
    opp_FTA  = ("opp_FTA",  "sum"),
    opp_OREB = ("opp_OREB", "sum"),
    opp_DREB = ("opp_DREB", "sum"),
    opp_REB  = ("opp_REB",  "sum"),
    opp_TOV  = ("opp_TOV",  "sum"),
    opp_AST  = ("opp_AST",  "sum"),
    opp_STL  = ("opp_STL",  "sum"),
    opp_BLK  = ("opp_BLK",  "sum"),
    opp_3PA  = ("opp_3PA",  "sum"),
    opp_3PM  = ("opp_3PM",  "sum"),
).reset_index()

team_agg["PACE40"] = safe_div(team_agg["tm_POSS"], team_agg["GP"])
team_agg["eFG%"]   = safe_div(team_agg["tm_FGM"], team_agg["tm_FGA"]) * 100
team_agg["TS%"]    = safe_div(team_agg["tm_PTS"], 2 * (team_agg["tm_FGA"] + 0.44 * team_agg["tm_FTA"])) * 100
team_agg["TOV%"]   = safe_div(team_agg["tm_TOV"], team_agg["tm_FGA"] + 0.44 * team_agg["tm_FTA"] + team_agg["tm_TOV"]) * 100
team_agg["OREB%"]  = safe_div(team_agg["tm_OREB"], team_agg["tm_OREB"] + team_agg["opp_OREB"]) * 100
team_agg["ORtg"]   = safe_div(team_agg["tm_PTS"]  * 100, team_agg["tm_POSS"])
team_agg["DRtg"]   = safe_div(team_agg["opp_PTS"] * 100, team_agg["opp_POSS"])
team_agg["NETRtg"] = team_agg["ORtg"] - team_agg["DRtg"]
team_agg["PPP"]    = safe_div(team_agg["tm_PTS"], team_agg["tm_POSS"])
team_agg["PTS_PG"] = safe_div(team_agg["tm_PTS"], team_agg["GP"])
team_agg["AST_PG"] = safe_div(team_agg["tm_AST"], team_agg["GP"])
team_agg["REB_PG"] = safe_div(team_agg["tm_REB"], team_agg["GP"])
team_agg["TOV_PG"] = safe_div(team_agg["tm_TOV"], team_agg["GP"])
team_agg["POSS_PG"]= safe_div(team_agg["tm_POSS"], team_agg["GP"])

team_season = team_agg.rename(columns={
    "tm_PTS": "PTS", "tm_FGM": "FGM", "tm_FGA": "FGA",
    "tm_FTA": "FTA", "tm_FTM": "FTM", "tm_AST": "AST",
    "tm_REB": "REB", "tm_TOV": "TOV", "tm_STL": "STL",
    "tm_BLK": "BLK", "tm_POSS": "POSS",
})

# =========================
# SUITE SIDEBAR
# ← sliders volume qui dépendent de player_season
# =========================
with st.sidebar:
    st.markdown("---")
    st.subheader("Top joueurs")
    pg_stat_options = [c for c in player_season.columns if c.endswith("_PG") or c.endswith("%")]
    default_pg = next((c for c in pg_stat_options if "PTS" in c), pg_stat_options[0] if pg_stat_options else None)
    if pg_stat_options:
        stat_col = st.selectbox("Stat", pg_stat_options,
                                index=pg_stat_options.index(default_pg) if default_pg in pg_stat_options else 0)
        top_n = st.slider("Nombre de joueurs", 3, 20, 10)
    else:
        stat_col, top_n = None, 10

    st.markdown("---")
    st.subheader("Filtres volume")
    min_fga = st.slider("FGA min. (saison)", 0, int(player_season["FGA"].max()), 0, step=5)
    min_3pa = st.slider("3PA min. (saison)", 0, int(player_season["3PA"].max()), 0, step=5)
    min_fta = st.slider("FTA min. (saison)", 0, int(player_season["FTA"].max()), 0, step=5)

# =========================
# FILTRAGE
# =========================
df_filtered = df[df["competition"] == competition_filter].copy()

if team_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["team_name"] == team_filter]
if pos_filter != "Toutes" and "position" in df.columns:
    df_filtered = df_filtered[df_filtered["position"] == pos_filter]

adv_filtered = player_season[
    (player_season["MIN"] >= min_min) &
    (player_season["FGA"] >= min_fga) &
    (player_season["3PA"] >= min_3pa) &
    (player_season["FTA"] >= min_fta)
].copy()

if team_filter != "Toutes":
    adv_filtered = adv_filtered[adv_filtered["team_name"] == team_filter]
if pos_filter != "Toutes" and "position" in adv_filtered.columns:
    adv_filtered = adv_filtered[adv_filtered["position"] == pos_filter]


# =========================
# FICHE JOUEUR — DIALOG
# =========================
@st.dialog("📄 Fiche joueur", width="large")
def show_fiche(selected_fiche):
    pdata   = player_season[player_season["player_name"] == selected_fiche].iloc[0]
    pmatches = df_comp[df_comp["player_name"] == selected_fiche].copy().sort_values("fixture_id")

    pos_label = pdata.get("position", "—") if "position" in pdata.index else "—"
    st.markdown(f"## {selected_fiche}")
    st.caption(f"🏀 {pdata['team_name']}  ·  {pos_label}  ·  {int(pdata['GP'])} matchs joués")
    st.markdown("---")

    # KPIs
    k = st.columns(9)
    for col, (label, val, unit) in zip(k, [
        ("PTS", pdata.get("PTS_PG", 0), "/m"), ("AST", pdata.get("AST_PG", 0), "/m"),
        ("REB", pdata.get("REB_PG", 0), "/m"), ("STL", pdata.get("STL_PG", 0), "/m"),
        ("BLK", pdata.get("BLK_PG", 0), "/m"), ("FG%", pdata.get("FG%", 0), "%"),
        ("3P%", pdata.get("3P%", 0), "%"),     ("FT%", pdata.get("FT%", 0), "%"),
        ("TS%", pdata.get("TS%", 0), "%"),
    ]):
        col.metric(label, f"{val:.1f}{unit}")

    k2 = st.columns(6)
    for col, (label, val, unit) in zip(k2, [
        ("MIN",  pdata.get("MIN_PG", 0),  "/m"), ("USG%", pdata.get("USG%", 0),    "%"),
        ("ORtg", pdata.get("ORtg", 0),    ""),   ("DRtg", pdata.get("DRtg", 0),    ""),
        ("PIE",  pdata.get("PIE", 0),     "%"),  ("EVAL", pdata.get("EVAL_PG", 0), "/m"),
    ]):
        col.metric(label, f"{val:.1f}{unit}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Radar
    with col_left:
        st.subheader("🕸️ Profil")
        radar_stats_f = [s for s in ["ORtg","DRtg","TS%","USG%","TRB%","AST%","STL%","BLK%","Floor%","PIE"]
                         if s in player_season.columns]
        vals_f = []
        for s in radar_stats_f:
            col_max = player_season[s].replace([np.inf,-np.inf], np.nan).max()
            col_min = player_season[s].replace([np.inf,-np.inf], np.nan).min()
            norm = safe_div(pdata[s] - col_min, col_max - col_min) * 100
            if s == "DRtg": norm = 100 - norm
            vals_f.append(float(norm))
        fig_r = go.Figure(go.Scatterpolar(
            r=vals_f + [vals_f[0]], theta=radar_stats_f + [radar_stats_f[0]],
            fill="toself", line_color="#185FA5", opacity=0.75,
        ))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                            showlegend=False, height=320, margin=dict(l=30,r=30,t=20,b=20))
        st.plotly_chart(fig_r, use_container_width=True, key=f"dlg_radar_{selected_fiche}")
        st.caption("Normalisé 0-100 · DRtg inversé")

    # Percentiles
    with col_right:
        st.subheader("📊 Percentiles")
        pct_rows = []
        for s in [s for s in ["PTS_PG","AST_PG","REB_PG","STL_PG","BLK_PG","TS%","USG%","PIE","ORtg","DRtg"]
                  if s in player_season.columns]:
            series = player_season[s].replace([np.inf,-np.inf], np.nan).dropna()
            val    = pdata.get(s, np.nan)
            if pd.isna(val): continue
            pct = int((series < val).sum() / len(series) * 100)
            if s == "DRtg": pct = 100 - pct
            pct_rows.append({"Stat": s, "Valeur": round(float(val),1), "Percentile": pct})
        if pct_rows:
            pct_df = pd.DataFrame(pct_rows)
            pct_df["label"] = pct_df["Percentile"].astype(str) + "e"
            fig_p = px.bar(pct_df, x="Percentile", y="Stat",
                           orientation="h", text="label",
                           color="Percentile", color_continuous_scale="Blues", range_x=[0,100])
            fig_p.add_vline(x=50, line_dash="dash", line_color="grey", opacity=0.5)
            fig_p.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                                height=320, margin=dict(l=0,r=10,t=10,b=0))
            fig_p.update_traces(textposition="inside", insidetextanchor="end",
                                textfont=dict(color="black", size=12))
            st.plotly_chart(fig_p, use_container_width=True, key=f"dlg_pct_{selected_fiche}")

    st.markdown("---")

    # Évolution match par match
    st.subheader("📅 Évolution match par match")
    stat_opts = [s for s in ["PTS","AST","REB","STL","BLK","MIN","GmSc"] if s in pmatches.columns]
    stats_sel = st.multiselect("Stats", stat_opts,
                                default=[s for s in ["PTS","AST","REB"] if s in stat_opts],
                                key=f"dlg_evo_sel_{selected_fiche}")
    if stats_sel and len(pmatches) > 0:
        pmatches["match_label"] = (
            pmatches.get("away_team", pd.Series("", index=pmatches.index)).fillna("") +
            " @ " +
            pmatches.get("home_team",  pd.Series("", index=pmatches.index)).fillna("")
        )
        fig_e = go.Figure()
        for i, stat in enumerate(stats_sel):
            c_ = ["#185FA5","#E05A2B","#2CA02C","#9467BD","#8C564B","#E377C2"][i % 6]
            fig_e.add_trace(go.Scatter(
                x=list(range(1, len(pmatches)+1)), y=pmatches[stat].tolist(),
                mode="lines+markers", name=stat,
                line=dict(color=c_, width=2), marker=dict(size=6),
                hovertext=pmatches["match_label"].tolist(),
                hovertemplate="%{hovertext}<br>" + stat + ": %{y}<extra></extra>",
            ))
        fig_e.update_layout(xaxis_title="Match #", height=300,
                            margin=dict(l=0,r=0,t=20,b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            hovermode="x unified")
        st.plotly_chart(fig_e, use_container_width=True,
                        key=f"dlg_evo_{selected_fiche}_{'_'.join(stats_sel)}")

# =========================
# ONGLETS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dashboard",
    "📈 Stats classiques",
    "📊 Stats avancées joueurs",
    "🏟️ Stats avancées équipes",
    "⚔️ Comparaison joueurs",
])

# ══════════════════════════════════════════
# ONGLET 1 — DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.title("🏀 Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Joueurs uniques", df_filtered["player_name"].nunique())
    k2.metric("Matchs", df_filtered["fixture_id"].nunique())
    k3.metric("Lignes filtrées", len(df_filtered))
    k4.metric("Lignes totales", len(df))

    st.markdown("---")

    if stat_col and len(adv_filtered) > 0:
        st.subheader(f"🔥 Top {top_n} joueurs — {stat_col} (moy/match)")
        top = (adv_filtered[["player_name", stat_col]]
               .sort_values(stat_col, ascending=False)
               .head(top_n)
               .copy())
        top.columns = ["Joueur", "val"]
        fig = px.bar(top, x="val", y="Joueur", orientation="h",
                     color="val", color_continuous_scale="Blues", text_auto=".1f")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0), height=400)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True, key=f"top_bar_{competition_filter}_{team_filter}_{stat_col}_{top_n}")

        st.markdown("---")
        st.subheader("🎯 Pourcentages de tir")
        pct = adv_filtered[["player_name", "team_name", "GP", "FG%", "FGA_PG", "3P%", "3PA_PG", "FT%", "FTA_PG"]].copy()
        pct = pct.rename(columns={
            "player_name": "Joueur", "team_name": "Équipe",
            "FGA_PG": "FGA/G", "3PA_PG": "3PA/G", "FTA_PG": "FTA/G"
        })
        st.dataframe(pct.round(1).sort_values("FG%", ascending=False),
                     use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("📋 Données filtrées")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"{len(df_filtered)} lignes · {df_filtered.shape[1]} colonnes")
    with c2:
        st.download_button("⬇️ CSV", df_filtered.to_csv(index=False).encode("utf-8"),
                           "boxscores_filtres.csv", "text/csv")

    cols_to_hide = [
        "fixture_id", "player_id", "team_id", "side", "opp_team_name",
        "tm_FGM", "tm_FGA", "tm_FTA", "tm_FTM", "tm_OREB", "tm_DREB",
        "tm_REB", "tm_TOV", "tm_PTS", "tm_POSS", "tm_3PA", "tm_AST", "tm_MIN",
        "opp_FGM", "opp_FGA", "opp_FTA", "opp_FTM", "opp_OREB", "opp_DREB",
        "opp_REB", "opp_TOV", "opp_PTS", "opp_POSS", "opp_3PA", "opp_AST", "opp_MIN",
        "DORpct", "DFGpct", "FMwt", "PIE_num", "PIE_den", "ScPoss", "DPts",
        "FG_part", "AST_part", "FT_part", "OREB_part", "qAST",
    ]
    display_cols = [c for c in df_filtered.columns if c not in cols_to_hide]
    priority = ["player_name", "team_name", "home_team", "away_team", "position"]
    front = [c for c in priority if c in display_cols]
    rest  = [c for c in display_cols if c not in front]
    display_cols = front + rest

    st.dataframe(df_filtered[display_cols].rename(columns={"bib": "numéro"}),
                 use_container_width=True, height=500)

    with st.expander("🧾 Aperçu données brutes (5 premières lignes)"):
        st.dataframe(df[display_cols].rename(columns={"bib": "numéro"}).head(),
                     use_container_width=True)
        st.caption(f"Dataset complet : {df.shape[0]} lignes · {df.shape[1]} colonnes")


# ══════════════════════════════════════════
# ONGLET 2 — STATS CLASSIQUES
# ══════════════════════════════════════════
with tab2:
    st.title("📈 Stats classiques — Joueurs")

    # ── Recherche joueur avec autocomplétion ───────────────────
    players_available_classic = sorted(adv_filtered["player_name"].dropna().unique().tolist())
    selected_classic = st.multiselect(
        "🔍 Rechercher un joueur (tapez pour filtrer les suggestions)",
        options=players_available_classic,
        placeholder="Tapez un nom...",
        key="search_classic"
    )
    adv_classic = adv_filtered.copy()
    if selected_classic:
        adv_classic = adv_classic[adv_classic["player_name"].isin(selected_classic)]
    # ──────────────────────────────────────────────────────────

    st.caption(f"Saison complète · Minimum {min_min} min · {len(adv_classic)} joueurs · moyennes par match")

    cols_classic = [
        "player_name", "team_name", "GP", "MIN_PG",
        "PTS_PG",
        "FGM_PG", "FGA_PG", "FG%",
        "3PM_PG", "3PA_PG", "3P%",
        "FTM_PG", "FTA_PG", "FT%",
        "AST_PG", "OREB_PG", "DREB_PG", "REB_PG",
        "STL_PG", "BLK_PG", "TOV_PG",
        "EVAL_PG"
    ]
    av = [c for c in cols_classic if c in adv_classic.columns]

    df_classic_display = adv_classic[av].rename(columns={
            "player_name": "Joueur", "team_name": "Équipe",
            "MIN_PG": "MIN", "PTS_PG": "PTS",
            "FGM_PG": "FGM", "FGA_PG": "FGA",
            "3PM_PG": "3PM", "3PA_PG": "3PA",
            "FTM_PG": "FTM", "FTA_PG": "FTA",
            "AST_PG": "AST", "OREB_PG": "REB O",
            "DREB_PG": "REB D", "REB_PG": "REB",
            "STL_PG": "STL", "BLK_PG": "BLK",
            "TOV_PG": "TOV", "EVAL_PG": "EVAL"
        }).round(1).sort_values("PTS", ascending=False).reset_index(drop=True)
    st.caption("💡 Cliquez sur une ligne pour ouvrir la fiche joueur")
    sel_classic = st.dataframe(
        df_classic_display,
        use_container_width=True, height=550,
        selection_mode="single-row", on_select="rerun",
        key="sel_classic_table"
    )
    rows_classic = sel_classic.selection.rows if sel_classic.selection.rows else []
    if rows_classic:
        picked = df_classic_display.iloc[rows_classic[0]]["Joueur"]
        show_fiche(picked)

    st.markdown("---")
    st.subheader("🏟️ Stats classiques — Équipes")

    team_classic = team_agg.copy()
    team_classic["FG%"]  = safe_div(team_classic["tm_FGM"], team_classic["tm_FGA"]) * 100
    team_classic["3P%"]  = safe_div(team_classic["tm_3PM"], team_classic["tm_3PA"]) * 100
    team_classic["FT%"]  = safe_div(team_classic["tm_FTM"], team_classic["tm_FTA"]) * 100
    team_classic["EVAL"] = (
        team_classic["tm_PTS"] + team_classic["tm_REB"] + team_classic["tm_AST"]
        + team_classic["tm_STL"] + team_classic["tm_BLK"]
        - (team_classic["tm_FGA"] - team_classic["tm_FGM"])
        - (team_classic["tm_FTA"] - team_classic["tm_FTM"])
        - team_classic["tm_TOV"]
    )

    for col, src in [
        ("PTS",  "tm_PTS"),  ("FGM", "tm_FGM"),  ("FGA", "tm_FGA"),
        ("3PM",  "tm_3PM"),  ("3PA", "tm_3PA"),   ("FTM", "tm_FTM"),  ("FTA", "tm_FTA"),
        ("AST",  "tm_AST"),  ("OREB","tm_OREB"),  ("DREB","tm_DREB"),
        ("REB",  "tm_REB"),  ("STL", "tm_STL"),   ("BLK", "tm_BLK"),
        ("TOV",  "tm_TOV"),  ("EVAL","EVAL"),
    ]:
        if src in team_classic.columns:
            team_classic[f"{col}_PG"] = safe_div(team_classic[src], team_classic["GP"])

    cols_team_classic = [
        "team_name", "GP",
        "PTS_PG", "FGM_PG", "FGA_PG", "FG%",
        "3PM_PG", "3PA_PG", "3P%",
        "FTM_PG", "FTA_PG", "FT%",
        "AST_PG", "OREB_PG", "DREB_PG", "REB_PG",
        "STL_PG", "BLK_PG", "TOV_PG", "EVAL_PG"
    ]
    av_team = [c for c in cols_team_classic if c in team_classic.columns]

    st.dataframe(
        team_classic[av_team].rename(columns={
            "team_name": "Équipe",
            "PTS_PG": "PTS", "FGM_PG": "FGM", "FGA_PG": "FGA",
            "3PM_PG": "3PM", "3PA_PG": "3PA",
            "FTM_PG": "FTM", "FTA_PG": "FTA",
            "AST_PG": "AST", "OREB_PG": "REB O",
            "DREB_PG": "REB D", "REB_PG": "REB",
            "STL_PG": "STL", "BLK_PG": "BLK",
            "TOV_PG": "TOV", "EVAL_PG": "EVAL"
        }).round(1).sort_values("PTS", ascending=False),
        use_container_width=True, height=450
    )

    st.markdown("---")
    st.subheader("🏟️ Stats classiques — Adversaires")

    team_classic_opp = team_agg.copy()
    team_classic_opp["opp_FG%"]  = safe_div(team_classic_opp["opp_FGM"], team_classic_opp["opp_FGA"]) * 100
    team_classic_opp["opp_3P%"]  = safe_div(team_classic_opp["opp_3PM"], team_classic_opp["opp_3PA"]) * 100
    team_classic_opp["opp_FT%"]  = safe_div(team_classic_opp["opp_FTM"], team_classic_opp["opp_FTA"]) * 100
    team_classic_opp["opp_EVAL"] = (
        team_classic_opp["opp_PTS"] + team_classic_opp["opp_REB"] + team_classic_opp["opp_AST"]
        + team_classic_opp["opp_STL"] + team_classic_opp["opp_BLK"]
        - (team_classic_opp["opp_FGA"] - team_classic_opp["opp_FGM"])
        - (team_classic_opp["opp_3PA"] - team_classic_opp["opp_3PM"])
        - (team_classic_opp["opp_FTA"] - team_classic_opp["opp_FTM"])
        - team_classic_opp["opp_TOV"]
    )

    for col, src in [
        ("PTS",  "opp_PTS"),  ("FGM", "opp_FGM"),  ("FGA", "opp_FGA"),
        ("3PM",  "opp_3PM"),  ("3PA", "opp_3PA"),   ("FTM", "opp_FTM"),  ("FTA", "opp_FTA"),
        ("AST",  "opp_AST"),  ("OREB","opp_OREB"),  ("DREB","opp_DREB"),
        ("REB",  "opp_REB"),  ("STL", "opp_STL"),   ("BLK", "opp_BLK"),
        ("TOV",  "opp_TOV"),  ("EVAL","opp_EVAL"),
    ]:
        if src in team_classic_opp.columns:
            team_classic_opp[f"opp_{col}_PG"] = safe_div(team_classic_opp[src], team_classic_opp["GP"])

    cols_team_opp = [
        "team_name", "GP",
        "opp_PTS_PG", "opp_FGM_PG", "opp_FGA_PG", "opp_FG%",
        "opp_3PM_PG", "opp_3PA_PG", "opp_3P%",
        "opp_FTM_PG", "opp_FTA_PG", "opp_FT%",
        "opp_AST_PG", "opp_OREB_PG", "opp_DREB_PG", "opp_REB_PG",
        "opp_STL_PG", "opp_BLK_PG", "opp_TOV_PG", "opp_EVAL_PG"
    ]
    av_opp = [c for c in cols_team_opp if c in team_classic_opp.columns]

    st.dataframe(
        team_classic_opp[av_opp].rename(columns={
            "team_name": "Équipe",
            "opp_PTS_PG": "PTS", "opp_FGM_PG": "FGM", "opp_FGA_PG": "FGA", "opp_FG%": "FG%",
            "opp_3PM_PG": "3PM", "opp_3PA_PG": "3PA", "opp_3P%": "3P%",
            "opp_FTM_PG": "FTM", "opp_FTA_PG": "FTA", "opp_FT%": "FT%",
            "opp_AST_PG": "AST", "opp_OREB_PG": "REB O",
            "opp_DREB_PG": "REB D", "opp_REB_PG": "REB",
            "opp_STL_PG": "STL", "opp_BLK_PG": "BLK",
            "opp_TOV_PG": "TOV", "opp_EVAL_PG": "EVAL"
        }).round(1).sort_values("PTS", ascending=False),
        use_container_width=True, height=450
    )

    st.download_button("⬇️ Télécharger stats classiques",
                       adv_classic[av].round(2).to_csv(index=False).encode("utf-8"),
                       "stats_classiques_joueurs.csv", "text/csv")


# ══════════════════════════════════════════
# ONGLET 3 — STATS AVANCÉES JOUEURS
# ══════════════════════════════════════════
with tab3:
    st.title("📊 Stats avancées — Joueurs")

    # ── Recherche joueur avec autocomplétion ───────────────────
    players_available_adv = sorted(adv_filtered["player_name"].dropna().unique().tolist())
    selected_adv = st.multiselect(
        "🔍 Rechercher un joueur (tapez pour filtrer les suggestions)",
        options=players_available_adv,
        placeholder="Tapez un nom...",
        key="search_adv"
    )
    adv_search = adv_filtered.copy()
    if selected_adv:
        adv_search = adv_search[adv_search["player_name"].isin(selected_adv)]
    # ──────────────────────────────────────────────────────────

    st.caption(f"Saison complète · Minimum {min_min} min · {len(adv_search)} joueurs")

    def rd(d, dec=2):
        return d.round(dec)

    def safe_sort(df_, preferred_cols, fallback_col=None):
        for c in preferred_cols:
            if c in df_.columns:
                return df_.sort_values(c, ascending=False)
        if fallback_col and fallback_col in df_.columns:
            return df_.sort_values(fallback_col, ascending=False)
        return df_

    base = ["player_name", "team_name", "GP", "MIN_PG"]

    s1, s2, s3, s4, s5, s6 = st.tabs([
        "🎯 Tirs & scoring", "🎩 Playmaking",
        "💪 Rebonds", "🛡️ Défense",
        "⚡ Efficacité", "🔁 Possessions"
    ])

    with s1:
        cols = base + ["eFG%", "TS%", "FTrate", "3Prate", "GmSc"]
        av = [c for c in cols if c in adv_search.columns]
        df_s1 = rd(safe_sort(adv_search[av], ["PTS_PG", "eFG%"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne pour ouvrir la fiche joueur · eFG%/TS% = efficacité tir · GmSc = Gamescore (Hollinger)")
        sel_s1 = st.dataframe(df_s1, use_container_width=True, height=500,
                              selection_mode="single-row", on_select="rerun", key="sel_s1")
        if sel_s1.selection.rows:
            show_fiche(df_s1.iloc[sel_s1.selection.rows[0]]["player_name"])

    with s2:
        cols = base + ["AST%", "AST/TO", "qAST_pct", "USG%", "TOV%"]
        av = [c for c in cols if c in adv_search.columns]
        df_s2 = rd(safe_sort(adv_search[av], ["AST%", "USG%"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne · AST% = % tirs équipe assistés · USG% = % possessions utilisées · AST/TO = ratio passes/pertes")
        sel_s2 = st.dataframe(df_s2, use_container_width=True, height=500,
                              selection_mode="single-row", on_select="rerun", key="sel_s2")
        if sel_s2.selection.rows:
            show_fiche(df_s2.iloc[sel_s2.selection.rows[0]]["player_name"])

    with s3:
        cols = base + ["OREB%", "DREB%", "TRB%"]
        av = [c for c in cols if c in adv_search.columns]
        df_s3 = rd(safe_sort(adv_search[av], ["TRB%", "DREB%"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne · OREB%/DREB%/TRB% = % rebonds disponibles captés")
        sel_s3 = st.dataframe(df_s3, use_container_width=True, height=500,
                              selection_mode="single-row", on_select="rerun", key="sel_s3")
        if sel_s3.selection.rows:
            show_fiche(df_s3.iloc[sel_s3.selection.rows[0]]["player_name"])

    with s4:
        cols = base + ["STL%", "BLK%", "Stop%"]
        av = [c for c in cols if c in adv_search.columns]
        df_s4 = rd(safe_sort(adv_search[av], ["STL%", "BLK%"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne · Stop% = % possessions adverses stoppées (Dean Oliver)")
        sel_s4 = st.dataframe(df_s4, use_container_width=True, height=500,
                              selection_mode="single-row", on_select="rerun", key="sel_s4")
        if sel_s4.selection.rows:
            show_fiche(df_s4.iloc[sel_s4.selection.rows[0]]["player_name"])

    with s5:
        cols = base + ["PER", "PIE", "GmSc", "ORtg", "DRtg", "NETRtg", "PM_PG"]
        av = [c for c in cols if c in adv_search.columns]
        df_s5 = rd(safe_sort(adv_search[av], ["PIE", "PER"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne pour ouvrir la fiche joueur")
        sel_s5 = st.dataframe(df_s5, use_container_width=True, height=400,
                              selection_mode="single-row", on_select="rerun", key="sel_s5")
        if sel_s5.selection.rows:
            show_fiche(df_s5.iloc[sel_s5.selection.rows[0]]["player_name"])

        st.markdown("---")
        if "ORtg" in adv_search.columns and "DRtg" in adv_search.columns:
            st.subheader("ORtg vs DRtg")
            fig2 = px.scatter(
                adv_search, x="ORtg", y="DRtg", text="player_name",
                color="team_name", size="MIN",
                hover_data=["GP", "PER", "PIE", "NETRtg"],
                labels={"ORtg": "Rating offensif (per 100 poss)",
                        "DRtg": "Rating défensif (per 100 poss)"},
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        st.caption("PER/PIE/GmSc = moyenne par match · ORtg/DRtg per 100 possessions (Dean Oliver)")

    with s6:
        cols = base + ["PPP", "Floor%", "USG%", "TOV%", "ORtg", "NETRtg"]
        av = [c for c in cols if c in adv_search.columns]
        df_s6 = rd(safe_sort(adv_search[av], ["Floor%", "PPP"])).reset_index(drop=True)
        st.caption("💡 Cliquez sur une ligne · PPP = points/possession · Floor% = % possessions productrices")
        sel_s6 = st.dataframe(df_s6, use_container_width=True, height=500,
                              selection_mode="single-row", on_select="rerun", key="sel_s6")
        if sel_s6.selection.rows:
            show_fiche(df_s6.iloc[sel_s6.selection.rows[0]]["player_name"])

    st.markdown("---")
    st.download_button("⬇️ Télécharger stats avancées joueurs",
                       adv_search.round(3).to_csv(index=False).encode("utf-8"),
                       "stats_avancees_joueurs.csv", "text/csv")


# ══════════════════════════════════════════
# ONGLET 4 — STATS AVANCÉES ÉQUIPES
# ══════════════════════════════════════════
with tab4:
    st.title("🏟️ Stats avancées — Équipes")
    st.caption(f"Saison complète · {len(team_season)} équipes")

    if len(team_season) > 0:
        best_net = team_season.loc[team_season["NETRtg"].idxmax()]
        best_off = team_season.loc[team_season["ORtg"].idxmax()]
        best_def = team_season.loc[team_season["DRtg"].idxmin()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Meilleur NETRtg", best_net["team_name"], f"{best_net['NETRtg']:.1f}")
        k2.metric("Meilleure attaque", best_off["team_name"], f"{best_off['ORtg']:.1f} ORtg")
        k3.metric("Meilleure défense", best_def["team_name"], f"{best_def['DRtg']:.1f} DRtg")

    st.markdown("---")
    te1, = st.tabs(["📈 Ratings & efficacité"])

    with te1:
        cols = ["team_name", "GP", "ORtg", "DRtg", "NETRtg", "PPP",
                "eFG%", "TS%", "TOV%", "OREB%", "PACE40", "POSS_PG"]
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

    st.markdown("---")
    st.download_button("⬇️ Télécharger stats avancées équipes",
                       team_season.round(3).to_csv(index=False).encode("utf-8"),
                       "stats_avancees_equipes.csv", "text/csv")


# ══════════════════════════════════════════
# ONGLET 5 — COMPARAISON JOUEURS
# ══════════════════════════════════════════
with tab5:
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

        st.markdown("---")
        stats_kpi = ["GP", "MIN_PG", "PTS_PG", "AST_PG", "REB_PG", "STL_PG", "BLK_PG",
                     "FG%", "3P%", "FT%", "EVAL_PG",
                     "ORtg", "DRtg", "NETRtg", "PER", "PIE", "TS%", "USG%"]
        stats_kpi = [s for s in stats_kpi if s in player_season.columns]

        cols_kpi = st.columns(len(stats_kpi))
        for i, stat in enumerate(stats_kpi):
            v1, v2 = d1[stat], d2[stat]
            cols_kpi[i].metric(
                stat, f"{v1:.1f}",
                delta=f"{v1 - v2:+.1f}",
                delta_color="normal" if stat != "DRtg" else "inverse"
            )

        st.markdown("---")

        radar_stats = ["ORtg", "DRtg", "TS%", "USG%", "TRB%", "AST%",
                       "STL%", "BLK%", "Floor%", "PIE"]
        radar_stats = [s for s in radar_stats if s in player_season.columns]

        fig_radar = go.Figure()
        for player, data, color in [(p1, d1, "#185FA5"), (p2, d2, "#D85A30")]:
            vals = []
            for s in radar_stats:
                col_max = player_season[s].replace([np.inf, -np.inf], np.nan).max()
                col_min = player_season[s].replace([np.inf, -np.inf], np.nan).min()
                norm = safe_div(data[s] - col_min, col_max - col_min) * 100
                if s == "DRtg":
                    norm = 100 - norm
                vals.append(float(norm))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_stats + [radar_stats[0]],
                fill="toself", name=player, line_color=color, opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=500, margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Valeurs normalisées 0-100 · DRtg inversé (plus haut = meilleure défense)")

        st.markdown("---")
        st.subheader("Tableau comparatif complet")
        all_stats = [c for c in player_season.columns
                     if c not in ["player_name", "team_name"]
                     and player_season[c].dtype in [np.float64, np.int64]]

        compare_df = pd.DataFrame({
            "Stat": all_stats,
            p1: [round(float(d1[s]), 2) if s in d1 and not pd.isna(d1[s]) else "—" for s in all_stats],
            p2: [round(float(d2[s]), 2) if s in d2 and not pd.isna(d2[s]) else "—" for s in all_stats],
        })
        st.dataframe(compare_df, use_container_width=True, height=500)

        st.download_button(
            "⬇️ Télécharger comparaison",
            compare_df.to_csv(index=False).encode("utf-8"),
            f"comparaison_{p1}_vs_{p2}.csv", "text/csv"
        )