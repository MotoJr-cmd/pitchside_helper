from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import soccerdata as sd  # type: ignore[import]
except ImportError as e:
    raise ImportError(
        "The 'soccerdata' package is required. Install it with 'pip install soccerdata'."
    ) from e

try:
    from rapidfuzz import fuzz, process  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None
    process = None


LEAGUE = "ENG-Premier League"
SEASON = "2025-2026"

PROG_PASS_THRESHOLD = 8.0
RATING_THRESHOLD = 6.8
MIN_MINUTES_DEFAULT = 300
FUZZY_SCORE_THRESHOLD = 90

PLAYER_NAME_MAP: Dict[str, str] = {
    # Example mappings; extend as needed
    "Martin Odegaard": "Martin Ødegaard",
    "Heung-Min Son": "Son Heung-Min",
}


@dataclass
class GapAnalysisConfig:
    league: str = LEAGUE
    season: str = SEASON
    prog_pass_threshold: float = PROG_PASS_THRESHOLD
    rating_threshold: float = RATING_THRESHOLD
    min_minutes: int = MIN_MINUTES_DEFAULT
    fuzzy_score_threshold: int = FUZZY_SCORE_THRESHOLD


def _normalize_fbref_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()

    rename_map = {}
    if "player" in df.columns:
        rename_map["player"] = "player_name"
    elif "Player" in df.columns:
        rename_map["Player"] = "player_name"

    if "team" in df.columns:
        rename_map["team"] = "team_name"
    elif "Squad" in df.columns:
        rename_map["Squad"] = "team_name"

    if "Min" in df.columns:
        rename_map["Min"] = "minutes"

    df = df.rename(columns=rename_map)

    prog_candidates = ["ProgP", "Prog", "Prog_Pass"]
    final_third_candidates = ["1/3", "Final 1/3", "Final_Third"]

    def pick_column(candidates: Iterable[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"Expected one of {candidates} in FBref passing data; "
            f"available columns: {list(df.columns)}"
        )

    prog_col = pick_column(prog_candidates)
    final_third_col = pick_column(final_third_candidates)

    df = df.rename(
        columns={
            prog_col: "prog_passes_per90",
            final_third_col: "final_third_passes_per90",
        }
    )

    required = ["player_name", "team_name", "minutes",
                "prog_passes_per90", "final_third_passes_per90"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required FBref columns after normalization: {missing}")

    return df[required + [c for c in df.columns if c not in required]]


def get_fbref_passing_stats(
    config: GapAnalysisConfig | None = None,
) -> pd.DataFrame:
    cfg = config or GapAnalysisConfig()
    fbref = sd.FBref(leagues=cfg.league, seasons=cfg.season)

    df = fbref.read_player_season_stats(stat_type="passing_adv")
    df = _normalize_fbref_columns(df)

    df = df[df["minutes"] >= cfg.min_minutes].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def get_sofascore_ratings(
    config: GapAnalysisConfig | None = None,
) -> pd.DataFrame:
    cfg = config or GapAnalysisConfig()
    sofa = sd.Sofascore(leagues=cfg.league, seasons=cfg.season)

    schedule = sofa.read_schedule()

    rating_cols = [c for c in schedule.columns if "rating" in c.lower()]
    if not rating_cols:
        raise ValueError(
            "Sofascore schedule from soccerdata does not expose player-level ratings. "
            "Check soccerdata.Sofascore documentation for rating availability."
        )

    if "player" not in schedule.columns and "Player" not in schedule.columns:
        raise ValueError(
            "Sofascore data does not contain a player column; cannot aggregate ratings "
            "per player. Available columns: "
            f"{list(schedule.columns)}"
        )

    df = schedule.copy()
    if "player" in df.columns:
        df = df.rename(columns={"player": "player_name"})
    else:
        df = df.rename(columns={"Player": "player_name"})

    if "team" in df.columns:
        df = df.rename(columns={"team": "team_name"})
    elif "Team" in df.columns:
        df = df.rename(columns={"Team": "team_name"})

    rating_col = rating_cols[0]

    group_cols: List[str] = ["player_name"]
    if "team_name" in df.columns:
        group_cols.append("team_name")

    agg = (
        df.groupby(group_cols)[rating_col]
        .mean()
        .reset_index()
        .rename(columns={rating_col: "avg_rating"})
    )

    if "minutes" in df.columns:
        minutes_agg = (
            df.groupby(group_cols)["minutes"].sum().reset_index()["minutes"]
        )
        agg["minutes"] = minutes_agg

    agg["matches_played"] = df.groupby(group_cols).size().values

    return agg


def apply_player_name_mapping(
    df: pd.DataFrame,
    name_col: str = "player_name",
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    mapping = mapping or PLAYER_NAME_MAP
    if not mapping or name_col not in df.columns:
        return df

    out = df.copy()
    out[name_col] = out[name_col].replace(mapping)
    return out


def _build_fuzzy_mapping(
    source_names: Iterable[str],
    target_names: Iterable[str],
    score_threshold: int,
) -> Dict[str, str]:
    if process is None or fuzz is None:
        return {}

    target_list = list(dict.fromkeys(target_names))
    mapping: Dict[str, str] = {}

    for name in source_names:
        if name in target_list:
            continue

        match: Optional[Tuple[str, int, int]] = process.extractOne(
            name, target_list, scorer=fuzz.WRatio
        )
        if not match:
            continue
        best_match, score, _ = match
        if score >= score_threshold:
            mapping[name] = best_match

    return mapping


def fuzzy_match_players(
    fbref_df: pd.DataFrame,
    sofa_df: pd.DataFrame,
    config: GapAnalysisConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or GapAnalysisConfig()

    if "player_name" not in fbref_df.columns or "player_name" not in sofa_df.columns:
        return fbref_df, sofa_df

    fbref_names = fbref_df["player_name"].astype(str)
    sofa_names = sofa_df["player_name"].astype(str)

    exact_targets = set(sofa_names)
    to_match = [n for n in fbref_names.unique() if n not in exact_targets]

    fuzzy_mapping = _build_fuzzy_mapping(
        source_names=to_match,
        target_names=sofa_names.unique(),
        score_threshold=cfg.fuzzy_score_threshold,
    )

    if not fuzzy_mapping:
        fbref_df = fbref_df.copy()
        fbref_df["matched_fuzzily"] = False
        return fbref_df, sofa_df

    fbref_df = fbref_df.copy()
    fbref_df["matched_fuzzily"] = fbref_df["player_name"].isin(
        fuzzy_mapping.keys())
    fbref_df["player_name"] = fbref_df["player_name"].replace(fuzzy_mapping)

    return fbref_df, sofa_df


def merge_stats_and_ratings(
    passing_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    for col in ("player_name", "team_name"):
        if col not in passing_df.columns:
            raise ValueError(
                f"Passing data is missing required column '{col}'")

    if "player_name" not in ratings_df.columns:
        raise ValueError(
            "Ratings data is missing required column 'player_name'")

    on_cols: List[str] = ["player_name"]
    if "team_name" in ratings_df.columns:
        on_cols.append("team_name")

    merged = passing_df.merge(
        ratings_df,
        how="left",
        on=on_cols,
        suffixes=("_passing", "_ratings"),
    )

    merged = merged[~merged["avg_rating"].isna()].copy()
    merged.reset_index(drop=True, inplace=True)
    return merged


def perform_gap_analysis(
    merged_df: pd.DataFrame,
    config: GapAnalysisConfig | None = None,
) -> pd.DataFrame:
    cfg = config or GapAnalysisConfig()

    required_cols = ["prog_passes_per90", "avg_rating"]
    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for gap analysis: {missing}")

    df = merged_df.copy()
    if "minutes" in df.columns:
        df = df[df["minutes"] >= cfg.min_minutes]

    mask = (df["prog_passes_per90"] > cfg.prog_pass_threshold) & (
        df["avg_rating"] < cfg.rating_threshold
    )
    df = df[mask].copy()

    cols_order: List[str] = []
    for c in (
        "player_name",
        "team_name",
        "prog_passes_per90",
        "final_third_passes_per90",
        "avg_rating",
        "minutes",
        "matches_played",
        "matched_fuzzily",
    ):
        if c in df.columns:
            cols_order.append(c)

    df = df[cols_order]
    df = df.sort_values(
        by=["prog_passes_per90", "avg_rating"],
        ascending=[False, True],
    )
    df.reset_index(drop=True, inplace=True)
    return df


def run_gap_analysis(
    config: GapAnalysisConfig | None = None,
) -> pd.DataFrame:
    cfg = config or GapAnalysisConfig()

    passing_df = get_fbref_passing_stats(cfg)
    ratings_df = get_sofascore_ratings(cfg)

    passing_df = apply_player_name_mapping(passing_df, mapping=PLAYER_NAME_MAP)
    ratings_df = apply_player_name_mapping(ratings_df, mapping=PLAYER_NAME_MAP)

    passing_df, ratings_df = fuzzy_match_players(passing_df, ratings_df, cfg)

    merged = merge_stats_and_ratings(passing_df, ratings_df)
    result = perform_gap_analysis(merged, cfg)
    return result


__all__ = [
    "GapAnalysisConfig",
    "get_fbref_passing_stats",
    "get_sofascore_ratings",
    "apply_player_name_mapping",
    "fuzzy_match_players",
    "merge_stats_and_ratings",
    "perform_gap_analysis",
    "run_gap_analysis",
]

#Test1