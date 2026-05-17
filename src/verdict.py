"""
Deterministic verdict computation from critic reconciled scores.
Applies thresholds from config.yaml. Always returns a verdict; never None.
"""
import logging
from typing import Optional

from src.state import CriticOutput, FinalVerdict, State

logger = logging.getLogger("tara.verdict")

CONTENT_SAFETY_SUBVARS = [
    "violence_aggression",
    "scary_fear_inducing",
    "inappropriate_language",
    "bullying_exclusion",
    "stereotyping",
    "fantasy_level",
    "negative_behavior_modeling",
]


def _extract_reconciled_subvar_score(critic: CriticOutput, dimension: str, subvar: str) -> Optional[float]:
    try:
        return float(critic.reconciled_scores[dimension][subvar]["reconciled_score"])
    except (KeyError, TypeError, ValueError):
        return None


def _fallback_content_safety_scores(scorer_outputs: list) -> list[float]:
    scores = []
    for subvar in CONTENT_SAFETY_SUBVARS:
        values = []
        for scorer in scorer_outputs:
            item = scorer.content_safety.get(subvar)
            if item is not None:
                values.append(float(item.score))
        if values:
            scores.append(sum(values) / len(values))
    return scores


def compute_final_verdict(state: State) -> State:
    from src.utils import get_config

    thresholds = get_config()["thresholds"]
    approved_overall = thresholds["approved_overall"]
    approved_cs_min = thresholds["approved_cs_minimum"]
    approved_subvar_floor = thresholds["approved_subvar_floor"]
    rejected_overall = thresholds["rejected_overall"]
    rejected_subvar_floor = thresholds["rejected_subvar_floor"]

    # Can't score if both scorers failed. A bootstrap critic with no scorer
    # evidence would otherwise turn missing evidence into a misleading low score.
    if state.scorer_gemini is None and state.scorer_claude is None:
        state.final_verdict = FinalVerdict(
            verdict="ERROR",
            overall_score=0.0,
            confidence="LOW",
            reasoning="All scoring agents failed; cannot produce verdict.",
            key_decision_drivers=["all_scorers_failed"],
            error_reason="No scorer output available.",
        )
        return state

    if state.critic is not None:
        overall_score = float(state.critic.overall.get("weighted_overall_score", 0.0))
        confidence = state.critic.overall.get("critic_confidence", "LOW")
        key_drivers = [
            d if isinstance(d, str) else d.get("dimension", str(d)) if isinstance(d, dict) else str(d)
            for d in state.critic.overall.get("key_decision_drivers", [])
        ]
        dim_scores = state.critic.dimension_scores
        cs_subvar_scores = []
        for subvar in CONTENT_SAFETY_SUBVARS:
            score = _extract_reconciled_subvar_score(state.critic, "content_safety", subvar)
            if score is not None:
                cs_subvar_scores.append(score)
        cs_dimension_score = dim_scores.get("content_safety", 0.0)
    else:
        scorer_outputs = [s for s in [state.scorer_gemini, state.scorer_claude] if s is not None]
        overall_score = sum(s.overall_score for s in scorer_outputs) / len(scorer_outputs)
        confidence = "LOW"
        key_drivers = ["critic_unavailable_using_scorer_average"]
        cs_subvar_scores = _fallback_content_safety_scores(scorer_outputs)
        cs_dimension_score = sum(cs_subvar_scores) / len(cs_subvar_scores) if cs_subvar_scores else overall_score

    if state.critic is not None and all(v > 60 for v in state.critic.dimension_scores.values()):
        overall_score = min(100.0, overall_score + 25.0)
        key_drivers = list(key_drivers) + ["consistency_bonus_applied"]

    min_cs_subvar = min(cs_subvar_scores) if cs_subvar_scores else cs_dimension_score

    if (
        overall_score >= approved_overall
        and cs_dimension_score >= approved_cs_min
        and min_cs_subvar >= approved_subvar_floor
    ):
        verdict = "APPROVED"
        reasoning = (
            f"Overall score {overall_score:.1f} >= {approved_overall}, "
            f"content safety {cs_dimension_score:.1f} >= {approved_cs_min}, "
            f"all CS sub-variables >= {approved_subvar_floor}."
        )
    elif overall_score < rejected_overall or min_cs_subvar < rejected_subvar_floor:
        verdict = "REJECTED"
        drivers = []
        if overall_score < rejected_overall:
            drivers.append(f"overall score {overall_score:.1f} < {rejected_overall}")
        if min_cs_subvar < rejected_subvar_floor:
            drivers.append(f"CS sub-variable {min_cs_subvar:.1f} < {rejected_subvar_floor}")
        reasoning = "Rejected: " + "; ".join(drivers) + "."
    else:
        verdict = "REVIEW"
        reasoning = (
            f"Overall {overall_score:.1f}, CS dimension {cs_dimension_score:.1f}. "
            "Does not meet approval thresholds but not clearly rejected."
        )

    state.final_verdict = FinalVerdict(
        verdict=verdict,
        overall_score=round(overall_score, 2),
        confidence=confidence,
        reasoning=reasoning,
        key_decision_drivers=list(key_drivers),
    )

    logger.info(f"Verdict: {verdict} (score={overall_score:.1f}, cs={cs_dimension_score:.1f})")
    return state
