"""
Ranks candidate doctors for a patient.

Transparent weighted scoring rather than a learned model, deliberately:
there is no historical "which doctor did this patient pick, and were they
happy" data to train on, so a learned ranker would have nothing real to
learn from. Every score here can be explained in a sentence, which is what
the `reasons` field carries back to the app.

Weights sum to 100 across whichever signals are actually present, so a
doctor is never penalised for missing data the database doesn't hold yet.
"""

from schemas.doctor_recommendation import (
    DoctorCandidate,
    DoctorRecommendationRequest,
    ScoredDoctor,
)

# Relative importance. Normalised against whichever signals are available,
# so these are ratios rather than fixed point values.
_WEIGHTS = {
    "specialty": 35,
    "rating": 20,
    "experience": 15,
    "availability": 12,
    "distance": 10,
    "fee": 5,
    "verified": 3,
}

_EXPERIENCE_PLATEAU_YEARS = 20  # past this, more years stops counting for more
_DISTANCE_PLATEAU_KM = 25.0  # past this, everything is equally "far"


def _specialty_matches(doctor: DoctorCandidate, needed: str) -> bool:
    return needed.strip().lower() in doctor.specialty.strip().lower()


def _passes_filters(doctor: DoctorCandidate, request: DoctorRecommendationRequest) -> bool:
    """Hard requirements. A specialty mismatch is disqualifying — a
    dermatologist should never surface as the best option for a cardiology
    need, no matter how well they score elsewhere."""
    if request.needed_specialty and not _specialty_matches(doctor, request.needed_specialty):
        return False
    if request.require_verified and doctor.is_verified is False:
        return False
    if (
        request.max_consultation_fee is not None
        and doctor.consultation_fee is not None
        and doctor.consultation_fee > request.max_consultation_fee
    ):
        return False
    return True


def _score(doctor: DoctorCandidate, request: DoctorRecommendationRequest) -> tuple[float, list[str]]:
    """Returns a 0-100 score and the human-readable reasons behind it.
    Each signal contributes 0.0-1.0 of its own weight; the total is divided
    by the weight actually available so scores stay comparable."""
    parts: list[tuple[str, float]] = []
    reasons: list[str] = []

    if request.needed_specialty:
        parts.append(("specialty", 1.0))
        reasons.append(f"Specialises in {doctor.specialty}")

    if doctor.rating is not None:
        parts.append(("rating", doctor.rating / 5.0))
        reasons.append(f"Rated {doctor.rating:g}/5")

    if doctor.years_of_experience is not None:
        normalised = min(doctor.years_of_experience, _EXPERIENCE_PLATEAU_YEARS) / _EXPERIENCE_PLATEAU_YEARS
        parts.append(("experience", normalised))
        reasons.append(f"{doctor.years_of_experience} years of experience")

    if doctor.available_today is not None:
        parts.append(("availability", 1.0 if doctor.available_today else 0.0))
        reasons.append("Available today" if doctor.available_today else "Not available today")

    if doctor.distance_km is not None:
        closeness = 1.0 - min(doctor.distance_km, _DISTANCE_PLATEAU_KM) / _DISTANCE_PLATEAU_KM
        parts.append(("distance", closeness))
        reasons.append(f"{doctor.distance_km:g} km away")

    if doctor.consultation_fee is not None:
        if request.max_consultation_fee:
            affordability = 1.0 - min(doctor.consultation_fee, request.max_consultation_fee) / request.max_consultation_fee
        else:
            # No budget given: mild preference for lower fees, flattening out
            # at a fee high enough that further increases stop mattering.
            affordability = 1.0 - min(doctor.consultation_fee, 1000.0) / 1000.0
        parts.append(("fee", affordability))
        reasons.append(f"Consultation fee {doctor.consultation_fee:g}")

    if doctor.is_verified is not None:
        parts.append(("verified", 1.0 if doctor.is_verified else 0.0))
        if doctor.is_verified:
            reasons.append("Verified doctor")

    available_weight = sum(_WEIGHTS[name] for name, _ in parts)
    if not available_weight:
        return 0.0, ["No comparable information available for this doctor"]

    earned = sum(_WEIGHTS[name] * value for name, value in parts)
    return round(100 * earned / available_weight, 1), reasons


def _signal_availability(request: DoctorRecommendationRequest) -> tuple[list[str], list[str]]:
    """Which signals any candidate actually carries — so the caller can see
    what the ranking could and couldn't weigh."""
    checks = {
        "specialty": lambda d: bool(request.needed_specialty),
        "rating": lambda d: d.rating is not None,
        "experience": lambda d: d.years_of_experience is not None,
        "availability": lambda d: d.available_today is not None,
        "distance": lambda d: d.distance_km is not None,
        "fee": lambda d: d.consultation_fee is not None,
        "verified": lambda d: d.is_verified is not None,
    }
    used, unavailable = [], []
    for name, present in checks.items():
        (used if any(present(d) for d in request.doctors) else unavailable).append(name)
    return used, unavailable


def recommend(request: DoctorRecommendationRequest) -> dict:
    eligible = [d for d in request.doctors if _passes_filters(d, request)]

    scored: list[ScoredDoctor] = []
    for doctor in eligible:
        score, reasons = _score(doctor, request)
        scored.append(
            ScoredDoctor(
                doctor_id=doctor.id,
                full_name=doctor.full_name,
                specialty=doctor.specialty,
                score=score,
                reasons=reasons,
                consultation_fee=doctor.consultation_fee,
                hospital_name=doctor.hospital_name,
            )
        )

    # Ties broken by name so the same input always gives the same order.
    scored.sort(key=lambda d: (-d.score, d.full_name))
    used, unavailable = _signal_availability(request)

    return {
        "recommendations": scored[: request.limit],
        "total_considered": len(request.doctors),
        "excluded_count": len(request.doctors) - len(eligible),
        "signals_used": used,
        "signals_unavailable": unavailable,
    }
