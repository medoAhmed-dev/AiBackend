"""
Renders a pre-visit summary request into the text the model reads.

Mirrors services/symptom_intake.py: the whole brief is written in the
target language, because a directive in one language wrapped around
labels in another loses — the labels win.
"""

from schemas.previsit import PreVisitSummaryRequest
from services import safety

_LABELS = {
    "en": {
        "header": "Pre-visit information:",
        "reason": "Reason for visit",
        "symptoms": "Reported symptoms",
        "duration": "Duration",
        "severity": "Severity as described by the patient",
        "age": "Age",
        "sex": "Sex",
        "conditions": "Known conditions",
        "allergies": "Allergies",
        "medications": "Current medications",
        "notes": "Additional notes from the patient",
        "under_a_day": "less than a day",
        "one_day": "1 day",
        "days": "{n:g} days",
        "severity_values": {"mild": "mild", "moderate": "moderate", "severe": "severe"},
        "sex_values": {"male": "male", "female": "female", "other": "other"},
    },
    "ar": {
        "header": "معلومات ما قبل الزيارة:",
        "reason": "سبب الزيارة",
        "symptoms": "الأعراض المذكورة",
        "duration": "المدة",
        "severity": "شدة الأعراض كما وصفها المريض",
        "age": "العمر",
        "sex": "الجنس",
        "conditions": "الحالات المرضية المعروفة",
        "allergies": "الحساسية",
        "medications": "الأدوية الحالية",
        "notes": "ملاحظات إضافية من المريض",
        "under_a_day": "أقل من يوم",
        "one_day": "يوم واحد",
        "days": "{n:g} أيام",
        "severity_values": {"mild": "خفيفة", "moderate": "متوسطة", "severe": "شديدة"},
        "sex_values": {"male": "ذكر", "female": "أنثى", "other": "آخر"},
    },
}


def scannable_text(request: PreVisitSummaryRequest) -> str:
    """Everything the patient wrote, for language detection and the
    red-flag scan."""
    parts = [request.reason_for_visit, *request.symptoms]
    if request.additional_notes:
        parts.append(request.additional_notes)
    return " ".join(parts)


def _duration_text(days: float, labels: dict) -> str:
    if days < 1:
        return labels["under_a_day"]
    if days == 1:
        return labels["one_day"]
    return labels["days"].format(n=days)


def render(request: PreVisitSummaryRequest) -> str:
    labels = _LABELS[safety.detect_language(scannable_text(request))]

    lines = [
        labels["header"],
        f"- {labels['reason']}: {request.reason_for_visit}",
    ]

    if request.symptoms:
        lines.append(f"- {labels['symptoms']}: {', '.join(request.symptoms)}")
    if request.duration_days is not None:
        lines.append(f"- {labels['duration']}: {_duration_text(request.duration_days, labels)}")
    if request.severity:
        lines.append(f"- {labels['severity']}: {labels['severity_values'][request.severity]}")
    if request.age is not None:
        lines.append(f"- {labels['age']}: {request.age}")
    if request.sex:
        lines.append(f"- {labels['sex']}: {labels['sex_values'][request.sex]}")
    if request.medical_conditions:
        lines.append(f"- {labels['conditions']}: {request.medical_conditions}")
    if request.allergies:
        lines.append(f"- {labels['allergies']}: {request.allergies}")
    if request.current_medications:
        lines.append(f"- {labels['medications']}: {request.current_medications}")
    if request.additional_notes:
        lines.append(f"- {labels['notes']}: {request.additional_notes}")

    return "\n".join(lines)
