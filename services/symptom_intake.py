"""
Turns a structured SymptomCheckRequest into the text the model reads, and
into the text the safety layer scans.

Kept separate from the endpoint so both the rendering and the red-flag
scanning are testable on their own.
"""

from schemas.symptom_check import SymptomCheckRequest
from services import safety

# The model is told which language to answer in rather than left to infer it.
# Asked to infer, it answered English reports in Arabic every time. The whole
# report is then rendered in that same language: when the directive said
# Arabic but the surrounding labels were English, the English won.
_LABELS = {
    "en": {
        "directive": "Write your entire response in English.",
        "header": "Symptom report:",
        "symptoms": "Symptoms",
        "duration": "Duration",
        "severity": "Severity as described by the patient",
        "age": "Age",
        "sex": "Sex",
        "notes": "Additional notes from the patient",
        "under_a_day": "less than a day",
        "one_day": "1 day",
        "days": "{n:g} days",
        "severity_values": {"mild": "mild", "moderate": "moderate", "severe": "severe"},
        "sex_values": {"male": "male", "female": "female", "other": "other"},
    },
    "ar": {
        "directive": "اكتب ردك بالكامل باللغة العربية.",
        "header": "تقرير الأعراض:",
        "symptoms": "الأعراض",
        "duration": "المدة",
        "severity": "شدة الأعراض كما وصفها المريض",
        "age": "العمر",
        "sex": "الجنس",
        "notes": "ملاحظات إضافية من المريض",
        "under_a_day": "أقل من يوم",
        "one_day": "يوم واحد",
        "days": "{n:g} أيام",
        "severity_values": {"mild": "خفيفة", "moderate": "متوسطة", "severe": "شديدة"},
        "sex_values": {"male": "ذكر", "female": "أنثى", "other": "آخر"},
    },
}


def _duration_text(days: float, labels: dict) -> str:
    if days < 1:
        return labels["under_a_day"]
    if days == 1:
        return labels["one_day"]
    return labels["days"].format(n=days)


def render(request: SymptomCheckRequest) -> str:
    """A compact, readable intake report. Optional fields are omitted
    entirely rather than sent as 'unknown', so the model isn't nudged into
    reasoning about absent data."""
    labels = _LABELS[safety.detect_language(scannable_text(request))]

    lines = [
        labels["directive"],
        "",
        labels["header"],
        f"- {labels['symptoms']}: {', '.join(request.symptoms)}",
    ]

    if request.duration_days is not None:
        lines.append(f"- {labels['duration']}: {_duration_text(request.duration_days, labels)}")
    if request.severity:
        lines.append(f"- {labels['severity']}: {labels['severity_values'][request.severity]}")
    if request.age is not None:
        lines.append(f"- {labels['age']}: {request.age}")
    if request.sex:
        lines.append(f"- {labels['sex']}: {labels['sex_values'][request.sex]}")
    if request.additional_notes:
        lines.append(f"- {labels['notes']}: {request.additional_notes}")

    # Repeated at the end as well: a trailing instruction carries more weight
    # than one buried above the content.
    lines += ["", labels["directive"]]

    return "\n".join(lines)


def scannable_text(request: SymptomCheckRequest) -> str:
    """Everything the patient actually wrote, for the red-flag keyword scan.
    Structured symptoms and free-text notes both have to be checked — an
    emergency can be reported through either."""
    parts = list(request.symptoms)
    if request.additional_notes:
        parts.append(request.additional_notes)
    return " ".join(parts)
