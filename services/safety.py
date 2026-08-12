import logging

logger = logging.getLogger("medical_chat.safety")

# Hardcoded red-flag symptoms, English and Arabic (incl. common colloquial
# phrasing, not just formal MSA). If any of these appear in the patient's
# message, urgency is forced to "red" regardless of what the model returns.
RED_FLAG_KEYWORDS = [
    # English
    "chest pain",
    "crushing chest",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "difficulty breathing",
    "not breathing",
    "unconscious",
    "unresponsive",
    "severe bleeding",
    "uncontrolled bleeding",
    "won't stop bleeding",
    "stroke",
    "slurred speech",
    "face drooping",
    "sudden numbness",
    "suicidal",
    "overdose",
    # Arabic
    "ألم في الصدر",
    "ألم شديد في الصدر",
    "وجع في الصدر",
    "في الصدر",
    "لا أستطيع التنفس",
    "مش قادر اتنفس",
    "مش قادرة اتنفس",
    "مقدرش اتنفس",
    "صعوبة في التنفس",
    "لا يتنفس",
    "فقدان الوعي",
    "فاقد الوعي",
    "نزيف حاد",
    "نزيف شديد",
    "النزيف لا يتوقف",
    "أعراض السكتة الدماغية",
    "سكتة دماغية",
    "تعثر الكلام",
    "خدر مفاجئ",
    "أفكار انتحارية",
    "جرعة زائدة",
]

# Phrases that indicate the patient is asking for something outside safe
# scope (a specific diagnosis, a drug dosage, a prescription). These are
# logged for audit, not blocked.
OUT_OF_SCOPE_KEYWORDS = [
    "what dose",
    "how many mg",
    "how much mg",
    "what dosage",
    "diagnose me",
    "what disease do i have",
    "what's wrong with me",
    "prescribe",
    "prescription for",
]

DISCLAIMER_EN = (
    "This is general guidance, not a medical diagnosis. Please confirm with a doctor."
)
DISCLAIMER_AR = "هذا إرشاد عام وليس تشخيصًا طبيًا، يرجى التأكد من الطبيب."

EMERGENCY_REPLY_EN = (
    "Your message describes symptoms that may be a medical emergency. "
    "Please seek immediate/emergency care now — call your local emergency "
    "number or go to the nearest emergency room. This is guidance, not a "
    "diagnosis; a doctor should confirm."
)
EMERGENCY_REPLY_AR = (
    "تشير رسالتك إلى أعراض قد تكون حالة طبية طارئة. "
    "يرجى طلب الرعاية الطارئة فورًا الآن — اتصل برقم الطوارئ المحلي "
    "أو توجه إلى أقرب قسم طوارئ. هذا إرشاد عام وليس تشخيصًا؛ يجب أن يؤكده الطبيب."
)


def _is_arabic(text: str) -> bool:
    arabic_chars = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return arabic_chars > latin_chars


def detect_red_flag(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in RED_FLAG_KEYWORDS)


def emergency_reply_for(message: str) -> str:
    return EMERGENCY_REPLY_AR if _is_arabic(message) else EMERGENCY_REPLY_EN


def flag_out_of_scope(message: str, patient_id: str) -> None:
    lowered = message.lower()
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in lowered:
            logger.warning(
                "Out-of-scope request from patient_id=%s matched keyword=%r message=%r",
                patient_id,
                keyword,
                message,
            )
            return


def ensure_disclaimer(reply: str) -> str:
    if _is_arabic(reply):
        if "ليس تشخيص" in reply:
            return reply
        return f"{reply}\n\n{DISCLAIMER_AR}"
    if "not a diagnosis" in reply.lower() or "not a medical diagnosis" in reply.lower():
        return reply
    return f"{reply}\n\n{DISCLAIMER_EN}"
