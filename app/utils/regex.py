"""
Central place for regex patterns and the known-label keyword dictionary.
Extend LABEL_KEYWORDS with whatever fields your forms typically contain.
"""
import re

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\+]?[0-9][0-9\s\-\(\)]{7,14}[0-9]$")
DATE_RE = re.compile(
    r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})$|^(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})$"
)
PINCODE_RE = re.compile(r"^\d{5,6}$")
AADHAAR_RE = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")
PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

# label text -> normalized key. Matching is done fuzzily (see label_detector.py)
# so slight OCR noise ("Nam3:", "D.O.B") will still resolve correctly.
LABEL_KEYWORDS = {
    "name": "name",
    "full name": "name",
    "first name": "first_name",
    "last name": "last_name",
    "father name": "father_name",
    "father's name": "father_name",
    "mother name": "mother_name",
    "date of birth": "date_of_birth",
    "dob": "date_of_birth",
    "gender": "gender",
    "sex": "gender",
    "address": "address",
    "permanent address": "permanent_address",
    "current address": "current_address",
    "city": "city",
    "state": "state",
    "country": "country",
    "pincode": "pincode",
    "pin code": "pincode",
    "zip code": "pincode",
    "phone": "phone",
    "phone number": "phone",
    "mobile": "phone",
    "mobile number": "phone",
    "contact number": "phone",
    "email": "email",
    "email id": "email",
    "e-mail": "email",
    "aadhaar number": "aadhaar_number",
    "aadhar number": "aadhaar_number",
    "pan number": "pan_number",
    "pan": "pan_number",
    "date": "date",
    "signature": "signature",
    "occupation": "occupation",
    "nationality": "nationality",
    "marital status": "marital_status",
    "employer": "employer",
    "designation": "designation",
    "amount": "amount",
    "account number": "account_number",
    "account no": "account_number",
    "ifsc code": "ifsc_code",
    "bank name": "bank_name",
}

# Field-type-specific validators keyed by normalized label
FIELD_VALIDATORS = {
    "email": EMAIL_RE,
    "phone": PHONE_RE,
    "date_of_birth": DATE_RE,
    "date": DATE_RE,
    "pincode": PINCODE_RE,
    "aadhaar_number": AADHAAR_RE,
    "pan_number": PAN_RE,
}

CHECKBOX_CHECKED_SYMBOLS = {"☑", "✓", "✔", "☒", "X", "x"}
CHECKBOX_UNCHECKED_SYMBOLS = {"☐", "□", "○"}
