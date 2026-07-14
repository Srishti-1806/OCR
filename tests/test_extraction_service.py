import json

from app.services.extraction_service import _salvage_truncated_json


def test_salvage_truncated_json_handles_unescaped_quotes_and_truncated_payload():
    raw = '''{
  "fields": [
    {
      "label": "city",
      "value": "Ha?isburg",
      "confidence": 0.86,
      "field_type": "text"
    },
    {
      "label": "state",
      "value": "\"",
      "confidence": 0.83,
      "field_type": "text"
    }
  ],
  "warnings": ["done"]
'''

    repaired = _salvage_truncated_json(raw)
    parsed = json.loads(repaired)

    assert parsed["fields"][0]["label"] == "city"
    assert parsed["fields"][1]["value"] == '"'
    assert parsed["warnings"] == ["done"]
