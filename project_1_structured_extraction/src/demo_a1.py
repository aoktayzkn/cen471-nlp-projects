import json, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.src.parser import clean_raw_output

FENCE = chr(96) * 3
raw = FENCE + 'json\n{"sentiment": "positive"}\n' + FENCE
print("cleaned:", repr(clean_raw_output(raw)))
print(json.loads(clean_raw_output(raw)))