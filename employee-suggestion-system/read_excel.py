import pandas as pd
import json
import sys
import os

if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding='utf-8')

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(json.dumps({"error": "File not found"}))
    sys.exit(1)

try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)

records = df.to_dict('records')
print(json.dumps(records, ensure_ascii=False, default=str))
