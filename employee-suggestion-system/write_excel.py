import pandas as pd
import json
import sys
import os

if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding='utf-8')

json_data = sys.argv[1]
output_path = sys.argv[2]

data = json.loads(json_data)
df = pd.DataFrame(data)
df.to_excel(output_path, index=False)
print('Saved successfully')
