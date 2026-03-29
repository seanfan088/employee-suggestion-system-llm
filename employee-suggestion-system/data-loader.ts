import { readFile } from 'fs/promises';
import { parse } from 'path';
import { Suggestion, SuggestionSchema, OutputFields } from './types';

let cachedData: Suggestion[] | null = null;

export async function loadSuggestionData(): Promise<Suggestion[]> {
  if (cachedData) {
    return cachedData;
  }

  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);

  const excelPath = 'C:\\Users\\364328\\Desktop\\2025 ESS simple.xlsx';
  const script = `
import pandas as pd
import json
df = pd.read_excel(r'${excelPath}')
records = df.to_dict('records')
print(json.dumps(records, ensure_ascii=False, default=str))
`;

  const { stdout } = await execAsync(`python -c "${script.replace(/"/g, '\\"').replace(/\n/g, ' ')}"`);
  const rawData = JSON.parse(stdout.trim());
  
  const validData: Suggestion[] = [];
  for (const row of rawData) {
    try {
      const cleaned: Record<string, unknown> = {};
      for (const key of Object.keys(row)) {
        if (row[key] !== null && row[key] !== undefined) {
          cleaned[key] = String(row[key]);
        } else {
          cleaned[key] = '';
        }
      }
      validData.push(SuggestionSchema.parse(cleaned));
    } catch (e) {
      console.warn('Skipping invalid row:', e);
    }
  }

  cachedData = validData;
  return validData;
}

export function getTextForEmbedding(record: Suggestion): string {
  return `员工ID: ${record.Gid}, 姓名: ${record.Name}, 描述: ${record.Description}, 建议: ${record.Suggestion}`;
}

export function getOutputFields(record: Suggestion): Record<string, string> {
  const result: Record<string, string> = {};
  for (const field of OutputFields) {
    result[field] = record[field] as string;
  }
  return result;
}
