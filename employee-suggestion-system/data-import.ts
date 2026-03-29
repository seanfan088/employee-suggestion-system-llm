import { readFile } from 'fs/promises';
import { basename } from 'path';
import { SuggestionSchema, Suggestion } from './types';

export interface ImportResult {
  success: boolean;
  imported: number;
  errors: string[];
  data?: Suggestion[];
}

export async function importFromExcel(filePath: string): Promise<ImportResult> {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);
  const path = await import('path');

  const scriptPath = path.join(process.cwd(), 'read_excel.py');
  const { stdout } = await execAsync(`python "${scriptPath}" "${filePath}"`);
  const rawData = JSON.parse(stdout.trim());
  
  const validData: Suggestion[] = [];
  const errors: string[] = [];

  for (let i = 0; i < rawData.length; i++) {
    try {
      const row = rawData[i];
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
      errors.push(`Row ${i + 1}: ${e}`);
    }
  }

  return {
    success: errors.length === 0,
    imported: validData.length,
    errors,
    data: validData,
  };
}

export interface ExcelData extends ImportResult {
  data: Suggestion[];
}

export async function appendData(filePath: string, existingData: Suggestion[]): Promise<ImportResult> {
  const result = await importFromExcel(filePath) as ExcelData;
  
  if (result.data.length > 0) {
    existingData.push(...result.data);
  }
  
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);
  const path = await import('path');

  const scriptPath = path.join(process.cwd(), 'write_excel.py');
  const dataJson = JSON.stringify(existingData);
  const outputPath = 'C:\\Users\\364328\\Desktop\\2025 ESS simple.xlsx';
  
  await execAsync(`python "${scriptPath}" "${dataJson}" "${outputPath}"`);
  
  return {
    success: result.success,
    imported: result.imported,
    errors: result.errors,
  };
}

export function getDataSummary(data: Suggestion[]): {
  total: number;
  departments: string[];
  laborTypes: string[];
  itemTypes: string[];
  areaTypes: string[];
} {
  const departments = [...new Set(data.map(d => d.Department).filter(Boolean))];
  const laborTypes = [...new Set(data.map(d => d.LaborType).filter(Boolean))];
  const itemTypes = [...new Set(data.map(d => d.ItemType).filter(Boolean))];
  const areaTypes = [...new Set(data.map(d => d.AreaType).filter(Boolean))];

  return {
    total: data.length,
    departments,
    laborTypes,
    itemTypes,
    areaTypes,
  };
}
