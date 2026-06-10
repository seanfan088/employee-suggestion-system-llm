import json
import os
import sys

import pandas as pd

RUNTIME_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNTIME_ROOT)

EXCEL_PATH = r'C:\Users\364328\Desktop\2025 ESS simple.xlsx'
BASE_RECORDS_PATH = os.path.join(RUNTIME_ROOT, 'data', 'base_records.jsonl')
DELTA_RECORDS_PATH = os.path.join(RUNTIME_ROOT, 'data', 'delta_records.jsonl')
FEEDBACK_LOG_PATH = os.path.join(RUNTIME_ROOT, 'data', 'submitted_ai_feedback.jsonl')
PREDICTION_LOG_PATH = os.path.join(RUNTIME_ROOT, 'data', 'prediction_results.jsonl')

FIELD_MAP = {
    'SN': 'sn', 'Gid': 'gid', 'Name': 'name', 'Title': 'title',
    'Department': 'department', 'Tel': 'tel', 'LaborType': 'labor_type',
    'Shift': 'shift', 'UserAreaName': 'user_area_name',
    'ProblemAreaName': 'problem_area_name', 'LocationName': 'location_name',
    'ManagerGid': 'manager_gid', 'ManagerName': 'manager_name',
    'SubmissionDate': 'submission_date', 'LeanFlowName': 'lean_flow_name',
    'Description': 'description', 'Suggestion': 'suggestion',
    'ReplyOpinion': 'reply_opinion', 'RejectJustification': 'reject_justification',
    'Status': 'status', 'ItemType': 'item_type', 'GoodRequest': 'good_request',
    'OwnerGid': 'owner_gid', 'OwnerName': 'owner_name', 'OwnerTel': 'owner_tel',
    'OwnerManagerGid': 'owner_manager_gid', 'OwnerManagerName': 'owner_manager_name',
    'Score': 'score', 'IsRepeat': 'is_repeat', 'AreaType': 'area_type',
}

REVERSE_FIELD_MAP = {v: k for k, v in FIELD_MAP.items()}


def map_row(row):
    mapped = {}
    for excel_key, db_key in FIELD_MAP.items():
        val = row.get(excel_key)
        if isinstance(val, float) and pd.isna(val):
            val = None
        mapped[db_key] = val
    return mapped


def migrate_excel():
    if not os.path.exists(EXCEL_PATH):
        print(f'Excel not found: {EXCEL_PATH}')
        return 0

    df = pd.read_excel(EXCEL_PATH, engine='openpyxl').fillna('')
    records = [map_row(row) for _, row in df.iterrows()]
    print(f'Excel records: {len(records)}')
    return records


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def migrate_jsonl():
    return read_jsonl(BASE_RECORDS_PATH), read_jsonl(DELTA_RECORDS_PATH)


def migrate_feedback():
    return read_jsonl(FEEDBACK_LOG_PATH)


def migrate_prediction_logs():
    return read_jsonl(PREDICTION_LOG_PATH)


def main():
    from db import get_session, Suggestion, AiFeedback, PredictionLog

    session = get_session()

    print('=== Migrating suggestions from Excel ===')
    excel_records = migrate_excel()
    if excel_records:
        for rec in excel_records:
            session.add(Suggestion(**rec))
        session.commit()
        print(f'  Imported {len(excel_records)} suggestions from Excel')

    base_raw, delta_raw = migrate_jsonl()
    print(f'  Skipping JSONL (same source as Excel): {len(base_raw)} base + {len(delta_raw)} delta')

    print('\n=== Migrating AI feedback ===')
    feedback_records = migrate_feedback()
    for rec in feedback_records:
        session.add(AiFeedback(
            submission_id=rec.get('submissionId', ''),
            status=rec.get('status', 'committed'),
            retry_key=rec.get('retryKey', ''),
            gid=rec.get('gid', ''),
            description=rec.get('description', ''),
            suggestion=rec.get('suggestion', ''),
            reply_opinion=rec.get('replyOpinion', ''),
            predicted_fields=rec.get('predictedFields'),
            retrieved_cases=rec.get('retrievedCases'),
            generated_ai_suggestions=rec.get('generatedAiSuggestions'),
            accepted_suggestion_ids=rec.get('acceptedSuggestionIds'),
            edited_ai_suggestions=rec.get('editedAiSuggestions'),
            discarded_suggestion_ids=rec.get('discardedSuggestionIds'),
            images=rec.get('images'),
            submitted_at=rec.get('submittedAt', ''),
        ))
    session.commit()
    print(f'  Imported {len(feedback_records)} AI feedback records')

    print('\n=== Migrating prediction logs ===')
    prediction_records = migrate_prediction_logs()
    for rec in prediction_records:
        session.add(PredictionLog(
            gid=rec.get('gid', ''),
            description_length=rec.get('descriptionLength'),
            suggestion_length=rec.get('suggestionLength'),
            similarity=rec.get('similarity'),
            predicted_fields=rec.get('predictedFields', rec),
            predicted_at=rec.get('predictedAt', ''),
        ))
    session.commit()
    print(f'  Imported {len(prediction_records)} prediction logs')

    total = session.query(Suggestion).count()
    print(f'\n=== Done! Total suggestions in DB: {total} ===')
    session.close()


if __name__ == '__main__':
    main()
