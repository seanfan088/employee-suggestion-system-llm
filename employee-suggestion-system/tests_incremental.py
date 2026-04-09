import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


APP_PATH = os.path.join(os.path.dirname(__file__), 'app.py')


def load_app_module(module_name='employee_suggestion_app_under_test'):
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IncrementalRuntimeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='incremental-runtime-')
        self.module = load_app_module(f'employee_suggestion_app_{id(self)}')
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        self.configure_runtime_paths()

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def configure_runtime_paths(self):
        runtime_root = os.path.join(self.temp_dir, 'runtime')
        self.module.DATA_DIR = os.path.join(runtime_root, 'data')
        self.module.IMPORTS_DIR = os.path.join(runtime_root, 'imports')
        self.module.ARCHIVE_DIR = os.path.join(self.module.DATA_DIR, 'archive')
        self.module.MODELS_DIR = os.path.join(runtime_root, 'models')
        self.module.BASE_RECORDS_PATH = os.path.join(self.module.DATA_DIR, 'base_records.jsonl')
        self.module.DELTA_RECORDS_PATH = os.path.join(self.module.DATA_DIR, 'delta_records.jsonl')
        self.module.FEEDBACK_LOG_PATH = os.path.join(self.module.DATA_DIR, 'submitted_ai_feedback.jsonl')
        self.module.PREDICTION_LOG_PATH = os.path.join(self.module.DATA_DIR, 'prediction_results.jsonl')
        self.module.TRAINING_META_PATH = os.path.join(self.module.MODELS_DIR, 'training_meta.json')
        self.module.BASE_MODEL_PATH = os.path.join(self.module.MODELS_DIR, 'base_model.pkl')
        self.module.INCREMENTAL_TFIDF_PATH = os.path.join(self.module.MODELS_DIR, 'incremental_tfidf.pkl')
        self.module.BASE_W2V_PATH = os.path.join(self.module.MODELS_DIR, 'base_w2v.model')
        self.module.MODEL_PATH = os.path.join(self.temp_dir, 'model.pkl')
        self.module.LEGACY_W2V_PATH = os.path.join(self.temp_dir, 'model_w2v.model')
        self.module.EXCEL_PATH = os.path.join(self.temp_dir, 'legacy.xlsx')

    def write_jsonl(self, path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def read_jsonl(self, path):
        with open(path, 'r', encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def build_legacy_dataframe(self):
        return pd.DataFrame([
            {
                'Gid': 'g001',
                'Description': 'Legacy description',
                'Suggestion': 'Legacy suggestion',
                'ReplyOpinion': 'Legacy reply',
                'Department': 'Dept A',
                'ItemType': 'Type A',
                'SubmissionDate': '2026-03-07 09:00:00',
            },
            {
                'Gid': 'g002',
                'Description': 'Second description',
                'Suggestion': 'Second suggestion',
                'ReplyOpinion': '',
                'Department': 'Dept B',
                'ItemType': 'Type B',
                'SubmissionDate': '2026-03-07 10:00:00',
            },
        ]).fillna('')

    def write_legacy_state(self):
        with open(self.module.MODEL_PATH, 'wb') as handle:
            pickle.dump({'legacy': True}, handle)

        with open(self.module.LEGACY_W2V_PATH, 'wb') as handle:
            handle.write(b'legacy-w2v')

        self.build_legacy_dataframe().to_excel(self.module.EXCEL_PATH, index=False)

    def test_default_training_meta_contains_required_keys_and_status_defaults(self):
        meta = self.module.default_training_meta()

        expected_keys = {
            'baseCount',
            'deltaCount',
            'lastTfidfRefreshAt',
            'lastWord2VecRefreshAt',
            'lastFullMergeAt',
            'tfidfRefreshNeeded',
            'word2vecRefreshNeeded',
            'lastImportBatchId',
            'schemaVersion',
            'refreshStatus',
            'recoveryNeeded',
            'lastSuccessfulModelBuildAt',
        }

        self.assertTrue(expected_keys.issubset(meta.keys()))
        self.assertEqual(meta['baseCount'], 0)
        self.assertEqual(meta['deltaCount'], 0)
        self.assertFalse(meta['tfidfRefreshNeeded'])
        self.assertFalse(meta['word2vecRefreshNeeded'])
        self.assertEqual(meta['refreshStatus'], 'clean')
        self.assertFalse(meta['recoveryNeeded'])

    def test_compute_refresh_flags_uses_spec_thresholds(self):
        clean_flags = self.module.compute_refresh_flags(299)
        tfidf_flags = self.module.compute_refresh_flags(300)
        almost_full_flags = self.module.compute_refresh_flags(999)
        full_flags = self.module.compute_refresh_flags(1000)

        self.assertEqual(clean_flags['refreshStatus'], 'clean')
        self.assertFalse(clean_flags['tfidfRefreshNeeded'])
        self.assertFalse(clean_flags['word2vecRefreshNeeded'])

        self.assertEqual(tfidf_flags['refreshStatus'], 'stale_tfidf')
        self.assertTrue(tfidf_flags['tfidfRefreshNeeded'])
        self.assertFalse(tfidf_flags['word2vecRefreshNeeded'])

        self.assertEqual(almost_full_flags['refreshStatus'], 'stale_tfidf')
        self.assertTrue(almost_full_flags['tfidfRefreshNeeded'])
        self.assertFalse(almost_full_flags['word2vecRefreshNeeded'])

        self.assertEqual(full_flags['refreshStatus'], 'stale_full')
        self.assertTrue(full_flags['tfidfRefreshNeeded'])
        self.assertTrue(full_flags['word2vecRefreshNeeded'])

    def test_recovery_migrates_legacy_runtime_state_when_metadata_is_missing(self):
        self.write_legacy_state()

        self.module.recover_runtime_state_on_startup()

        meta = self.module.load_training_meta()
        base_records = self.read_jsonl(self.module.BASE_RECORDS_PATH)

        self.assertEqual(len(base_records), 2)
        self.assertTrue(os.path.exists(self.module.BASE_MODEL_PATH))
        self.assertTrue(os.path.exists(self.module.BASE_W2V_PATH))
        self.assertEqual(meta['baseCount'], 2)
        self.assertEqual(meta['deltaCount'], 0)
        self.assertFalse(meta['recoveryNeeded'])
        self.assertEqual(meta['refreshStatus'], 'clean')

    def test_recovery_replaces_corrupt_metadata_and_preserves_existing_base_count(self):
        self.write_jsonl(self.module.BASE_RECORDS_PATH, [{'Description': 'kept', 'Suggestion': 'value'}])
        os.makedirs(os.path.dirname(self.module.TRAINING_META_PATH), exist_ok=True)
        with open(self.module.TRAINING_META_PATH, 'w', encoding='utf-8') as handle:
            handle.write('{not-json')

        self.module.recover_runtime_state_on_startup()

        meta = self.module.load_training_meta()
        self.assertEqual(meta['baseCount'], 1)
        self.assertEqual(meta['refreshStatus'], 'clean')
        self.assertFalse(meta['recoveryNeeded'])

    def test_recovery_rebuilds_metadata_when_corpus_exists_without_meta(self):
        self.write_jsonl(
            self.module.BASE_RECORDS_PATH,
            [
                {'Description': 'base 1', 'Suggestion': 'one'},
                {'Description': 'base 2', 'Suggestion': 'two'},
            ],
        )
        self.write_jsonl(self.module.DELTA_RECORDS_PATH, [{'Description': 'delta 1', 'Suggestion': 'three'}])

        self.module.recover_runtime_state_on_startup()

        meta = self.module.load_training_meta()
        self.assertEqual(meta['baseCount'], 2)
        self.assertEqual(meta['deltaCount'], 1)
        self.assertFalse(meta['tfidfRefreshNeeded'])
        self.assertFalse(meta['word2vecRefreshNeeded'])
        self.assertEqual(meta['refreshStatus'], 'clean')

    def test_recovery_marks_stale_model_state_when_base_data_exists_without_rebuilt_models(self):
        self.write_jsonl(self.module.BASE_RECORDS_PATH, [{'Description': 'base 1', 'Suggestion': 'one'}])
        self.module.save_training_meta({
            **self.module.default_training_meta(),
            'baseCount': 1,
            'lastImportBatchId': 'batch-1',
        })

        self.module.recover_runtime_state_on_startup()

        meta = self.module.load_training_meta()
        self.assertTrue(meta['recoveryNeeded'])
        self.assertTrue(meta['tfidfRefreshNeeded'])
        self.assertTrue(meta['word2vecRefreshNeeded'])
        self.assertEqual(meta['refreshStatus'], 'stale_full')

    def test_recovery_marks_stale_model_state_when_artifacts_exist_but_base_corpus_is_newer(self):
        self.write_jsonl(
            self.module.BASE_RECORDS_PATH,
            [
                {'Description': 'base 1', 'Suggestion': 'one'},
                {'Description': 'base 2', 'Suggestion': 'two'},
            ],
        )
        os.makedirs(self.module.MODELS_DIR, exist_ok=True)
        with open(self.module.BASE_MODEL_PATH, 'wb') as handle:
            handle.write(b'old-model')
        with open(self.module.BASE_W2V_PATH, 'wb') as handle:
            handle.write(b'old-w2v')

        old_timestamp = '2026-03-07 08:00:00'
        self.module.save_training_meta({
            **self.module.default_training_meta(),
            'baseCount': 1,
            'lastImportBatchId': 'batch-2',
            'lastSuccessfulModelBuildAt': old_timestamp,
        })

        future_timestamp = self.module.pd.Timestamp('2026-03-07 09:00:00').timestamp()
        os.utime(self.module.BASE_RECORDS_PATH, (future_timestamp, future_timestamp))
        stale_timestamp = self.module.pd.Timestamp(old_timestamp).timestamp()
        os.utime(self.module.BASE_MODEL_PATH, (stale_timestamp, stale_timestamp))
        os.utime(self.module.BASE_W2V_PATH, (stale_timestamp, stale_timestamp))

        self.module.recover_runtime_state_on_startup()

        meta = self.module.load_training_meta()
        self.assertEqual(meta['baseCount'], 2)
        self.assertTrue(meta['recoveryNeeded'])
        self.assertTrue(meta['tfidfRefreshNeeded'])
        self.assertTrue(meta['word2vecRefreshNeeded'])
        self.assertEqual(meta['refreshStatus'], 'stale_full')

    def test_startup_bootstrap_invokes_recovery_once_before_serving(self):
        module = self.module
        module._runtime_recovery_completed = False
        call_order = []

        with mock.patch.object(module, 'recover_runtime_state_on_startup', side_effect=lambda: call_order.append('recover')) as recover_mock:
            with mock.patch.object(module, 'load_model', return_value=True):
                with mock.patch.object(module.app, 'run', side_effect=lambda *args, **kwargs: call_order.append('run')):
                    module.start_app_server()
                    module.bootstrap_runtime_state()

        recover_mock.assert_called_once_with()
        self.assertEqual(call_order, ['recover', 'run'])


if __name__ == '__main__':
    unittest.main()
