import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import classifier


class ReleaseTests(unittest.TestCase):
    def test_nine_category_metadata_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, config = self.fixture(root)
            report_path = run / 'report.json'
            report = json.loads(report_path.read_text())
            report['architecture'] = classifier.CONVNEXTV2_ARCHITECTURE
            for labels, valid in ((classifier.NINE_CATEGORIES, True), (list(reversed(classifier.NINE_CATEGORIES)), False)):
                report['labels'] = labels
                report_path.write_text(json.dumps(report))
                config['report_sha256'] = classifier.sha(report_path); path.write_text(json.dumps(config))
                if valid: self.assertEqual(classifier.verify_release(root, path)[1]['labels'], labels)
                else:
                    with self.assertRaises(ValueError): classifier.verify_release(root, path)
            report['labels'] = classifier.NINE_CATEGORIES; report['architecture'] = classifier.ARCHITECTURE
            report_path.write_text(json.dumps(report))
            config['report_sha256'] = classifier.sha(report_path); path.write_text(json.dumps(config))
            with self.assertRaises(ValueError): classifier.verify_release(root, path)

    def test_selected_vocabularies_are_not_interchangeable(self):
        for model_id, architecture, labels in (
            ('convnext_v2', classifier.CONVNEXTV2_ARCHITECTURE, classifier.NINE_CATEGORIES),
            ('convnext_v1', classifier.CONVNEXT_ARCHITECTURE, classifier.CATEGORIES),
            ('efficientnet_b0', classifier.ARCHITECTURE, classifier.CATEGORIES),
        ):
            report = {'architecture': architecture, 'labels': labels}
            with patch.object(classifier, 'verify_release', return_value=(Path('.'), report, 'sha')):
                self.assertEqual(classifier.selected_release(model_id)[1]['labels'], labels)
            report = {**report, 'labels': classifier.CATEGORIES if model_id == 'convnext_v2' else classifier.NINE_CATEGORIES}
            with patch.object(classifier, 'verify_release', return_value=(Path('.'), report, 'sha')):
                with self.assertRaisesRegex(ValueError, 'category order'): classifier.selected_release(model_id)

    def test_reviewed_records_accept_new_and_legacy_categories(self):
        from backend.engine import fabric_record
        for category in ('linen', 'nylon', 'leather', 'cotton_mixed'):
            record = fabric_record({'reviewed': True, 'fabric_category': category})
            self.assertEqual(record['fabric_category'], category)
            self.assertEqual(record['material'], 'unknown')

    def test_model_selection_allowlist(self):
        for value in ('../outside', 'random', [], True, ''):
            with self.assertRaises(ValueError): classifier.selected_config(value)
        self.assertEqual(classifier.selected_config('convnext_v2'), classifier.CONFIG)
        self.assertEqual(classifier.selected_config('convnext_v1').name, 'fabric_classifier_release.convnext_v1.json')

    def test_mismatched_model_cannot_silently_use_another_architecture(self):
        with patch.object(classifier, 'verify_release', return_value=(Path('.'), {'architecture': classifier.ARCHITECTURE}, 'sha')):
            with self.assertRaisesRegex(ValueError, 'pinned architecture'):
                classifier.selected_release('convnext_v2')

    def test_convnext_metadata_supported_unknown_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, config = self.fixture(root)
            report_path = run / 'report.json'
            report = json.loads(report_path.read_text())
            for architecture in (classifier.CONVNEXT_ARCHITECTURE, classifier.CONVNEXTV2_ARCHITECTURE, 'unrecognised_model'):
                report['architecture'] = architecture
                report_path.write_text(json.dumps(report))
                config['report_sha256'] = classifier.sha(report_path); path.write_text(json.dumps(config))
                if architecture in (classifier.CONVNEXT_ARCHITECTURE, classifier.CONVNEXTV2_ARCHITECTURE):
                    self.assertEqual(classifier.verify_release(root, path)[1]['architecture'], architecture)
                    self.assertEqual(classifier.model_name(architecture), 'ConvNeXt-Tiny' if architecture == classifier.CONVNEXT_ARCHITECTURE else 'ConvNeXt V2-Tiny')
                else:
                    with self.assertRaises(ValueError): classifier.verify_release(root, path)

    def fixture(self, root):
        run = root / 'runtime/training/fixture'; run.mkdir(parents=True)
        checkpoint = run / 'model.safetensors'; checkpoint.write_bytes(b'not a real tensor file; metadata-only test')
        report = {'architecture': classifier.ARCHITECTURE, 'labels': classifier.CATEGORIES,
                  'manifest_sha256': 'manifest', 'checkpoint_sha256': classifier.sha(checkpoint),
                  'validation': {'accuracy': 0.5, 'macro_f1': 0.5}}
        report_path = run / 'report.json'; report_path.write_text(json.dumps(report))
        config = {'enabled': True, 'status': 'experimental_validation_only', 'run_directory': 'runtime/training/fixture',
                  'checkpoint_sha256': classifier.sha(checkpoint), 'report_sha256': classifier.sha(report_path), 'manifest_sha256': 'manifest'}
        path = root / 'release.json'; path.write_text(json.dumps(config))
        return path, run, config

    def test_verified_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, _ = self.fixture(root)
            verified, report, checksum = classifier.verify_release(root, path)
            self.assertEqual(verified, run)
            self.assertEqual(report['labels'], classifier.CATEGORIES)
            self.assertEqual(checksum, classifier.sha(run / 'model.safetensors'))

    def test_rejects_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, _ = self.fixture(root)
            (run / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint checksum'):
                classifier.verify_release(root, path)

    def test_rejects_modified_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, _ = self.fixture(root)
            (run / 'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'report checksum'):
                classifier.verify_release(root, path)

    def test_rejects_wrong_category_order_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, run, config = self.fixture(root)
            report_path = run / 'report.json'
            report = json.loads(report_path.read_text()); report['labels'] = list(reversed(report['labels']))
            report_path.write_text(json.dumps(report))
            config['report_sha256'] = classifier.sha(report_path); path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'category order'):
                classifier.verify_release(root, path)

    def test_rejects_directory_escape_and_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _, config = self.fixture(root)
            for change in ({'run_directory': '../outside'}, {'enabled': False}):
                path.write_text(json.dumps({**config, **change}))
                with self.assertRaises(ValueError): classifier.verify_release(root, path)

    def test_missing_config_fails_closed(self):
        with patch.object(classifier, 'CONFIG', Path('nonexistent-release-file.json')):
            self.assertFalse(classifier.availability()['available'])
            result = classifier.predict(None)
            self.assertFalse(result['classifier']['available'])
            self.assertEqual(result['predictions'], {})

    def test_bad_tensor_file_fails_closed(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None: self.skipTest('Optional ML dependency absent')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _, _ = self.fixture(root)
            with patch.object(classifier, 'verify_release', wraps=lambda **kwargs: self.verify_fixture(root, path)), patch.object(classifier, '_CACHE', None):
                result = classifier.predict(None)
            self.assertFalse(result['classifier']['available'])
            self.assertEqual(result['predictions'], {})

    @staticmethod
    def verify_fixture(root, path):
        # Preserve the unpatched verifier for a metadata-valid, tensor-invalid run.
        return ORIGINAL_VERIFY(root, path)


ORIGINAL_VERIFY = classifier.verify_release


if __name__ == '__main__': unittest.main()
