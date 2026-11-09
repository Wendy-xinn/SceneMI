import copy
import hashlib
import json
import unittest
from unittest import mock

from scenemi import amass_cli_compatibility as compatibility
from scenemi import amass_finetuning as branch


def digest(value):
    return hashlib.sha256(value).hexdigest()


class CliCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.original = b'# original CLI\n'
        self.current = (self.original + compatibility.GIMO_DISPATCH.encode()
                        + compatibility.GIMO_PARSER.encode())
        self.spec = dict(kind='gimo_cli_dispatch_only', source=compatibility.CLI,
                         parent_sha256=digest(self.original),
                         current_sha256=digest(self.current))

    def test_exact_cli_extension_is_authenticated(self):
        result = compatibility.verify_cli_migration(self.spec, source=self.current)
        self.assertTrue(result['reconstructed_parent_verified'])
        self.assertFalse(result['training_implementation_changed'])

    def test_other_changes_cannot_be_admitted_by_updating_current_hash(self):
        source = self.current + b'# additional change\n'
        spec = dict(self.spec, current_sha256=digest(source))
        with self.assertRaisesRegex(ValueError, 'beyond the GIMO'):
            compatibility.verify_cli_migration(spec, source=source)
        with self.assertRaisesRegex(ValueError, 'current CLI changed'):
            compatibility.verify_cli_migration(self.spec, source=source)

    def test_missing_or_duplicate_dispatch_and_other_sources_are_rejected(self):
        for source in (self.original, self.current + compatibility.GIMO_DISPATCH.encode()):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'exactly'):
                compatibility.verify_cli_migration(
                    dict(self.spec, current_sha256=digest(source)), source=source)
        with self.assertRaisesRegex(ValueError, 'Only the declared'):
            compatibility.verify_cli_migration(
                dict(self.spec, source='scenemi/models.py'), source=self.current)

    def test_parent_is_preserved_and_original_verifier_is_restored(self):
        old = dict(sources={compatibility.CLI: self.spec['parent_sha256'],
                            'scenemi/models.py': 'unchanged'}, config={})
        old['sha256'] = digest(json.dumps(old, sort_keys=True).encode())
        snapshot = copy.deepcopy(old)
        seal = dict(sources={compatibility.CLI: self.spec['current_sha256'],
                             'scenemi/models.py': 'unchanged'})
        config = dict(continuation=dict(source_migration=self.spec))

        def check_parent(c, paths, comparison):
            self.assertEqual(comparison['sources'][compatibility.CLI], self.spec['parent_sha256'])
            branch.verify_training_seal(old, {}, paths, 'index')
            return dict(contract=old), {}, {}

        with mock.patch.object(compatibility, 'verify_cli_migration', return_value=self.spec), \
                mock.patch.object(compatibility, '_parent_state', side_effect=check_parent), \
                mock.patch.object(branch, 'verify_training_seal') as verifier:
            state, _, provenance = compatibility.parent_state(config, {}, seal)
            self.assertIs(branch.verify_training_seal, verifier)
            view = verifier.call_args.args[0]
            self.assertEqual(view['sources'][compatibility.CLI], self.spec['current_sha256'])
            self.assertEqual(view['sources']['scenemi/models.py'], 'unchanged')
            self.assertEqual(state['contract'], snapshot)
            self.assertEqual(old, snapshot)
            self.assertIn('source_migration', provenance)
            old['sha256'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'Parent training contract changed'):
                compatibility.parent_state(config, {}, seal)
            self.assertIs(branch.verify_training_seal, verifier)


if __name__ == '__main__':
    unittest.main()
