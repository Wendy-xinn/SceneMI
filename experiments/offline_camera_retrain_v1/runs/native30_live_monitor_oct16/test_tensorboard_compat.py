import importlib.util
import json
from pathlib import Path
import unittest

from google.protobuf import json_format
from tensorboard.plugins.custom_scalar.layout_pb2 import Layout

SPEC = importlib.util.spec_from_file_location('tensorboard_compat_server', Path(__file__).with_name('tensorboard_compat_server.py'))
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


class ProtobufCompatibilityTests(unittest.TestCase):
    def test_tensorboard_legacy_json_argument_serves_current_proto(self):
        original = json_format.MessageToJson
        try:
            compat.install_compatibility()
            result = json.loads(json_format.MessageToJson(Layout(), including_default_value_fields=True))
            self.assertEqual(result['category'], [])
            self.assertEqual(result['version'], 0)
        finally:
            json_format.MessageToJson = original


if __name__ == '__main__':
    unittest.main()
