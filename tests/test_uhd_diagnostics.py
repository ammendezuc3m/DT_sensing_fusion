"""Radio diagnostics without hardware: preserve metadata and command failures."""
import argparse
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/python/ssb_python'))
from uhd_diagnostics import start_diagnostics, stop_diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_native_environment_and_redacted_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            args = argparse.Namespace(uhd_log_dir=tmp, log_csv=tmp + '/online.csv',
                                      uhd_log_level='debug', serial='test-radio', api_token='secret-token')
            logger = start_diagnostics(args)
            self.assertEqual(os.environ['UHD_LOG_FILE_LEVEL'], 'debug')
            self.assertEqual(os.environ['UHD_LOG_LEVEL'], 'debug')
            self.assertTrue(os.environ['UHD_LOG_FILE'].endswith('_uhd.log'))
            stop_diagnostics(logger)
            output = next(Path(tmp).glob('*_receiver.log')).read_text()
            self.assertIn('test-radio', output)
            self.assertNotIn('secret-token', output)
            self.assertIn('Session end', output)

    def test_capture_logs_timeout_metadata_and_command_exception(self):
        import numpy as np
        metadata = types.SimpleNamespace(error_code='timeout', has_time_spec=True,
            time_spec=types.SimpleNamespace(get_real_secs=lambda: 123.25),
            out_of_sequence=True, strerror=lambda: 'ERROR_CODE_TIMEOUT')
        fake = types.SimpleNamespace(types=types.SimpleNamespace(
            RXMetadata=lambda: metadata, StreamCMD=lambda mode: types.SimpleNamespace(),
            StreamMode=types.SimpleNamespace(num_done='num_done'),
            RXMetadataErrorCode=types.SimpleNamespace(none='none')))
        spec = importlib.util.spec_from_file_location('diagnostic_capture',
            ROOT / 'src/python/ssb_python/profile_online_datassb_pipeline.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'uhd': fake}):
            spec.loader.exec_module(module)
        streamer = types.SimpleNamespace(issue_stream_cmd=lambda cmd: None, recv=lambda *args: 0)
        with self.assertLogs('ssb.uhd', logging.ERROR) as captured:
            with self.assertRaisesRegex(RuntimeError, 'ERROR_CODE_TIMEOUT'):
                module.capture_one_block(streamer, 100, 32)
        output = '\n'.join(captured.output)
        for text in ('requested=100', 'received=0', '123.25', 'out_of_sequence'):
            self.assertIn(text, output)
        def fail(cmd):
            raise RuntimeError('control ACK timed out')
        streamer.issue_stream_cmd = fail
        with self.assertLogs('ssb.uhd', logging.ERROR) as captured:
            with self.assertRaisesRegex(RuntimeError, 'control ACK'):
                module.capture_one_block(streamer, 100, 32)
        self.assertIn('stream command failed', '\n'.join(captured.output))


if __name__ == '__main__':
    unittest.main()
