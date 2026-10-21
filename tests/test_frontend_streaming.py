from pathlib import Path
import shutil
import subprocess
import unittest


class FrontendStreamingTests(unittest.TestCase):
    def test_real_ui_stream_consumer(self):
        node=shutil.which('node')
        if not node: self.skipTest('Node.js is required for frontend regression checks.')
        result=subprocess.run([node,str(Path(__file__).with_name('streaming_ui.cjs'))],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
