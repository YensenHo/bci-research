"""Offline checks for the public dataset exploratory reader."""
import io
import tempfile
import unittest
from pathlib import Path
import struct
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from on006866_pilot_20261117 import trials_from_events
from on006866_verify_read_20261117 import fetch

class ReaderTests(unittest.TestCase):
    def test_matlab_layout(self):
        x = np.arange(69*2000,dtype=np.float32).reshape(69,2000,order='F')
        file = io.BytesIO()
        savemat(file, {'data':x}, do_compression=False)
        raw = file.getvalue()
        words = struct.unpack('<14I',raw[128:184])
        self.assertEqual(words[8:10],(69,2000))
        decoded = np.frombuffer(raw[184:],dtype='<f4',count=69*2000).reshape(2000,69).T
        np.testing.assert_array_equal(x,decoded)
    def test_ambiguous_rating_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'events.tsv'
            p.write_text('onset\tvalue\n'+''.join(f'{i*10}\t18\n{i*10+2}\t49\n{i*10+3}\t3\n{i*10+4}\t48\n{i*10+5}\t5\n' for i in range(30)))
            self.assertEqual(len(trials_from_events(p)),30)
            p.write_text(p.read_text().replace('4\t48\n','4\t47\n',1))
            with self.assertRaises(ValueError): trials_from_events(p)
    def test_full_http_response_rejected(self):
        class Response:
            status_code=200
            content=b'abcd'
            headers={}
            def raise_for_status(self): pass
        with patch('on006866_verify_read_20261117.requests.get',return_value=Response()):
            with self.assertRaises(ValueError): fetch(0,4)

if __name__=='__main__': unittest.main()
