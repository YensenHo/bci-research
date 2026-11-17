import unittest
import numpy as np
from unittest.mock import patch
import on006866_cross_subject_20261117 as c

class Checks(unittest.TestCase):
 def test_rating_rejection(self):
  rows=[]
  for i in range(30):
   for t,v in [(0,18),(3,49),(4,3),(5,48),(6,2)]:rows.append({'onset':str(i*10+t),'value':str(v)})
  self.assertEqual(len(c.parse_trials(rows)),30)
  rows[3]['value']='47'
  with self.assertRaises(ValueError):c.parse_trials(rows)
 def test_calibration_changes_only_calibrated_models(self):
  rng=np.random.default_rng(1);data={i:{'X':rng.normal(size=(30,40)),'y':np.ones(30)*5,'condition':np.full(30,18)} for i in range(2,10)}
  a=c.evaluate(data);data[8]['y'][:10]=9;b=c.evaluate(data)
  for k in ('source_mean','source_condition','source_eeg'):self.assertEqual(a['target_macro_mae'][k],b['target_macro_mae'][k])
  self.assertNotEqual(a['target_macro_mae']['cal_mean'],b['target_macro_mae']['cal_mean'])
 def test_http_range_rejected(self):
  class R:
   status_code=200;content=b'abcd';headers={}
   def raise_for_status(self):pass
  with patch.object(c.requests,'get',return_value=R()):
   with self.assertRaises(ValueError):c.request('http://x',0,4,4)

if __name__=='__main__':unittest.main()
