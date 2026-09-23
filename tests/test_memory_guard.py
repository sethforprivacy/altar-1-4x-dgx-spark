import sys
import unittest
from unittest.mock import patch
import json
import subprocess
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory_guard import decision,GIB
import memory_guard

class GuardTest(unittest.TestCase):
    def test_transient_pressure_recovers(self):
        self.assertEqual(decision(5*GIB,0),(1,None))
        self.assertEqual(decision(6*GIB,1),(0,None))
    def test_two_low_samples_stop(self):
        self.assertEqual(decision(5*GIB,1),(2,'sustained-pressure'))
    def test_emergency_does_not_wait(self):
        self.assertEqual(decision(2*GIB-1,0),(1,'emergency'))
    def test_exact_emergency_boundary_uses_soft_gate(self):
        self.assertEqual(decision(2*GIB,0),(1,None))

class WatchdogActionsTest(unittest.TestCase):
    def run_guard(self, samples, execute):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'guard.json'
            with patch.object(sys,'argv',['memory_guard','--container','test-container','--state',str(state)]), \
                 patch.object(memory_guard,'available',side_effect=samples), \
                 patch.object(memory_guard.subprocess,'run',side_effect=execute), \
                 patch.object(memory_guard.time,'sleep'):
                memory_guard.main()
            receipt=json.loads(state.read_text())
            self.assertEqual(json.loads(state.with_suffix('.TRIPPED').read_text()),receipt)
            return receipt
    def test_emergency_kills_without_potentially_blocking_inspection(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            self.assertEqual(command,['docker','kill','test-container'])
            return subprocess.CompletedProcess(command,0)
        receipt=self.run_guard([GIB],execute)
        self.assertEqual(receipt['trip'],'emergency')
        self.assertEqual(len(calls),1)
    def test_recovery_resets_pressure_before_coordinated_stop(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command,0,stdout='running\n')
        receipt=self.run_guard([5*GIB,7*GIB,5*GIB,5*GIB],execute)
        self.assertEqual(receipt['trip'],'sustained-pressure')
        self.assertEqual(receipt['strikes'],2)
        self.assertEqual(sum(c[1]=='inspect' for c in calls),4)
        self.assertEqual(calls[-1],['docker','stop','-t','5','test-container'])

if __name__=='__main__':unittest.main()
