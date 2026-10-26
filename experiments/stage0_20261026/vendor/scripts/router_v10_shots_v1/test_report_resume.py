import os
assert os.environ.get('SLURM_JOB_ID')
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SOURCE=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter/scripts/router_v10_shots_v1/report.py')
spec=importlib.util.spec_from_file_location('router_report_resume',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class ResumeTests(unittest.TestCase):
    def root(self,temp):
        root=Path(temp)
        for name in ('DOWNSTREAM_FROZEN.json','ROUTER_FROZEN.json'):
            r.write(root/name,dict(all_checks_pass=True,cell_count=30,files={}))
        r.write(root/'DOWNSTREAM_REGISTRATION.json',dict(code_hashes={}))
        return root

    def test_retry_after_failure_keeps_first_start_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.root(temp)
            with patch.object(r,'historical_rows',side_effect=RuntimeError('injected computation failure')):
                for job in ('test_first','test_retry'):
                    with patch.dict(os.environ,SLURM_JOB_ID=job):
                        with self.assertRaisesRegex(RuntimeError,'injected computation failure'):r.evaluate(root)
            self.assertEqual(r.read(root/'EVALUATION_STARTED.json')['slurm_job_id'],'test_first')
            self.assertFalse((root/'classification_report').exists())

    def test_partial_output_retry_and_valid_completed_reuse(self):
        self.assertTrue(hasattr(r,'_evaluate_attempt'),'Missing isolated report attempt writer')
        with tempfile.TemporaryDirectory() as temp:
            root=self.root(temp);calls=[]
            def synthetic(root,out,*args):
                calls.append(out)
                r.write(out/'RESULTS.json',dict(per_split=[{}]*90,mean_std=[{}]*30))
                if len(calls)==1:raise RuntimeError('injected write failure')
                for name in ('REPORT.md','per_split.csv','mean_std.csv'):(out/name).write_text(name)
                final=root/'classification_report'
                r.write(out/'AUDIT.json',dict(all_checks_pass=True,rows=90,groups=30,seeds=[42,43,44],historical_hashes={},
                    files={str(final/p.name):r.sha(p) for p in out.iterdir()},
                    downstream_frozen_sha256=r.sha(root/'DOWNSTREAM_FROZEN.json'),router_frozen_sha256=r.sha(root/'ROUTER_FROZEN.json')))
            with patch.object(r,'_evaluate_attempt',side_effect=synthetic):
                with self.assertRaisesRegex(RuntimeError,'injected write failure'):r.evaluate(root)
                self.assertFalse((root/'classification_report').exists())
                failed=list((root/'classification_report_attempts').glob('*/FAILURE.json'))
                self.assertEqual(len(failed),1)
                r.evaluate(root)
                final=root/'classification_report';self.assertTrue((final/'AUDIT.json').exists())
                self.assertEqual(len(calls),2)
                r.evaluate(root);self.assertEqual(len(calls),2)
                (final/'REPORT.md').write_text('tampered')
                with self.assertRaises(ValueError):r.evaluate(root)

    def test_changed_freeze_rejected_on_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.root(temp)
            with patch.object(r,'historical_rows',side_effect=RuntimeError('stop')):
                with self.assertRaisesRegex(RuntimeError,'stop'):r.evaluate(root)
                (root/'ROUTER_FROZEN.json').write_text(json.dumps(dict(all_checks_pass=True,cell_count=30,files={},changed=True)))
                with self.assertRaisesRegex(ValueError,'freeze'):r.evaluate(root)
if __name__=='__main__':unittest.main(verbosity=2)
