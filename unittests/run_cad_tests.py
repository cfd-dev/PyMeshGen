import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittests.test_sfmesh_3d_afm import TestCADFileAFM

out_path = Path(__file__).parent / 'result_cad.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCADFileAFM)
    runner = unittest.TextTestRunner(verbosity=2, stream=f)
    result = runner.run(suite)
    f.write(f'\nTests run: {result.testsRun}\n')
    f.write(f'Errors: {len(result.errors)}\n')
    f.write(f'Failures: {len(result.failures)}\n')
    for test, traceback in result.errors:
        f.write(f'ERROR: {test}\n{traceback}\n')
    for test, traceback in result.failures:
        f.write(f'FAILURE: {test}\n{traceback}\n')
    f.write(f'SUCCESS: {result.wasSuccessful()}\n')

print('DONE', flush=True)
sys.exit(0 if result.wasSuccessful() else 1)