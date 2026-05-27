import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittests.test_sfmesh_3d_afm import TestArbitrary3DSurfaceAFM

suite = unittest.TestLoader().loadTestsFromTestCase(TestArbitrary3DSurfaceAFM)
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr)
result = runner.run(suite)
print('Tests run:', result.testsRun)
print('Errors:', len(result.errors))
print('Failures:', len(result.failures))
if result.errors:
    for test, traceback in result.errors:
        print('ERROR:', test)
        print(traceback)
if result.failures:
    for test, traceback in result.failures:
        print('FAILURE:', test)
        print(traceback)
sys.exit(0 if result.wasSuccessful() else 1)