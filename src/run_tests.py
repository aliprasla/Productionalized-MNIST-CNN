import unittest
import os

from tests import test_api


TESTING_SUITE_VERBOSE = int(os.environ.get('TESTING_SUITE_VERBOSE'))
TEST_LOADER = unittest.TestLoader()

TEST_SUITE = unittest.TestSuite()
TEST_SUITE.addTests(TEST_LOADER.loadTestsFromModule(test_api))

TEST_RUNNER = unittest.TextTestRunner(verbosity=TESTING_SUITE_VERBOSE)
TEST_RUNNER.run(TEST_SUITE)