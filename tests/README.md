# EncoderMap tests

## Test suites

EncoderMap's tests use the builtin `unittest` module for automated testing: https://docs.python.org/3/library/unittest.html

While not as expendable as `pytest`, the `unittest` framework allows EncoderMap to build tests using `tensorflow`'s `TestCase`, which implements more checks and allows checks to compare eager tensors, symbolic tensors and numpy arrays. Especially the `self.assertAllClose` and EncoderMap's own `assertAllClosePeriodic` tests are very useful when comparing datasets (potentially in periodic space).

### Sorting/filtering tests

There used to be a method inside the `tf.test.TestCase` class called `test_session`, which was not intended as a test, but because it starts with 'test', python's `unittest` interprets it as a test and runs it. Maybe this method has been renamed to `cached_session` as outlined int [this commit](https://github.com/tensorflow/tensorflow/commit/9962eb5e84b15e309410071b06c2ed2d6148ed44), but if not, the TestSuites in EncoderMap are filtered and these methods are removed as suggested in (this stackoverflow answer)[https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase]

In the `test_losses.py` file, the TestSuite `TestLossesNonPeriodic` used to fail, when the test methods were run in an arbitrary order. Maybe it has since been fixed, but if this error pops up again these code snippets can help in sorting the test methods:

```python
def sort_tests(tests):
    for test in tests:
        try:
            if test._testMethodName == "test_losses_not_periodic":
                return 1
        except AttributeError:
            for t in test._tests:
                try:
                    if t._testMethodName == "test_losses_not_periodic":
                        return 1
                except AttributeError:
                    for _ in t._tests:
                        if _._testMethodName == "test_losses_not_periodic":
                            return 1
    return 2


def unpack_tests(tests):
    for test in tests:
        try:
            print(test._testMethodName)
        except AttributeError:
            for t in test._tests:
                try:
                    print(t._testMethodName)
                except AttributeError:
                    for _ in t._tests:
                        print(_._testMethodName)


def filter_key_test_suites(suite):
    """Returns filters True/False depending on whether a
    test needs to be executed before or after the optional
    requirements are installed"""
    if "before_installing_reqs" in suite.__str__():
        return False
    else:
        return True


class SortableSuite(unittest.TestSuite):
    def sort(self):
        # print("old")
        # unpack_tests(self._tests)
        self._tests = list(sorted(self._tests, key=sort_tests))
        # print("\n new")
        # unpack_tests(self._tests)

    def filter(self):
        self._tests = list(filter(filter_key_test_suites, self._tests))
```

### Running tests with `pytest`

There is a `conftest.py` in the `tests/` directory, that sets some parameters for running the tests using pytest. As of EncoderMap 3.1.0, this needs a rework and isn't guarenteed to run the tests.

### Expensive tests

There are some very expensive tests inside the test suites of EncoderMap. These tests can be activated by setting the environment variable `RUN_EXPENSIVE_TESTS=True` i.e:

```bash
RUN_EXPENSIVE_TESTS=True python -m unittests discover -s tests
```

The file `test_tf1_tf2_deterministic.py` contains such an expensive test which you can run via:

```bash
RUN_EXPENSIVE_TESTS=True python tests/test_tf1_tf2_deterministic.py TestTf1Tf2Deterministic
```
