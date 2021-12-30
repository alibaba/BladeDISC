# How to write python tests for Tao?

## Background
Tao's python tests are build upon the very basic built-in package `uinttest`. To add a unit test case, all you need to do is to add yet another `.py` file in `tao/tao_bridge/test` directory with a test class inside. Please refer to `unittest`'s official documentation ([python2](https://docs.python.org/2/library/unittest.html) and [python3](https://docs.python.org/3.6/library/unittest.html)) for more details.

## Write tests
As you know, Tao compiler needs to plug an shared library into host Tensorflow and lookk for the compiler program in env variables. It's tedious and boring to do such work in every test class. So, the `TaoTestCase` class in `tao_ut_common` module comes to help. So your test class can inherit from `TaoTestCase` instead of `unittest.TestCase`. For example:
```python
from tao_ut_common import *

class TestFoo(TaoTestCase):
    def test_foo(self):
        pass
```
More specificity, `TaoTestCase` implement `setUp` and `tearDown` methods of `unittest.TestCase` to do it's work. So if your test class also have `setUp` or `tearDown` work to do, don't forget to calls corresponding methods in `TaoTestCase` in the first place. For example:

```python
from tao_ut_common import *

class TestFoo(TaoTestCase):
    def setUp(self):
        super(TaoTestCase, self).setUp()
        # bla..bla..bla..
    
    def tearDown(self):
        super(TaoTestCase, self).tearDown()
        # bla..bla..bla..

    def test_foo(self):
        pass
```

Call `unittest.main()` so that this test file can be executed individually. It optional since CI/CD use `pytest` to run all tests and generate reports.
```python
if __name__ == "__main__":
    unittest.main()
```

Since the host Tensorflow may come with python 2 or 3, please make your code compatible with both python 2 and 3.


## Run tests
If you are running the tests manually, (not via `tao_build` command), please make sure `tao_compiler` and `tao_bridge` are compiled properly.

The `tao_build` build command use [pytest](https://docs.pytest.org/en/latest/) to run the tests and generate reports. `pytest` is installed to your virtual env path by the `tao_venv` command automatically. You can run all the tests like this:

```bash
cd tao/tao_bridge/test
${PATH_TO_YOUR_VENV}/bin/python -m pytest .
```
