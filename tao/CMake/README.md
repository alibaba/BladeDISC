# Build TAO with CMake

## Get Started
```bash
# install cmake 3.8.2
sudo yum install cmake -b current -y

# default configuration
mkdir build && cd build
cmake ..
make -j
```

## More Configuration

To compile with custom compiler, just set `CC` and `CXX` environment variable like make.

```bash
CC=/usr/local/gcc-5.3.0/bin/gcc CXX=/usr/local/gcc-5.3.0/bin/g++ cmake ../
```
To specify the `python` path where the host tensorflow is installed, add a `PYTHON` option like this:

```bash
cmake ../ -DPYTHON=/path/to/your/python_virtual_env/bin/python
```

By default, we have C++ tests disabled. To enabled it, set `TAO_ENABLE_CXX_TESTING` to `ON`, this will download and build `gtest` and compile testing source files.

```bash
cmake ../ -DTAO_ENABLE_CXX_TESTING=ON
```