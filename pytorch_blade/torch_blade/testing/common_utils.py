import unittest
import re
import time
import torch
import functools
from collections import namedtuple
from numbers import Number
from torch._six import string_classes
from torch.testing import assert_allclose
from torch.testing._internal.common_utils import (
    is_iterable,
    TestCase as TorchTestCase,
)
from torch_blade import version

__all__ = ['benchmark', 'assert_almost_equal', 'TestCase']

class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x)  # instead of Heaviside step fn
        return output


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class TestCase(TorchTestCase):

    def setUp(self):
        self.device = torch.device('cuda') if version.cuda_available else torch.device('cpu')

    @staticmethod
    def _iter_but_no_len(o):
        def has_len(o):
            return hasattr(o, '__len__')

        return is_iterable(o) and not has_len(o)

    def assertAlmostEqual(self, x, y, *, places=None, msg=None, delta=None):
        # This method is added since TorchTestCase.assertAlmostEqual is removed from torch1.8.1
        prec = delta
        if places:
            prec = 10**(-places)
        rtol = None if prec is None else 0
        self.assertEqual(x, y, msg=msg, atol=prec, rtol=rtol)

    def assertEqual(self, x, y, *args, **kwargs):
        # assertEqual defined in TorchTestCase class could not handle
        # this well, fallback to assertEqual in unittest.TestCase
        if self._iter_but_no_len(x) or self._iter_but_no_len(y):
            # extract "message" argument from
            # `TorchTestCase.assertEqual` and pass it to
            # `unittest.TestCase.assertEqual` as "msg" argument.
            # TorchTestCase.assertEqual signature:
            #   assertEqual(self, x, y, prec=None, message='', allow_inf=False,
            #               exact_dtype=None)
            if args and len(args) >= 2:
                msg = args[1]
                args = args[:1] + args[2:]
            else:
                msg = kwargs.pop('message', None)
            return unittest.TestCase.assertEqual(self, x, y, msg=msg)
        else:
            return super().assertEqual(x, y, *args, **kwargs)


def benchmark_call(num_warmup_iters=10, num_iters=20, sync_call=None):
    def decorator(func):
        @torch.no_grad()
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for iter in range(num_warmup_iters):
                outputs = func(*args, **kwargs)
            if (sync_call is not None):
                sync_call()
            start_time = time.perf_counter()
            for iter in range(num_iters):
                outputs = func(*args, **kwargs)
            if (sync_call is not None):
                sync_call()
            end_time = time.perf_counter()
            total_elapsed = end_time - start_time
            stats = namedtuple(
                'BenchmarkStats',
                ['latency_avg_ms', 'num_warmup_iters', 'num_iters', 'outputs'])
            stats.num_warmup_iters = num_warmup_iters
            stats.num_iters = num_iters
            stats.outputs = outputs
            stats.latency_avg_ms = total_elapsed / stats.num_iters * 1000.0
            return stats

        return wrapper

    return decorator


def benchmark(model, inputs, num_warmup_iters, num_iters):
    if (not isinstance(inputs, tuple)):
        inputs = (inputs, )

    @benchmark_call(num_warmup_iters, num_iters, torch.cuda.synchronize)
    def _benchmark():
        return model.forward(*inputs)

    return _benchmark()


def assert_almost_equal(x, y, rtol=None, atol=None):
    if (isinstance(x, bool) and isinstance(y, bool)) or \
       (isinstance(x, string_classes) and isinstance(y, string_classes)):
        assert x == y
    elif (isinstance(x, Number) and isinstance(y, Number)) or \
         (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        assert_allclose(x, y, rtol=rtol, atol=atol)
    elif (isinstance(x, dict) and isinstance(y, dict)):
        assert (sorted(x.keys()) == sorted(y.keys()))
        key_list = sorted(x.keys())
        for key in key_list:
            try:
                assert_almost_equal(x[key], y[key], rtol=rtol, atol=atol)
            except AssertionError as error:
                err_msg = re.split(" Error occurred during comparing (list|dict) elements", str(error))[0]
                raise AssertionError(
                    "%s Error occurred during comparing dict elements %s" % (err_msg, key))
    elif (is_iterable(x) and is_iterable(y)):
        for idx, (a, b) in enumerate(zip(x, y)):
            try:
                assert_almost_equal(a, b, rtol=rtol, atol=atol)
            except AssertionError as error:
                err_msg = re.split(" Error occurred during comparing (list|dict) elements", str(error))[0]
                raise AssertionError(
                    "%s Error occurred during comparing list elements %s" % (err_msg, idx))
    else:
        raise ValueError("Unsupported comparision between %s and %s" %
                         (type(x), type(y)))
