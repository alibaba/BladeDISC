from glob import glob
import shutil
import disc_opt
import pytest
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os, tempfile

ANSORTUNE_LOG = "profile_cache/kernel_tune.json"
AUTOTVMTUNE_LOG = "profile_cache/kernel_tune.log"

class Network():
    def __init__(self, x_shape, y_shape, ta, tb, dtype=tf.float64):
        self.x = tf.placeholder(shape=x_shape,dtype=dtype)
        self.y = tf.placeholder(shape=y_shape,dtype=dtype)  
        self.output = tf.matmul(self.x, self.y, transpose_a = ta, transpose_b = tb)


class TestDiscOpt:
    def setup_class(self):
        self._tmp = tempfile.mkdtemp()

    def teardown_class(self):
        shutil.rmtree(self._tmp)


    def _opt1(self, m, n, k, transa, transb, dtype=tf.float64):
        x_shape = [m, k] if not transa else [k, m]
        y_shape = [k, n] if not transb else [n, k]
        net = Network(x_shape, y_shape, transa, transb, dtype)
        x_data = np.linspace(-1,1, m*k).reshape(*x_shape)
        y_data = np.linspace(-1,1, k*n).reshape(*y_shape)
        input_dict = {net.x : x_data, net.y : y_data} 
        config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        disc_ctx = disc_opt.DiscOptItertionContext(optlevel=disc_opt.OptLevel.O2, limit=1, degree=[1, 1, 1])
        os.environ["BLADNN_KERNEL_CACHE"] =  self._tmp
        for i in range(100):
            with disc_ctx:
                if i == 0:
                    res_ori = sess.run(net.output,feed_dict=input_dict)
                    print(res_ori)
                else:
                    res_opt = sess.run(net.output,feed_dict=input_dict)
        key = "*_{}_{}_{}_{}_{}_0_{}_*".format(m, n, k, 1 if transa else 0, 1 if transb else 0, "float64" if dtype==tf.float64 else "float32")
        opt_kernels = glob("{}/{}.hsaco".format(self._tmp, key))
        opt_kernels_meta = glob("{}/{}.meta_for_tao.json".format(self._tmp, key))
        assert len(opt_kernels) == 1
        assert len(opt_kernels_meta) == 1
        np.testing.assert_almost_equal(res_ori, res_opt)


    def test_opt(self):
        self._opt1(64, 240, 1600, False, False)
    

if __name__ == "__main__":
    pytest.main(["-s", "test_discopt.py"])    

        
        

