# Copyright (c) Facebook, Inc. and its affiliates.

import random
import torch
from torch import Tensor, nn

from detectron2 import model_zoo
from detectron2.export import scripting_with_instances
from detectron2.export.flatten import TracingAdapter
from detectron2.structures import Boxes
from detectron2.utils.testing import (
    get_sample_coco_image,
)
import argparse
import torch_blade
import torch_blade.tensorrt as torch_blade_trt
import pandas as pd
import numpy as np
import timeit
import inspect

"""
https://detectron2.readthedocs.io/tutorials/deployment.html
contains some explanations of this file.
"""

results = []


def printStats(func_name, backend, timings, precision, batch_size=1):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = (
        "\n%s =================================\n"
        "batch size=%d, num iterations=%d\n"
        "  Median FPS: %.1f, mean: %.1f\n"
        "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
    ) % (
        backend,
        batch_size,
        steps,
        speed_med,
        speed_mean,
        time_med,
        time_mean,
        time_99th,
        time_std,
    )
    print(msg)
    meas = {
        "Model": func_name[4:],
        "Backend": backend,
        "precision": precision,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median(ms)": time_med,
        "Mean(ms)": time_mean,
        "99th_p": time_99th,
        "std_dev": time_std,
    }
    results.append(meas)


@torch.no_grad()
def benchmark(func_name, backend, enable_fp16, model, inp, batch_size):
    import torch_blade.utils as utils

    utils.disable_pytorch_jit()
    for i in range(100):
        model(*inp)
    torch.cuda.synchronize()
    timings = []
    for i in range(200):
        start_time = timeit.default_timer()
        model(*inp)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)
        print("Iteration {}: {:.6f} s".format(i, end_time - start_time))

    precision = "fp16" if enable_fp16 else "fp32"
    printStats(func_name, backend, timings, precision, batch_size)


# TODO: this test requires manifold access, see: T88318502
class TestDetectron2:
    def testRetinaNet_scripted(self):
        def load_retinanet(config_path):
            model = model_zoo.get(config_path, trained=True).eval()
            fields = {
                "pred_boxes": Boxes,
                "scores": Tensor,
                "pred_classes": Tensor,
            }
            script_model = scripting_with_instances(model, fields)
            return model, script_model

        image = get_sample_coco_image()
        small_image = nn.functional.interpolate(
            image, scale_factor=random.uniform(0.5, 0.7)
        )
        large_image = nn.functional.interpolate(
            image, scale_factor=random.uniform(2, 3)
        )

        inputs = ([{"image": image}],)
        small_inputs = ([{"image": small_image}],)
        large_inputs = ([{"image": large_image}],)
        model, script_model = load_retinanet(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        )

        batch = 1
        func_name = inspect.stack()[0][3]
        with torch.no_grad():
            self._benchmark(
                func_name,
                model,
                script_model,
                inputs,
                small_inputs,
                large_inputs,
                batch,
                enable_fp16=True,
            )
            self._benchmark(
                func_name,
                model,
                script_model,
                inputs,
                small_inputs,
                large_inputs,
                batch,
                enable_fp16=False,
            )

    def testMaskRCNNFPN(self):
        def inference_func(model, image):
            inputs = [{"image": image.float()}]
            return model.inference(inputs, do_postprocess=False)[0]

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name,
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            inference_func,
        )

    def testMaskRCNNFPN_pproc(self):
        def inference_func(model, image):
            inputs = [
                {"image": image, "height": image.shape[1], "width": image.shape[2]}
            ]
            return model.inference(inputs, do_postprocess=True)[0]["instances"]

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name,
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            inference_func,
        )

    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image.float()}]
            return model.inference(inputs, do_postprocess=False)[0]

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name,
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
            inference_func,
        )

    def testCascadeRCNN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name, "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    def gen_shapes(self, batch, inputs):
        input_shapes = [[batch] + list(inp.shape)[1:] for inp in inputs]
        return input_shapes

    # bug fixed by https://github.com/pytorch/pytorch/pull/67734
    def testRetinaNet(self):
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name, "COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func
        )

    def testMaskRCNNFPN_b2(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        func_name = inspect.stack()[0][3]
        self._test_model(
            func_name,
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            inference_func,
            batch=2,
        )

    def _test_model(self, func_name, config_path, inference_func, batch=1):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        # trace with smaller images, and the trace must still work
        small_inputs = tuple(
            nn.functional.interpolate(image, scale_factor=random.uniform(0.5, 0.7))
            for _ in range(batch)
        )
        large_inputs = tuple(
            nn.functional.interpolate(image, scale_factor=random.uniform(2, 3))
            for _ in range(batch)
        )

        wrapper = TracingAdapter(model, inputs, inference_func)
        wrapper.eval()

        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, inputs)
            self._benchmark(
                func_name,
                wrapper,
                traced_model,
                inputs,
                small_inputs,
                large_inputs,
                batch,
                enable_fp16=True,
            )
            self._benchmark(
                func_name,
                wrapper,
                traced_model,
                inputs,
                small_inputs,
                large_inputs,
                batch,
                enable_fp16=False,
            )

    def _benchmark(
        self,
        func_name,
        wrapper,
        traced_model,
        inputs,
        small_inputs,
        large_inputs,
        batch,
        enable_fp16,
    ):
        with torch.no_grad():
            config = torch_blade.config.Config()
            config.dynamic_tuning_inputs = {
                "min": small_inputs,
                "max": large_inputs,
                "opts": [inputs],
            }
            config.optimization_pipeline = torch_blade_trt.backend_name()
            config.enable_fp16 = enable_fp16
            config.enable_onnx_shape_white_list = False
            with config:
                blade_model = torch_blade.optimize(traced_model, False, inputs)

            with torch.cuda.amp.autocast(enable_fp16):
                benchmark(func_name, "Torch", enable_fp16, wrapper, inputs, batch)
                benchmark(
                    func_name, "TorchBlade", enable_fp16, blade_model, inputs, batch
                )
            # with open('trace.code.py', 'w') as f: f.write(traced_model.code)
            with open(f"{func_name}.code.py", "w") as f:
                f.write(blade_model.code)
            with open(f"{func_name}.graph.txt", "w") as f:
                f.write(str(blade_model.graph))


if __name__ == "__main__":
    tracing = TestDetectron2()
    parser = argparse.ArgumentParser(description="TorchBlade Detectron2 Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        help="model",
        required=True,
        choices=[f for f in dir(tracing) if f.startswith("test")],
    )
    args = parser.parse_args()
    tracing.__getattribute__(f"{args.model}")()

    # Generate report
    print("Model Summary:")
    summary = pd.DataFrame(results)
    print(summary.to_markdown())
