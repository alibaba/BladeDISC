#include "tensorflow/core/framework/op.h"

namespace tensorflow {
#ifdef PLATFORM_ALIBABA
REGISTER_OP("TaoLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("function: func")
    .Attr("fallback_function: func")
    .Attr("mlir_function: func")
    .Attr("inner: bool")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of XlaLaunch.
    .SetIsStateful()
    .Doc(R"(TaoLaunchOp supports static shape JIT, which only run in Alibaba "
"internal cluster.  TaoLaunchOp using XLA as the backend. The compilation execution
can be failed, thus this operator would use TF execution as the fallback.  The TaoLaunch
operator tried to compile and execution a larger(outter) cluster form the whole TF graph,
if failed, then we launch the TaoMlirLaunchOp to compile and execute a smaller(inner) cluster)");

REGISTER_OP("TaoMlirLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("fixedshapes: Tfixedshapes")
    .Attr("Tfixedshapes: list(type) >= 0")
    .Input("hostargs: Thostargs")
    .Attr("Thostargs: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("hostresults: Thostresults")
    .Attr("Thostresults: list(type) >= 0")
    .Output("deviceresults: Tdeviceresults")
    .Attr("Tdeviceresults: list(type) >= 0")
    .Attr("function: func")
    .Attr("fallback_function: func")
    .Attr("mlir_function: func")
    .Attr("inner: bool")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of XlaLaunch.
    .SetIsStateful()
    .Doc(
        R"(TaoMlirLaunchOp supports dynamic shape JIT, which only run in the
Alibaba internal cluster. TaoMlirLaunchOp using DISC as the backend.)");
#else
REGISTER_OP("DiscLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("fixedshapes: Tfixedshapes")
    .Attr("Tfixedshapes: list(type) >= 0")
    .Input("hostargs: Thostargs")
    .Attr("Thostargs: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("hostresults: Thostresults")
    .Attr("Thostresults: list(type) >= 0")
    .Output("deviceresults: Tdeviceresults")
    .Attr("Tdeviceresults: list(type) >= 0")
    .Attr("mlir_function: func")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of XlaLaunch.
    .SetIsStateful()
    .Doc(
        R"(DiscLaunchOp supports dynamic shape JIT, it is the open source implementation.
DiscLaunchOp using DISC as the backend.)");
#endif

} // namespace tensorflow
