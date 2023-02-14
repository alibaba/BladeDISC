// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoupling/mlir_compiler.h"

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/disc/disc_compiler.h"
#include "mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tao {

using llvm::SmallVector;
using mlir::DenseElementsAttr;
using mlir::RankedTensorType;

StatusOr<std::vector<TensorShape>> ParseArgShapes(
    const TaoCompilerInput& input) {
  std::vector<TensorShape> args;
  args.reserve(input.args_size());
  for (int i = 0; i < input.args_size(); ++i) {
    auto& arg = input.args(i);
    tensorflow::TensorShapeProto tensor_shape_proto;
    if (!tensor_shape_proto.ParseFromString(arg.shape())) {
      return tensorflow::errors::Internal("Parse failed for arg shape");
    }
    args.push_back(tensorflow::TensorShape(tensor_shape_proto));
    for (int j = 0; j < args.back().dims(); ++j) {
      VLOG(0) << "arg #" << i << " dim #" << j << ": "
              << args.back().dim_size(j);
    }
  }
  return args;
}

StatusOr<std::unordered_map<std::string, PartialTensorShape>>
ParseKnownArgShapes(const TaoCompilerInput& input) {
  std::unordered_map<std::string, PartialTensorShape> arg_shapes;
  FunctionDefLibrary input_flib_def;
  if (!input_flib_def.ParseFromString(input.options().flib_def())) {
    return tensorflow::errors::Internal("Parse failed for input flib_def");
  }
  auto func_def = input_flib_def.function(0);
  if (func_def.attr().count("input_args_info") == 0) {
    return arg_shapes;
  }
  auto attr_value = func_def.attr().at("input_args_info");
  if (!attr_value.has_func()) {
    return arg_shapes;
  }
  auto name_attr_list = attr_value.func();
  if (name_attr_list.name() != "shape_info") {
    return arg_shapes;
  }
  for (auto iter : name_attr_list.attr()) {
    arg_shapes[iter.first] = PartialTensorShape(iter.second.shape());
    VLOG(2) << "KnownArgShapes: " << iter.first << ", "
            << arg_shapes[iter.first];
  }
  return arg_shapes;
}

Status ConvertInputInfo(const TaoCompilerInput& input, Graph* graph,
                        GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  std::vector<std::string> data_types;
  std::vector<std::vector<int>> shapes;

  TF_ASSIGN_OR_RETURN(auto arg_shapes, ParseArgShapes(input));
  TF_ASSIGN_OR_RETURN(auto known_arg_shapes, ParseKnownArgShapes(input));

  for (Node* n : graph->op_nodes()) {
    VLOG(2) << "ConvertInputInfo: " << n->type_string() << "@" << n->name();
    if (n->type_string() == "_Arg") {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      VLOG(2) << "_Arg " << index << ", " << n->name();
      if (array_names.size() <= index) {
        array_names.resize(index + 1);
        data_types.resize(index + 1);
        shapes.resize(index + 1);
      }
      array_names[index] = n->name();
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
      data_types[index] = (dtype == DT_INVALID ? "" : DataType_Name(dtype));
      // Force to codegen dynamic shape (not dynamic rank) code
      std::vector<int> dims(arg_shapes[index].dims(), -1);
      if (known_arg_shapes.count(absl::AsciiStrToLower(n->name())) > 0) {
        auto known_shape = known_arg_shapes[absl::AsciiStrToLower(n->name())];
        for (auto i = 0; i < dims.size(); i++) {
          dims[i] = known_shape.dim_size(i);
        }
      }
      shapes[index] = std::move(dims);
      for (int i = 0; i < shapes[index].size(); ++i) {
        VLOG(2) << "input #" << index << " dim #" << i << ": "
                << shapes[index][i];
      }
    }
  }
  std::vector<llvm::Optional<std::vector<int>>> optional_shapes;
  for (auto& shape : shapes) optional_shapes.emplace_back(shape);
  return ParseInputArrayInfo(array_names, data_types, optional_shapes,
                             &specs->inputs);
}

Status ConvertOutputInfo(Graph* graph, GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == "_Retval") {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      const Edge* e = nullptr;
      TF_RETURN_IF_ERROR(n->input_edge(0, &e));
      if (array_names.size() <= index) {
        array_names.resize(index + 1);
      }
      array_names[index] = absl::StrCat(e->src()->name(), ":", e->src_output());
    }
  }
  return ParseOutputArrayInfo(array_names, &specs->outputs);
}

mlir::Type DataTypeToMlirType(mlir::OpBuilder b, DataType dtype) {
  if (dtype == DataType::DT_FLOAT) {
    return b.getF32Type();
  } else if (dtype == DataType::DT_DOUBLE) {
    return b.getF64Type();
  } else if (dtype == DataType::DT_HALF) {
    return b.getF16Type();
  } else if (dtype == DataType::DT_INT64) {
    return b.getIntegerType(64);
  } else if (dtype == DataType::DT_INT32) {
    return b.getIntegerType(32);
  } else if (dtype == DataType::DT_BOOL) {
    return b.getIntegerType(1);
  } else {
    LOG(FATAL) << "Unimplemented DataTypeToMlirType conversion";
  }
}

Status AppendIOAttr(mlir::ModuleOp module, const GraphImportConfig& specs,
                    const TaoCompilerInput& input,
                    const std::string& default_device) {
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
  auto dict_attr =
      main_func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  assert(dict_attr && "main_func must has tf.entry_function attr");
  SmallVector<mlir::NamedAttribute, 2> attributes;
  for (auto attr : dict_attr) {
    attributes.push_back(attr);
  }
  mlir::OpBuilder builder(module);
  SmallVector<mlir::StringRef, 4> input_placements;
  SmallVector<mlir::StringRef, 4> output_placements;

  for (int i = 0; i < specs.inputs.size(); ++i) {
    auto& arg_proto = input.args(i);
    if (arg_proto.kind_v2() == ArgumentKind::kConstant) {
      // compile_time_const
      input_placements.push_back("const");
    } else if (arg_proto.kind_v2() == ArgumentKind::kFixedShaped) {
      input_placements.push_back("cpu");
    } else if (arg_proto.kind_v2() == ArgumentKind::kHostArgs) {
      input_placements.push_back("cpu");
    } else {
      input_placements.push_back(default_device);
    }
  }
  for (int i = 0; i < specs.outputs.size(); ++i) {
    if (input.options().output_placements_size() > i) {
      output_placements.push_back(input.options().output_placements(i));
    } else {
      output_placements.push_back(default_device);
    }
  }
  attributes.push_back(builder.getNamedAttr(
      "input_placements",
      builder.getStringAttr(llvm::join(input_placements, ","))));
  attributes.push_back(builder.getNamedAttr(
      "output_placements",
      builder.getStringAttr(llvm::join(output_placements, ","))));

  // extract const inputs info
  for (int i = 0; i < specs.inputs.size(); ++i) {
    auto& arg_proto = input.args(i);
    if (arg_proto.kind_v2() == ArgumentKind::kConstant) {
      auto attr_name =
          (mlir::disc_ral::kDhloInputValueAttr + ("_" + llvm::Twine(i))).str();
      TensorProto tensor_proto;
      if (!tensor_proto.ParseFromString(arg_proto.constant_value())) {
        return tensorflow::errors::Internal(
            "Mlir parse failed for arg constant value");
      }
      Tensor constant_tensor;
      constant_tensor.FromProto(tensor_proto);
      DenseElementsAttr attr;
      auto elem_type = DataTypeToMlirType(builder, DataType(arg_proto.type()));
      SmallVector<int64_t, 4> shape;
      for (int dim = 0; dim < constant_tensor.dims(); ++dim) {
        shape.push_back(constant_tensor.dim_size(dim));
      }
      auto shaped_type = mlir::RankedTensorType::get(shape, elem_type);
      if (arg_proto.type() == DataType::DT_FLOAT) {
        VLOG(0) << "Warning: usually there shouldn't be float const inputs";
        auto data = constant_tensor.flat<float>();
        attr = DenseElementsAttr::get(
            shaped_type, llvm::makeArrayRef(data.data(), data.size()));
      } else if (arg_proto.type() == DataType::DT_INT64) {
        auto data = constant_tensor.flat<int64>();
        attr = DenseElementsAttr::get(
            shaped_type, llvm::makeArrayRef(data.data(), data.size()));
      } else if (arg_proto.type() == DataType::DT_INT32) {
        auto data = constant_tensor.flat<int32>();
        attr = DenseElementsAttr::get(
            shaped_type, llvm::makeArrayRef(data.data(), data.size()));
      } else if (arg_proto.type() == DataType::DT_BOOL) {
        auto data = constant_tensor.flat<bool>();
        attr = DenseElementsAttr::get(
            shaped_type, llvm::makeArrayRef(data.data(), data.size()));
      } else {
        return tensorflow::errors::Internal(
            "Mlir datatype not implemented for constant input");
      }
      attributes.push_back(builder.getNamedAttr(attr_name, attr));

    } else if (arg_proto.kind_v2() == ArgumentKind::kFixedShaped) {
      auto attr_name =
          (mlir::disc_ral::kDhloInputShapeAttr + ("_" + llvm::Twine(i))).str();
      SmallVector<int64_t, 4> input_shape;
      tensorflow::TensorShapeProto shape_proto;
      shape_proto.ParseFromString(arg_proto.shape());
      for (int dim = 0; dim < shape_proto.dim_size(); ++dim) {
        input_shape.push_back(shape_proto.dim(dim).size());
      }
      auto elem_tp = DataTypeToMlirType(builder, DataType(arg_proto.type()));
      auto type = RankedTensorType::get(input_shape, elem_tp);
      // a DenseElementsAttr with Splat zero value is to represent the
      // shape/dtype
      mlir::Attribute attr;
      if (elem_tp.isSignlessInteger()) {
        attr = DenseElementsAttr::get(type, mlir::IntegerAttr::get(elem_tp, 0));
      } else {
        attr = DenseElementsAttr::get(type, mlir::FloatAttr::get(elem_tp, 0));
      }
      attributes.push_back(builder.getNamedAttr(attr_name, attr));
    }
  }

  main_func->setAttr("tf.entry_function",
                     builder.getDictionaryAttr(attributes));
  return tsl::OkStatus();
}

CompilerMLIR::CompilerMLIR() {
  std::vector<std::string> str_opts = {"tao_compiler_main",
                                       "--mlir-elide-elementsattrs-if-larger",
                                       "8", "--mlir-print-debuginfo"};
  std::vector<const char*> c_opts;
  c_opts.reserve(str_opts.size());
  for (auto& opt : str_opts) c_opts.push_back(opt.c_str());

  int argc = static_cast<int>(c_opts.size());
  const char** argv = &c_opts[0];

  mlir::registerAllPasses();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm_init_.reset(new llvm::InitLLVM(argc, argv));
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Compiler\n");
}

CompilerMLIR::~CompilerMLIR() {}

Status CompilerMLIR::Init(const TaoCompilerInput& input,
                          const string& output_file) {
  return tsl::OkStatus();
}

Status CompilerMLIR::ConvertToMlir(const TaoCompilerInput& input,
                                   const string& output_file) {
  NameAttrList fn_name_attrs;
  if (!fn_name_attrs.ParseFromString(input.function())) {
    return tensorflow::errors::Internal("Parse failed for input function");
  }

  FunctionDefLibrary input_flib_def;
  if (!input_flib_def.ParseFromString(input.options().flib_def())) {
    return tensorflow::errors::Internal("Parse failed for input flib_def");
  }

  auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), input_flib_def);

  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, Env::Default(), nullptr,
                                        TF_GRAPH_DEF_VERSION, flib_def.get(),
                                        opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  FunctionLibraryRuntime::Handle func_handle;
  tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
  TF_RETURN_IF_ERROR(lib_runtime->Instantiate(fn_name_attrs.name(),
                                              AttrSlice(&fn_name_attrs.attr()),
                                              inst_ops, &func_handle));

  const FunctionBody* fbody = lib_runtime->GetFunctionBody(func_handle);
  std::unique_ptr<Graph> graph(
      new Graph(lib_runtime->GetFunctionLibraryDefinition()));
  CopyGraph(*fbody->graph, graph.get());
  GraphDef graph_def;
  graph->ToGraphDef(&graph_def);
  *graph_def.mutable_library() =
      lib_runtime->GetFunctionLibraryDefinition()->ToProto();

  std::string graph_def_str = graph_def.SerializeAsString();
  VLOG(2) << "GraphDef str:\n" << graph_def.DebugString();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  context_.reset(new mlir::MLIRContext(registry));
  auto& context = *context_;
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  specs.prune_unused_nodes = false;
  specs.convert_legacy_fed_inputs = false;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;

  TF_RETURN_IF_ERROR(ConvertInputInfo(input, graph.get(), &specs));
  TF_RETURN_IF_ERROR(ConvertOutputInfo(graph.get(), &specs));

  VLOG(2) << "Input size = " << specs.inputs.size()
          << ", Output size = " << specs.outputs.size();

  TF_ASSIGN_OR_RETURN(auto module, ConvertGraphdefToMlir(graph_def, debug_info,
                                                         specs, &context));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "MLIR Module BEGIN {\n";
    module->dump();
    VLOG(2) << "\nMLIR Module END }\n";
  }

  TF_RETURN_IF_ERROR(mlir::TF::RunBridgeWithStandardPipeline(
      *module, /*enable_logging=*/true, /*enable_inliner=*/true));

  TF_RETURN_IF_ERROR(AppendIOAttr(*module, specs, input, DefaultDevice()));

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "MLIR Module (after standard pipeline) BEGIN {\n";
    module->dump();
    VLOG(2) << "\nMLIR Module (after standard pipeline) END }\n";
  }

  TF_RETURN_IF_ERROR(::tensorflow::ConvertTF2MlirHlo(*module));
  if (VLOG_IS_ON(2)) {
    VLOG(0) << "MLIR Module (after tf2xla) BEGIN {\n";
    module->dump();
    VLOG(0) << "\nMLIR Module (after tf2xla) END }\n";
  }

  module_ = std::move(module);
  return tsl::OkStatus();
}

Status CompilerMLIR::FillDeviceInfo(
    mlir::disc_ral::DISCLoweringOptions& options) {
  return tsl::OkStatus();
}

Status CompilerMLIR::CompileMlirToExecutable(const TaoCompilerInput& input,
                                             const std::string& output_file) {
  std::string so_name = output_file + ".so";
  mlir::disc_ral::DISCLoweringOptions hlo_to_llvm_options(so_name);
  TF_RETURN_IF_ERROR(FillDeviceInfo(hlo_to_llvm_options));
  if (failed(::mlir::disc_ral::LowerHLOToSharedLibrary(*module_,
                                                       hlo_to_llvm_options))) {
    return errors::Internal("lower to mlir llvm failed");
  }

  auto& metadata_file = hlo_to_llvm_options.metadata_file_path;
  result_proto_.mutable_mlir()->set_so_lib_filename(so_name);
  result_proto_.mutable_mlir()->set_const_proto_filename(metadata_file);

  TF_RETURN_IF_ERROR(
      WriteTextProto(Env::Default(), output_file, result_proto_));

  return tsl::OkStatus();
}

Status CompilerMLIR::Compile(const TaoCompilerInput& input,
                             const string& output_file) {
  TF_RETURN_IF_ERROR(Init(input, output_file));
  TF_RETURN_IF_ERROR(ConvertToMlir(input, output_file));
  TF_RETURN_IF_ERROR(CompileMlirToExecutable(input, output_file));
  return tsl::OkStatus();
}

}  // namespace tao
}  // namespace tensorflow
