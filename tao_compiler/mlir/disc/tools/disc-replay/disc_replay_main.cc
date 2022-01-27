#include "tensorflow/compiler/mlir/disc/tools/disc-replay/cluster_args.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"

int main(int argc, char** argv) {
  std::string tao_input_fname = "tao_input.pb";
  std::string record_fname = "record.pb";
  replay::DiscInterpreter inter;
  TF_RETURN_IF_ERROR(inter.Run(tao_input_fname, record_fname));
}