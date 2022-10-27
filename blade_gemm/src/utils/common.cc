
#include "bladnn/utils/common.h"

namespace bladnn {
namespace utils {
bool startswith(const std::string& str, const std::string& prefix) {
  return str.compare(0, prefix.length(), prefix) == 0;
}
}  // namespace utils
}  // namespace bladnn