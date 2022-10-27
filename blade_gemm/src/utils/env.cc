#include "bladnn/utils/env.h"

#include "bladnn/utils/log.h"

namespace bladnn {
namespace utils {

bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val,
                        bool* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(env_var_name);
  if (tf_env_var_val == nullptr) {
    return true;
  }
  const std::string& str_value = std::string(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return true;
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return true;
  }
  BLADNN_LOG(FATAL) << "Failed to parse the env-var "
                    << std::string(env_var_name) << " into bool: " << str_value
                    << ". Use the default value: " << default_val;
  return false;
}

bool ReadStringFromEnvVar(const char* env_var_name,
                          const std::string& default_val, std::string* value) {
  const char* tf_env_var_val = getenv(env_var_name);
  if (tf_env_var_val != nullptr) {
    *value = std::string(tf_env_var_val);
  } else {
    *value = default_val;
  }
  return true;
}

}  // namespace utils
}  // namespace bladnn
