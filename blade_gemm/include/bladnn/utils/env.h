#pragma once
#include <string>

namespace bladnn {
namespace utils {

bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val,
                        bool* value);

bool ReadStringFromEnvVar(const char* env_var_name,
                          const std::string& default_val, std::string* value);

}  // namespace utils
}  // namespace bladnn
