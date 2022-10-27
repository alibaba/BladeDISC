include(ExternalProject)

ExternalProject_Add(json_extern
  BUILD_IN_SOURCE 1
  URL "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/nlohmann_json_lib/v3.6.1-stripped.tar.gz"
  DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/third_party/json"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/json"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND  ""
  BUILD_COMMAND ""
  INSTALL_COMMAND "")