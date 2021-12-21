list(APPEND MLIR_HEADERS
    "mlir_executable.h"
)

list(APPEND MLIR_SOURCES
    "mlir_executable.cc"
)

add_library(mlir OBJECT ${MLIR_SOURCES})
