VERS_1.0 {
  global:
    # Expose all symbols in mlir namespace and several in llvm namespace.
    # Exposing too many of llvm namespace 
    extern "C++" {
        *mlir::*;
        *llvm::SmallVectorBase*;
        *llvm::detail::IEEEFloat*;
        *llvm::APFloatBase*;
        *llvm::detail::DoubleAPFloat*;
        *llvm::DisableABIBreakingChecks*;
        *llvm::raw_ostream*;
        *llvm::raw_string_ostream*;
        *llvm::report_fatal_error*;
        *llvm::APInt*;
        *llvm::deallocate_buffer*;
    };

  local:
    *;
};
