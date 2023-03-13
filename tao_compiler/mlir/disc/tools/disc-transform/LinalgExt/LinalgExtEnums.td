#ifndef DISC_LINALGEXT_ENUMS
#define DISC_LINALGEXT_ENUMS

include "mlir/IR/EnumAttr.td"

def DISCLinalgExt_Any: I64EnumAttrCase<"kAny", 0>; // padding value doesn't matter.
def DISCLinalgExt_Exact: I64EnumAttrCase<"kExact", 1>; // padding value matters.
def DISCLinalgExt_PaddingValueMode : I64EnumAttr<"PaddingValueModeEnum",
    "Padding mode for padding related-ops",
    [
        DISCLinalgExt_Any,
        DISCLinalgExt_Exact
    ]> {
  let summary = "Padding mode for padding related-ops";
  let cppNamespace = "::mlir::disc_ral::disc_linalg_ext";
}

#endif  // DISC_LINALGEXT_ENUMS