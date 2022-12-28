transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %loops:3 = transform.structured.tile %0 [2, 3, 4]
}