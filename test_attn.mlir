
    func.func @main(%q : tensor<1x1x2048x64xf16>, %k : tensor<1x1x256x64xf16>, %v : tensor<1x1x256x64xf16>, %m : tensor<1x1x2048x256xi8>) -> tensor<1x1x2048x64xf32> {
    %nm = tensor.empty() : tensor<1x1x2048x256xi1>
    %truncm = linalg.generic
    {indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%m: tensor<1x1x2048x256xi8>)
    outs(%nm: tensor<1x1x2048x256xi1>) {
      ^bb0(%in: i8, %out: i1):
        %truncated = arith.trunci %in : i8 to i1
        linalg.yield %truncated : i1
    } -> tensor<1x1x2048x256xi1>
    %cst = arith.constant 1.250000e-01 : f16
    %o = tensor.empty() : tensor<1x1x2048x64xf32>
    %r = iree_linalg_ext.attention {
        indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
        ]}
        ins(%q, %k, %v, %cst, %truncm: tensor<1x1x2048x64xf16>, tensor<1x1x256x64xf16>, tensor<1x1x256x64xf16>, f16, tensor<1x1x2048x256xi1>) outs(%o : tensor<1x1x2048x64xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
     } -> tensor<1x1x2048x64xf32>
    return %r : tensor<1x1x2048x64xf32>
    }
    