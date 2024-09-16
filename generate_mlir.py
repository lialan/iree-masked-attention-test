import torch
import math
import numpy as np

import numpy as np
import math

def tensor_shape_and_type(tensor: np.ndarray) -> str:
    if tensor is None:
        return ""

    shape = 'x'.join(map(str, tensor.shape))
    dtype_map = {
        np.float16: "f16",
        np.float32: "f32",
        np.float64: "f64",
        np.int8: "i8",
        np.int16: "i16",
        np.int32: "i32",
        np.int64: "i64",
        np.uint8: "u8",
        np.bool_: "i1",
    }
    
    dtype_str = dtype_map.get(tensor.dtype.type, str(tensor.dtype))
    
    return f"tensor<{shape}x{dtype_str}>"

def generate_test_mlir(query: np.ndarray, key: np.ndarray, value: np.ndarray, output: np.ndarray, mask: np.ndarray = None):
    query_type_arg = tensor_shape_and_type(query)
    key_type_arg = tensor_shape_and_type(key)
    value_type_arg = tensor_shape_and_type(value)
    mask_type_arg = tensor_shape_and_type(mask)
    output_type_arg = tensor_shape_and_type(output)
    mask_arg = f", %m : {mask_type_arg}" if mask is not None else ""
    scale_value = 1 / math.sqrt(key.shape[-1])
    
    mlir = f'''
func.func @main(%q : {query_type_arg}, %k : {key_type_arg}, %v : {value_type_arg}{mask_arg}) -> {output_type_arg} {{
    %cst = arith.constant {scale_value:.6e} : f16
    %o = tensor.empty() : {output_type_arg}
    %r = iree_linalg_ext.attention {{
        indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
        {'affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,' if mask is not None else ''}
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
        ]}}
        ins(%q, %k, %v, %cst, %m: {query_type_arg}, {key_type_arg}, {value_type_arg}, f16{(', ' + mask_type_arg) if mask is not None else ""}) outs(%o : {output_type_arg}) -> {output_type_arg}
    return %r : {output_type_arg}
}}
    '''
    
    return mlir


with open('test_attn.mlir', 'w') as file:
    query = np.load('npys/attn_q.npy')
    key = np.load('npys/attn_k.npy')
    value = np.load('npys/attn_v.npy')
    mask = None
    try:
        mask = np.load('npys/attn_mask.npy')
    except:
        pass
    output = np.load('npys/attn_ref.npy')
    file.write(generate_test_mlir(query, key, value, output, mask=mask))