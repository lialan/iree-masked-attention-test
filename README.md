# HI!

This repo is intended as a flexible numerical test for `iree_linalg_ext.attention`, including its masking features.

To run the test, first setup a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```
run_test.py
```

This script:
- Creates all query, key, value, mask, and output arrays using `generate_npys.py`
- Generates a simple test MLIR script based on these arrays using `generate_mlir.py`
- Compiles this MLIR script via `iree-compile` to create a `fused_attn.vmfb`
- Runs this VMFB file via `iree-run-module`
- Compares the "golden-value" tensor (created through `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)`) and the MLIR-generated output tensor.

To change the specific contents and shapes for the query, key, value, mask, and output tensors, modify `generate_npys.py`. To specify the your IREE build directory, modify `iree_dir` within `run_test.py`.