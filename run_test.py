import subprocess
import numpy as np
import torch
import os
import time

def run_python_file(python_file):
    try:
        result = subprocess.run(['python', python_file], check=True)
        print(f"Python file '{python_file}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{python_file}': {e}")

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Bash command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing bash command '{command}': {e}")

def compare_npy_files(file1, file2):
    try:
        array1 = np.load(file1)
        array2 = np.load(file2)

        print(array1.shape)
        print(array2.shape)

        print(array1[0][0][0])
        print(array2[0][0][0])

        torch.testing.assert_close(array1, array2)

        if np.array_equal(array1, array2):
            print(f"Files '{file1}' and '{file2}' are identical.")
        else:
            print(f"Files '{file1}' and '{file2}' differ.")
    except Exception as e:
        print(f"Error comparing files '{file1}' and '{file2}': {e}")

if __name__ == "__main__":
    iree_dir = '../'
    python_file_1 = 'generate_npys.py'
    python_file_2 = 'generate_mlir.py'
    npy_file_1 = 'npys/attn_out.npy'
    npy_file_2 = 'npys/attn_ref.npy'

    run_bash_command('rm -rf npys')
    run_bash_command('rm -rf fused_attn.vmfb')
    run_bash_command('mkdir npys')
    run_python_file(python_file_1)
    run_python_file(python_file_2)

    bash_command_1 = f'{iree_dir}build/tools/iree-compile test_attn.mlir --iree-experimental-packed-i1-storage --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-global-opt-propagate-transposes=true --iree-opt-outer-dim-concat=true --iree-opt-const-eval=false --iree-opt-data-tiling=false --iree-vm-target-truncate-unsupported-floats -o fused_attn.vmfb'
    bash_command_2 = f"{iree_dir}build/tools/iree-run-module --module=fused_attn.vmfb --input=@npys/attn_q.npy --input=@npys/attn_k.npy --input=@npys/attn_v.npy {'--input=@npys/attn_mask.npy ' if os.path.exists('npys/attn_mask.npy') else ''}--output=@npys/attn_out.npy"
    run_bash_command(bash_command_1)
    run_bash_command(bash_command_2)
    compare_npy_files(npy_file_1, npy_file_2)
