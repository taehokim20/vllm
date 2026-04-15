#!/usr/bin/env python3
"""Build only the _lora_C shared library from MoE BGMV CUDA sources."""
import torch, os, subprocess, glob, sys

torch_dir = os.path.dirname(torch.__file__)
include_dirs = [
    '/workspace/csrc',
    os.path.join(torch_dir, 'include'),
    os.path.join(torch_dir, 'include/torch/csrc/api/include'),
    '/usr/local/cuda/include',
]
inc_flags = ' '.join(f'-I{d}' for d in include_dirs)

cu_files = sorted(glob.glob('/workspace/csrc/lora/bgmv_moe_cuda/*.cu')) + \
           ['/workspace/csrc/lora/moe_lora_ops.cu']

obj_files = []
for cu in cu_files:
    obj = os.path.basename(cu).replace('.cu', '.o')
    cmd = (f'/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler '
           f'-DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_lora_C '
           f'-D_lora_C_EXPORTS '
           f'{inc_flags} '
           f'-DONNX_NAMESPACE=onnx_c2 '
           f'--expt-relaxed-constexpr --expt-extended-lambda '
           f'-O2 -DNDEBUG -std=c++17 -Xcompiler=-fPIC '
           f'-DENABLE_FP8 '
           f'-gencode arch=compute_80,code=sm_80 '
           f'-gencode arch=compute_89,code=sm_89 '
           f'-gencode arch=compute_90,code=sm_90 '
           f'-c {cu} -o {obj}')
    print(f'Compiling {os.path.basename(cu)}...', flush=True)
    subprocess.check_call(cmd, shell=True)
    obj_files.append(obj)

# Link
torch_lib = os.path.join(torch_dir, 'lib')
objs = ' '.join(obj_files)
out = '_lora_C.abi3.so'
cmd = (f'g++ -shared -o {out} {objs} '
       f'-L{torch_lib} -ltorch -ltorch_python -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda '
       f'-L/usr/local/cuda/lib64 -lcudart '
       f'-Wl,-rpath,{torch_lib}')
print('Linking...', flush=True)
subprocess.check_call(cmd, shell=True)

# Install
site_pkg = os.path.dirname(torch_dir.rstrip('/').rsplit('/', 1)[0] + '/../vllm/__init__.py')
# More robust: find vllm package location
import importlib
vllm_dir = os.path.dirname(importlib.import_module('vllm').__file__)
dest = os.path.join(vllm_dir, out)
subprocess.check_call(['cp', out, dest])
print(f'Installed {out} -> {dest}', flush=True)
