"""Run the notebook's actual kernels on CPU via Triton's interpreter.

Linux setup:
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install triton numpy jaxtyping
  python test_puzzles_9_11.py
"""
import os
os.environ['TRITON_INTERPRET'] = '1'

import ast
import json
from pathlib import Path
import tempfile
import importlib.util

import torch
import triton


def load_notebook_kernels():
    notebook = json.loads(Path(__file__).with_name('Triton-Puzzles.ipynb').read_text())
    names = {'flashatt_kernel', 'conv2d_kernel', 'dot_kernel',
             'flashatt_spec', 'conv2d_spec', 'dot_spec'}
    source = 'import torch\nimport triton\nimport triton.language as tl\nfrom torch import Tensor\nfrom jaxtyping import Float32\n\n'
    found = set()
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or not any('def ' + name + '(' in text for name in names):
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                source += ast.unparse(node) + '\n\n'
                found.add(node.name)
    assert found == names
    # Triton inspects function source; give it a real temporary module.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'notebook_kernels.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('notebook_kernels', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def check(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def run(k):
    torch.manual_seed(0)
    count = 0
    # Original notebook shapes, plus tails, rectangular dimensions and B2 > 1.
    for n, t, block, scale in [(200, 200, 64, 1), (7, 13, 16, 1), (65, 129, 32, 1), (17, 79, 16, 30)]:
        q = (torch.rand(n) - .5) * scale
        key = (torch.rand(t) - .5) * scale
        v = torch.rand(t) - .5
        z = torch.full((n,), float('nan'))
        k.flashatt_kernel[(triton.cdiv(n, block),)](q, key, v, z, n, t, B0=block)
        check(z, k.flashatt_spec(q, key, v))
        count += 1
    print('Puzzle 9: 4 cases passed')
    for n, h, w, kh, kw, block in [(4, 8, 8, 4, 4, 1), (3, 5, 7, 2, 3, 2), (1, 2, 3, 4, 5, 1), (2, 1, 1, 1, 1, 4)]:
        x = torch.rand(n, h, w) - .5
        filt = torch.rand(kh, kw) - .5
        z = torch.full_like(x, float('nan'))
        k.conv2d_kernel[(triton.cdiv(n, block),)](x, filt, z, n, h, w, kh, kw, B0=block)
        expected = torch.nn.functional.conv2d(torch.nn.functional.pad(x[:, None], (0, kw-1, 0, kh-1)), filt[None, None])[:, 0]
        check(z, expected)
        if (n, h, w, kh, kw) == (4, 8, 8, 4, 4):
            check(z, k.conv2d_spec(x, filt))
        count += 1
    print('Puzzle 10: 4 cases passed')
    for batch, rows, cols, mid, b2 in [(4, 32, 32, 32, 1), (3, 19, 23, 37, 2), (1, 1, 1, 1, 1), (5, 33, 17, 9, 3)]:
        x = torch.rand(batch, rows, mid) - .5
        y = torch.rand(batch, mid, cols) - .5
        z = torch.full((batch, rows, cols), float('nan'))
        k.dot_kernel[(triton.cdiv(rows, 16), triton.cdiv(cols, 16), triton.cdiv(batch, b2))](x, y, z, rows, cols, batch, mid, B0=16, B1=16, B2=b2, B_MID=16)
        check(z, k.dot_spec(x, y))
        count += 1
    print('Puzzle 11: 4 cases passed')
    print(f'All {count} tests passed.')


if __name__ == '__main__':
    for kernels in load_notebook_kernels():
        run(kernels)
