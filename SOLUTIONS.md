# Puzzles 9–11

Implementations are in the original `Triton-Puzzles.ipynb` cells. Earlier and later puzzles are unchanged.

- **9 — Simple FlashAttention:** Process a block of queries against successive key/value blocks. Track the running maximum, exponential sum, and weighted-value sum; rescale both sums when the maximum changes. This avoids storing the full attention matrix and handles the final partial block.
- **10 — Two Dimensional Convolution:** Each program handles a batch block and iterates over spatial tiles. Accumulate each kernel offset, masking bottom/right reads to implement zero padding. This follows the reference's cross-correlation convention (no filter reversal).
- **11 — Matrix Multiplication:** Use all three grid axes for row, column, and batch blocks. Accumulate `tl.dot` over reduction tiles, masking partial rows, columns, batches, and reduction tiles. Use IEEE float32 inputs for the dot product.

## Testing

On Linux with Python 3.11:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install triton==3.4.0 numpy==1.26.4 jaxtyping
python test_puzzles_9_11.py
```

The test script extracts the actual functions from the notebook and executes them using `TRITON_INTERPRET=1`. It checks the original examples plus partial blocks, rectangular shapes, filters larger than the image, multiple batches per program, and large attention scores (12 cases total). Assertions use the notebook's `rtol=1e-3, atol=1e-3` tolerances. GitHub Actions runs the same checks. These are correctness tests; they do not establish GPU performance.

For the original interactive visualization, run the notebook's installation/setup cells, then the cells for puzzles 9, 10, and 11. Unsolved earlier exercises do not need to be executed.
