# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1

`python project/parallel_check.py` loop listings for map/zip/reduce (`prange` on the main loops; aligned map/zip skip indexing):

```
MAP:  for i in prange(size):  # aligned #2 and general #3
ZIP:  for i in prange(size):  # aligned #7 and general #8
REDUCE: for i in prange(size):  # #10; inner reduce has no function calls
```

NUMBA reports those regions as parallel after optimisation. Full `parallel_diagnostics(level=3)` output is from a local run of the official script's MAP/ZIP/REDUCE sections.

## Task 3.2

`python project/parallel_check.py` MATRIX MULTIPLY listing:

```
for p in prange(batch * rows * cols):  # loop #0, parallel structure already optimal
    ...
    for k in range(inner):
        acc += a[...] * b[...]
    out[...] = acc
```