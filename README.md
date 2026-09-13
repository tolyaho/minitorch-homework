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

## Task 3.2

`python project/parallel_check.py` MATRIX MULTIPLY listing:

```
for p in prange(batch * rows * cols):  # loop #0, parallel structure already optimal
    ...
    for k in range(inner):
        acc += a[...] * b[...]
    out[...] = acc
```

<details>
<summary>Full parallel diagnostics</summary>
<pre>
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py (164)
--------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        in_storage: Storage,                                                          |
        in_shape: Shape,                                                              |
        in_strides: Strides,                                                          |
    ) -> None:                                                                        |
        size = 1                                                                      |
        for i in range(len(out_shape)):                                               |
            size *= out_shape[i]                                                      |
                                                                                      |
        aligned = len(out_shape) == len(in_shape)                                     |
        if aligned:                                                                   |
            for i in range(len(out_shape)):                                           |
                if out_shape[i] != in_shape[i] or out_strides[i] != in_strides[i]:    |
                    aligned = False                                                   |
                    break                                                             |
        if aligned:                                                                   |
            for i in prange(size):----------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                            |
            return                                                                    |
                                                                                      |
        for i in prange(size):--------------------------------------------------------| #3
            out_index = np.zeros(MAX_DIMS, np.int32)----------------------------------| #0
            in_index = np.zeros(MAX_DIMS, np.int32)-----------------------------------| #1
            to_index(i, out_shape, out_index)                                         |
            broadcast_index(out_index, out_shape, in_shape, in_index)                 |
            o = index_to_position(out_index, out_strides)                             |
            j = index_to_position(in_index, in_strides)                               |
            out[o] = fn(in_storage[j])                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(188) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(189) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(222)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py (222)
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        b_storage: Storage,                                            |
        b_shape: Shape,                                                |
        b_strides: Strides,                                            |
    ) -> None:                                                         |
        size = 1                                                       |
        for i in range(len(out_shape)):                                |
            size *= out_shape[i]                                       |
                                                                       |
        aligned = len(out_shape) == len(a_shape) == len(b_shape)       |
        if aligned:                                                    |
            for i in range(len(out_shape)):                            |
                if (                                                   |
                    out_shape[i] != a_shape[i]                         |
                    or out_shape[i] != b_shape[i]                      |
                    or out_strides[i] != a_strides[i]                  |
                    or out_strides[i] != b_strides[i]                  |
                ):                                                     |
                    aligned = False                                    |
                    break                                              |
        if aligned:                                                    |
            for i in prange(size):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                |
            return                                                     |
                                                                       |
        for i in prange(size):-----------------------------------------| #8
            out_index = np.zeros(MAX_DIMS, np.int32)-------------------| #4
            a_index = np.zeros(MAX_DIMS, np.int32)---------------------| #5
            b_index = np.zeros(MAX_DIMS, np.int32)---------------------| #6
            to_index(i, out_shape, out_index)                          |
            broadcast_index(out_index, out_shape, a_shape, a_index)    |
            broadcast_index(out_index, out_shape, b_shape, b_index)    |
            o = index_to_position(out_index, out_strides)              |
            j = index_to_position(a_index, a_strides)                  |
            k = index_to_position(b_index, b_strides)                  |
            out[o] = fn(a_storage[j], b_storage[k])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(254) is hoisted out of the parallel loop labelled #8 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(255) is hoisted out of the parallel loop labelled #8 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(256) is hoisted out of the parallel loop labelled #8 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(289)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py (289)
-------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                           |
        out: Storage,                                                                      |
        out_shape: Shape,                                                                  |
        out_strides: Strides,                                                              |
        a_storage: Storage,                                                                |
        a_shape: Shape,                                                                    |
        a_strides: Strides,                                                                |
        reduce_dim: int,                                                                   |
    ) -> None:                                                                             |
        size = 1                                                                           |
        for i in range(len(out_shape)):                                                    |
            size *= out_shape[i]                                                           |
        reduce_size = a_shape[reduce_dim]                                                  |
        reduce_stride = a_strides[reduce_dim]                                              |
                                                                                           |
        for i in prange(size):-------------------------------------------------------------| #10
            out_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------| #9
            to_index(i, out_shape, out_index)                                              |
            o = index_to_position(out_index, out_strides)                                  |
            start_pos = index_to_position(out_index, a_strides)                            |
            acc = out[o]                                                                   |
            for s in range(reduce_size):                                                   |
                acc = fn(acc, a_storage[start_pos + s * reduce_stride])  # type: ignore    |
            out[o] = acc                                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(305) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py
(317)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/tolyaho/Desktop/hse/dl-2/hw1/minitorch-module-3/minitorch/fast_ops.py (317)
-----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                             |
    out: Storage,                                                                        |
    out_shape: Shape,                                                                    |
    out_strides: Strides,                                                                |
    a_storage: Storage,                                                                  |
    a_shape: Shape,                                                                      |
    a_strides: Strides,                                                                  |
    b_storage: Storage,                                                                  |
    b_shape: Shape,                                                                      |
    b_strides: Strides,                                                                  |
) -> None:                                                                               |
    """NUMBA tensor matrix multiply function.                                            |
                                                                                         |
    Should work for any tensor shapes that broadcast as long as                          |
                                                                                         |
    ```                                                                                  |
    assert a_shape[-1] == b_shape[-2]                                                    |
    ```                                                                                  |
                                                                                         |
    Optimizations:                                                                       |
                                                                                         |
    * Outer loop in parallel                                                             |
    * No index buffers or function calls                                                 |
    * Inner loop should have no global writes, 1 multiply.                               |
                                                                                         |
                                                                                         |
    Args:                                                                                |
    ----                                                                                 |
        out (Storage): storage for `out` tensor                                          |
        out_shape (Shape): shape for `out` tensor                                        |
        out_strides (Strides): strides for `out` tensor                                  |
        a_storage (Storage): storage for `a` tensor                                      |
        a_shape (Shape): shape for `a` tensor                                            |
        a_strides (Strides): strides for `a` tensor                                      |
        b_storage (Storage): storage for `b` tensor                                      |
        b_shape (Shape): shape for `b` tensor                                            |
        b_strides (Strides): strides for `b` tensor                                      |
                                                                                         |
    Returns:                                                                             |
    -------                                                                              |
        None : Fills in `out`                                                            |
                                                                                         |
    """                                                                                  |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               |
                                                                                         |
    batch = out_shape[0]                                                                 |
    rows = out_shape[1]                                                                  |
    cols = out_shape[2]                                                                  |
    inner = a_shape[2]                                                                   |
    for p in prange(batch * rows * cols):------------------------------------------------| #11
        n = p // (rows * cols)                                                           |
        rest = p % (rows * cols)                                                         |
        i = rest // cols                                                                 |
        j = rest % cols                                                                  |
        acc = 0.0                                                                        |
        for k in range(inner):                                                           |
            acc += (                                                                     |
                a_storage[n * a_batch_stride + i * a_strides[1] + k * a_strides[2]]      |
                * b_storage[n * b_batch_stride + k * b_strides[1] + j * b_strides[2]]    |
            )                                                                            |
        out[n * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
</pre>
</details>

## Task 3.4

### CPU reference

Naive Python matrix multiplication compared with FastOps. Warmup is excluded.

| n | naive (s) | FastOps (s) | speedup |
| --- | --- | --- | --- |
| 32 | 0.00070 | 0.00057 | 1.2x |
| 64 | 0.00566 | 0.00074 | 7.7x |
| 96 | 0.01917 | 0.00102 | 18.7x |
| 128 | 0.04632 | 0.00152 | 30.4x |

![MM speed](images/mm_speed.png)

CUDA validation and benchmarking require a CUDA-capable GPU and are pending.

## Task 3.5

Backend: FastOps (CPU). Seed: 0. Epochs: 500. Logs are reported every 10 epochs, with epoch 490 as the final checkpoint. GPU results are pending.

### Simple

- PTS=50, HIDDEN=10, RATE=0.05, BACKEND=cpu
- Epoch 100 loss 0.72 correct 47 time_per_epoch 0.1689s
- Epoch 490 loss 0.31 correct 50 time_per_epoch 0.1140s

![Simple](images/simple.png)

### Xor

- PTS=50, HIDDEN=10, RATE=0.05, BACKEND=cpu
- Epoch 100 loss 6.71 correct 33 time_per_epoch 0.0808s
- Epoch 490 loss 1.76 correct 46 time_per_epoch 0.0800s

![Xor](images/xor.png)

### Split

- PTS=50, HIDDEN=10, RATE=0.05, BACKEND=cpu
- Epoch 100 loss 7.18 correct 28 time_per_epoch 0.0795s
- Epoch 490 loss 0.58 correct 49 time_per_epoch 0.0803s

![Split](images/split.png)

### Split, larger model

Official command: `--BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`

- PTS=50, HIDDEN=100, RATE=0.05, BACKEND=cpu
- Epoch 100 loss 0.71 correct 48 time_per_epoch 0.0923s
- Epoch 490 loss 0.063 correct 50 time_per_epoch 0.0926s

![Split large](images/split-big.png)