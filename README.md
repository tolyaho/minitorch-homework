# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py

## Task 1.5

Trained `project/run_scalar.py` with `random.seed(0)` before each run. Optimizer is the starter SGD; `max_epochs=500`.

### Simple

- PTS=50, HIDDEN=2, RATE=0.5
- Epoch 100 loss 11.39 correct 49
- Epoch 500 loss 2.46 correct 49

![Simple](images/simple.png)

### Diag

- PTS=50, HIDDEN=2, RATE=0.5
- Epoch 100 loss 11.31 correct 47
- Epoch 500 loss 1.08 correct 50

![Diag](images/diag.png)

### Split

- PTS=50, HIDDEN=10, RATE=0.5
- Epoch 100 loss 14.86 correct 43
- Epoch 500 loss 2.50 correct 49

![Split](images/split.png)

### Xor

- PTS=50, HIDDEN=10, RATE=0.5
- Epoch 100 loss 12.49 correct 46
- Epoch 500 loss 4.66 correct 48

![Xor](images/xor.png)
