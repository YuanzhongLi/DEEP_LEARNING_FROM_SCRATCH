# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# 3-2
def step_function(x):
    if (x > 0):
        return 1
    else:
        return 0


import numpy as np
# x: 
def step_function(x):
    y = x > 0
    return y.astype(np.int)
