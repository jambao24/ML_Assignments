import numpy as np

# sigmoid function applied to an array
def _sigmoid_func_arr(x):
    return 1/(1 + np.exp(-x))


arr = np.array([-1,-1,-1,-10,1,2,3,4,5])

arr_new = _sigmoid_func_arr(arr)
print(arr_new)
print()