#numba cuda tryout:
import time
import numpy as np
import cupy as cp
from numba import cuda, types, float32, njit
import math

@njit
def con(x):
    return np.ascontiguousarray(x)

@njit
def matmuls(X, Y):
    return np.dot(con(X.astype(np.float64)), con(Y.astype(np.float64)))


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        if x < A.shape[0] and (ty+i*TPB) < A.shape[1]:
          sA[tx, ty] = A[x, ty + i * TPB]
        if y < B.shape[1] and (tx+i*TPB) < B.shape[0]:
          sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp
@njit
def cud(x, y):
    return cp.dot(x, y)

#%%
shape = [4096 * 2, 4096 * 2]
x_h = np.arange(np.prod(shape)).reshape(shape)
y_h = np.ones(shape)
z_h = np.zeros(shape)

x_d = cuda.to_device(x_h)
y_d = cuda.to_device(y_h)
z_d = cuda.to_device(z_h)

TPB = 3
threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

for i in range(10):
    start_time = time.time()
    fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
    z_h = z_d.copy_to_host()
    #print(z_h)
    print ("Our tensorflow calculation took", time.time() - start_time, "to run")
    start_time = time.time()
    x = matmuls(x_h, y_h)
    print ("Our numpy calculation took", time.time() - start_time, "to run")
    start_time = time.time()
    x = cud(x_h, y_h)
    print ("Our cupy calculation took", time.time() - start_time, "to run")
