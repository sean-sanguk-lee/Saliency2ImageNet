from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer
import pyimgsaliency as psal
import os
import pickle

demo_img = 'demo.jpeg'

# @cuda.jit
@jit
def func(a):
    for i in range(10000000):
        a[i] += 1

# @jit(target="cuda")
@jit(nopython=True)
def func2(a):
    for i in range(10000000):
        a[i] += 1

def encode_mbd_to_pickle(output_dir, mbd):
    # output_directory.jpeg -> output_directory.pickle
    filename = output_dir[:-5] + '.pickle'
    if os.path.isfile(filename):
        print('File already exists')
    else:
        with open(filename, 'wb') as f_out:
            pickle.dump(mbd, f_out)
        f_out.close()
        print('Successfully serialized saliency map (mbd) to:', filename)

if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float32)

    start = timer()
    func(a)
    print("With GPU (@jit):", timer() - start)

    start = timer()
    func2(a)
    print("With GPU (@jit(nopython=True)):", timer() - start)

    mbd = psal.get_saliency_mbd(demo_img)
    encode_mbd_to_pickle(os.getcwd() + '\\' + demo_img, mbd)
