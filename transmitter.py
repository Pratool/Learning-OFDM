import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 0.25e6

def write_floats(filename, tx_data):
    """
    Writes a binary file with type Float32 with input data
    
    INPUT
    filename:   string path of the output file to be written
    tx_data:    data to be written with form tuple of 2 arrays (real, imag)
    """
    real_data = np.array(tx_data[0])
    imag_data = np.array(tx_data[1])

    data = np.zeros(2*len(real_data), dtype=np.float32)
    data[0::2] = real_data
    data[1::2] = imag_data

    data.tofile(filename)

    return data

def arr_gen(input_arr):
    bit_len = 100
    out_arr = []
    for i in range(len(input_arr)):
        out_arr.extend([input_arr[i]]*bit_len)
    out_arr = 0.3*np.array(out_arr)
    return out_arr

if __name__ == '__main__':
    time = 2
    zero_pad = 1e5
    in_arr = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    in_arr = in_arr*2 - 1
    x_real = arr_gen(in_arr)
    # pad with zeros
    x_real = np.hstack((np.zeros(zero_pad/2), x_real, np.zeros(zero_pad/2)))
    plt.plot(x_real)
    plt.show()
    x_imag = np.zeros(len(x_real))
    test_data = (x_real, x_imag)
    write_floats('sent.dat', test_data)
