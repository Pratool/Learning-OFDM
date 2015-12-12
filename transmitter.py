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

if __name__ == '__main__':
    time = 2
    num_bits = 8
    zero_pad = 1e5
    num_samp = int(time*SAMPLE_RATE)
    tot_samp = num_samp + zero_pad
    # numpy array containing [0, 1] repeated num_bits/2 times
    x_real = 0.7*np.array( [np.floor(i/(num_samp/num_bits))%2 for i in range(num_samp)] )
    # pad with zeros
    x_real = np.hstack((np.zeros(zero_pad/2), x_real, np.zeros(zero_pad/2)))
    plt.plot(x_real)
    plt.show()
    x_imag = np.zeros(tot_samp)
    test_data = (x_real, x_imag)
    write_floats('sent.dat', test_data)
