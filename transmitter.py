import numpy as np
import matplotlib.pyplot as plt

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
    num_samp = 1e6
    #x_real = 0.7*np.sin(np.arange(num_samp))
    #x_real = np.zeros(num_samp)+0.7
    x_real = np.hstack((np.zeros(num_samp/4), np.ones(num_samp/4)))*0.7
    x_real = np.hstack((x_real, x_real))
    plt.plot(x_real)
    plt.show()
    x_imag = np.zeros(num_samp)
    test_data = (x_real, x_imag)
    write_floats('sent.dat', test_data)
