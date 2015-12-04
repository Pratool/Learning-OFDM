import numpy as np
import matplotlib.pyplot as plt

def read_floats(filename):
    """
    Reads binary file as Float32 data until end of file

    INPUT
    filename:   string path of the output file to be written
    """
    # reads binary file as float32 until end of file
    data = np.fromfile(filename, dtype=np.float32, count=-1)

    # extracts real and imaginary parts
    real = data[0::2]
    imag = data[1::2]

    return (real, imag)

def plot_rx(data):
    real = data[0]
    imag = data[1]
    plt.plot(data[0])
    plt.show()

if __name__ == '__main__':
    rx = read_floats('tx_samp.dat')
    plot_rx(rx)
