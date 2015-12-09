import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 0.25e6

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

def plot_rx_t(data):
    """
    Plots received data file in the time domain

    INPUT
    data:   tuple with first value as numpy array of real part of data
            and second value as numpy array of imaginary part of data
    """
    real = data[0]
    imag = data[1]
    time = np.arange(len(real)) * (1/SAMPLE_RATE)
    plt.plot(time, real)
    plt.show()

def fft_rx(data):
    """
    Returns the FFT of data with frequency bins in output

    INPUT
    data:   tuple with first value as numpy array of real part of data
            and second value as numpy array of imaginary part of data

    OUTPUT
    fft_data:   tuple with first value as numpy array of frequency bins
                and second value as numpy array of complex64 type
    """
    real = data[0]
    imag = data[1]
    z = data[0] + data[1]*1j

    z_fft = np.fft.fft(z)
    freq = SAMPLE_RATE*np.arange(len(z_fft)) / len(z_fft)
    fft_data = (freq, z)
    return fft_data

def plot_rx_f(data):
    """
    Plots received data in the frequency domain

    INPUT
    data:   tuple with first value as numpy array of real part of data
            and second value as numpy array of imaginary part of data
    """
    freq, z_fft = fft_rx(data)
    plt.plot(freq, z_fft)
    plt.show()

def plot_rx(data):
    real = data[0]
    imag = data[1]
    Y = (data[0]+data[1]*1j)
    fft_Y = np.fft.fft(((Y)**2))
    x = np.arange(len(fft_Y))
    freq_bins = 0.25e6*x/(len(x))
    # plt.plot(freq_bins, abs(fft_Y))
    """ FINDS MAXIMUM OF FFT TO GET 2*f_delta """
    location = np.where((abs(fft_Y)[0:len(freq_bins)/2])==(abs(fft_Y)[0:len(freq_bins)/2]).max())
    print freq_bins[location]

    """ USING MAX: PLOT THE DEMODULATED GRAPH """
    two_f_delta = freq_bins[location]
    demod = Y*np.exp(-1j*np.pi*two_f_delta*np.arange(len(Y)))
    plt.plot(abs(demod))


    plt.show()

if __name__ == '__main__':
    rx = read_floats('received.dat')
    #plot_rx(rx)
    #plot_rx_t(rx)
    plot_rx_f(rx)
