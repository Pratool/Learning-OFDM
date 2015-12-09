import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 0.25e6

def read_floats(filename):
    """
    Reads binary file as Float32 data until end of file

    INPUT
    filename:   string path of the output file to be written

    OUTPUT
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real)
    """
    # reads binary file as float32 until end of file
    data = np.fromfile(filename, dtype=np.float32, count=-1)

    # extracts real and imaginary parts
    real = data[0::2]
    imag = data[1::2]
    data_z = real + imag*1j
    return data_z

def plot_rx_t(data_z):
    """
    Plots received data file in the time domain

    INPUT
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real)
    """
    time = np.arange(len(data_z)) * (1/SAMPLE_RATE)
    plt.plot(time, data_z.real)
    plt.show()

def fft_rx(data_z):
    """
    Returns the FFT of data with frequency bins in output

    INPUT
    data_z: numpy array of complex64 type (float32 real, float32 imag)

    OUTPUT
    fft_data:   tuple with first value as numpy array of frequency bins
                and second value as numpy array of complex64 type
    """
    z = data_z
    z_fft = np.fft.fft(z)
    x = np.arange(len(z))
    freq = SAMPLE_RATE*np.arange(len(z)) / len(z)
    fft_data = (freq, z_fft)
    return fft_data

def plot_rx_f(data_z):
    """
    Plots received data in the frequency domain

    INPUT
    data:   tuple with first value as numpy array of real part of data
            and second value as numpy array of imaginary part of data
    """
    freq, z_fft = fft_rx(data_z)
    plt.plot(freq, z_fft)
    plt.show()

def sync_freq_defs(data_z):
    z = data_z
    freqs, z_sq_fft = fft_rx(z**2)

    # Finds maximum of FFT to get 2*f_delta
    two_f_delta = freqs[np.where((abs(z_sq_fft)[0:len(freqs)/2])==(abs(z_sq_fft)[0:len(freqs)/2]).max())]

    # demodulate original signal by complex number phase shift
    demod = z*np.exp(-1j*np.pi*two_f_delta*np.arange(len(z)))

    # plot frequency-synchronized signal
    plt.plot(abs(demod))
    plt.show()

if __name__ == '__main__':
    rx = read_floats('received.dat')
    sync_freq_defs(rx)
