import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 0.25e6

def read_floats(filename):
    """
    Reads binary file as Float32 data until end of file

    INPUT
    filename:   string path of the output file to be written

    OUTPUT
    time:       numpy array of the time at which the signal's "voltage" was
                recorded
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real), the signal "voltage"
    """
    # reads binary file as float32 until end of file
    data = np.fromfile(filename, dtype=np.float32, count=-1)

    # extracts real and imaginary parts
    real = data[0::2]
    imag = data[1::2]
    data_z = real + imag*1j

    # recovers time info
    time = np.arange(len(data_z))/SAMPLE_RATE

    return (time, data_z)

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
    Computes FFT of received data and plots in frequency domain

    INPUT
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real) in the time domain
    """
    freq, z_fft = fft_rx(data_z)
    plt.plot(freq, z_fft)
    plt.show()

def splice_digital_sig(time, rx_sig):
    """
    Returns the part of the signal where there are actually bits (not just
    white noise from waiting)
    """
    # ignoring impulse defect at beginning of time series
    t = time[100:]
    rx = rx_sig[100:]

    # normalizing signal
    rx = rx / rx.max()

    # truncate everything outside of signal
    ediff = np.ediff1d(abs(rx))
    sig_end = np.where(ediff < -0.2)[0][-1]+5
    sig_beg = np.where(ediff < -0.2)[0][0]+5
    rx = rx[sig_beg:sig_end]
    t = t[sig_beg:sig_end]

    return (t, rx)

def sync_freq_defs(data_z):
    """
    Synchronizes the received data by compensating for offset between
    transmitter frequency and receiver frequency

    INPUT
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real) with unsynchronized time data

    OUTPUT:
    synced_z:   numpy array of data type complex64 (float32 imaginary and
                float32 real) with frequency synchronized time data
    """
    z = data_z
    freqs, z_sq_fft = fft_rx(z**2)

    # recovers time info
    time = np.arange(len(data_z))/SAMPLE_RATE

    # Finds frequency corresponding to maximum of FFT(input^2) to get 2*f_delta
    abs_z_sq_fft = (abs(z_sq_fft)[0:len(freqs)])
    two_f_delta = freqs[np.where(abs_z_sq_fft==abs_z_sq_fft.max())]

    print two_f_delta
    # demodulate original signal by multiplying complex number for phase shift
    synced_z = z*np.exp(-1j*np.pi*two_f_delta*np.arange(len(time)))

    return synced_z

if __name__ == '__main__':
    t, rx = read_floats('received.dat')
    t, rx = splice_digital_sig(t, rx)
    synced_rx = sync_freq_defs(rx)
    plt.plot(t, abs(synced_rx))
    #plt.plot(t, np.angle(rx, deg=True))
    plt.show()
