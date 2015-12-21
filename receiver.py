import numpy as np
import matplotlib.pyplot as plt

# for implementing image capture
#import matplotlib.image as mpimg

SAMPLE_RATE = 0.25e6
FFT_SIZE    = 4.4e-2*SAMPLE_RATE
BIT_LENGTH  = 50

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

def fft_rx(data_z, samp_rate):
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

    # concatenate postive frequency and negative frequency magnitudes
    z_fft = np.hstack((z_fft[len(z_fft)/2:], z_fft[:int(len(z_fft)/2)]))

    # generate frequency bins
    freq = np.array(samp_rate*np.arange(len(z)) / len(z))
    # add an element so that the the negative frequencies has the correct
    # value
    freq = freq[:((len(freq)/2) + (len(z_fft)%2))]

    # horizontally flip and negate frequency bins half of frequency bins
    # and then concatentate with frequency bins
    freq = np.hstack((-1*freq[::-1], freq))

    # remove last entry of freq in case 2 elements were added if z_fft had
    # an odd length
    freq = freq[:len(z_fft)]
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

    INPUT
    time    times the signal was received (as numpy array)
    rx_sig  received signal value (as numpy array of data type complex64)

    OUTPUT
    t       times the signal was actually caught (as numpy array)
    rx      received signal values at transmission interval (as numpy array of
            data type complex64)
    """
    # ignoring impulse defect at beginning of time series
    t = time[100:]
    rx = rx_sig[100:]

    # normalizing signal
    rx = rx / rx.max()

    # 5 standard devations away from average value of noise
    #avg_noise = np.average(abs(rx[:1000])) + 5*np.std(abs(rx[:1000]))

    # root mean square of noise
    #avg_noise = np.sqrt( (( abs(rx[:1000]) ) ** 2).mean())

    # general threshold
    avg_noise = 0.55

    # truncate everything outside of signal
    sig_beg = 0
    sig_end = len(rx)-1
    for i in range(len(rx)):
        if abs(rx[i]) > avg_noise:
            sig_beg = i
            break

    for i in range(len(rx)-1, 0, -1):
        if abs(rx[i]) > avg_noise:
            sig_end = i
            break

    rx = rx[sig_beg:sig_end]
    t = t[sig_beg:sig_end]

    return (t, rx)

def sync_freq_defs(data_z, time):
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
    freqs, z_sq_fft = fft_rx(z**2, SAMPLE_RATE)
    abs_z_sq_fft = abs(z_sq_fft)

    # Finds frequency corresponding to maximum of FFT(input^2) to get 2*f_delta
    loc = np.where(abs_z_sq_fft==max(abs_z_sq_fft))[0][0]
    two_f_delta = freqs[loc]
    phi = np.angle(z_sq_fft[loc])

    # demodulate original signal by multiplying complex number for phase shift
    freq_bin = np.mean(np.diff(freqs))

    bestValue = 0
    for x in np.linspace(-freq_bin, freq_bin, 100):
        synced = z*np.exp(-1j*2*np.pi*(two_f_delta/2 + x)*time)
        value = np.var(synced.real)/np.var(synced.imag)
        if value > bestValue:
            bestValue = value
            best_synced = synced

    return best_synced

def sync_long_sig(data_z, time):
    """
    Synchronizes the received data in short segments and pieces them together

    INPUT
    data_z:     numpy array of data type complex64 (float32 imaginary and
                float32 real) with unsynchronized data values
    time:       numpy array with the times at which the data was received

    OUTPUT
    synced_segments     stitched together numpy arrays of complex64 with
                        frequency offsets removed
    """
    synced_segments = []
    i = 0
    while i+FFT_SIZE< len(time):
        synced_segment = sync_freq_defs(data_z[i:i+FFT_SIZE], time[i:i+FFT_SIZE])
        synced_segments.extend(synced_segment)
        i += FFT_SIZE
    if len(data_z) % FFT_SIZE> 0:
        synced_segments.extend(sync_freq_defs(data_z[i:], time[i:]))

    fr, fft = fft_rx(data_z[0:FFT_SIZE], SAMPLE_RATE)
    #plt.plot(fr, abs(fft))
    #plt.plot(fr, np.angle(fft))
    #plt.show()

    return np.array(synced_segments)

"""
def read_bytes(rx):
    orig_array = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]*1000
    bytes_array = []
    wrong = []
    count = 0
    for i in range(((len(rx)/BIT_LENGTH)+1)/16):
         count += 1
         print count
         if (rx[BIT_LENGTH/2 + BIT_LENGTH*16*i] >0 and rx[BIT_LENGTH/2 + BIT_LENGTH*16*i + BIT_LENGTH] > 0):
             print "here"
             for k in range(16*i, 16*i+16):
                 if (rx[BIT_LENGTH/2 + BIT_LENGTH*k] < 0):
                     bytes_array.append(0)
                 else:
                     bytes_array.append(1)
         elif (rx[BIT_LENGTH/2 + BIT_LENGTH*16*i] < 0 and rx[BIT_LENGTH/2 + BIT_LENGTH*16*i + BIT_LENGTH] < 0):
             for k in range(16*i, 16*i+16):
                 if (rx[BIT_LENGTH/2 + BIT_LENGTH*k] < 0):
                     bytes_array.append(1)
                 else:
                     bytes_array.append(0)
         else:
             for k in range(16*i, 16*i+16):
                 if (rx[BIT_LENGTH/2+BIT_LENGTH*k] < 0):
                     bytes_array.append(0)
                 else:
                     bytes_array.append(1)

    print bytes_array
    print len(bytes_array)
    print len(orig_array)

    for i in range((len(rx)/BIT_LENGTH)+1):
        if (rx[BIT_LENGTH/2+BIT_LENGTH*i] > 0):
            if (25+50*i )
            bytes_array.append(1)
        else:
            bytes_array.append(0)
    
    for j in range(len(bytes_array)):
        if (orig_array[j] != bytes_array[j]):
            wrong.append(j)
     print wrong
     print bytes_array
"""

def sample_correct_bits(rx):
    #rxdiff = np.ediff1d(rx)
    rxd = []
    nrxd = []
    for i in range(len(rx)-1):
        if abs(rx[i] - rx[i+1]) > 1*abs(rx[i]) and abs(rx[i+15 if i < len(rx)-16 else i]) > 0.1:
            rxd.append(i)
            plt.plot(i, rx[i], 'ro')
    plt.plot(rx)
    plt.show()
    print 'done with first for loop. len(rxd):', len(rxd)
    """
    j = 0
    while j < len(rxd)-1:
        nrxd.append(rxd[j])
        if abs(rxd[j]-rxd[j+1]) < 0.1:
            while abs(rxd[j]-rxd[j+1]) < 0.1:
                j += 1
        else:
            j += 1
    print 'done with nested while loop'

    byte_arr = []
    i = BIT_LENGTH / 2
    j = 0
    while i < len(rx):
        byte_arr.append(int(rx[i]>0))
        if i > nrxd[j] and j+1 < len(nrxd):
            j += 1
            i = nrxd[j]+ (BIT_LENGTH/2)
        else:
            i += BIT_LENGTH
    print 'done with 2nd while loop'
    return np.array(byte_arr)
    """

def check_with_bits(rx):
    flipped_rx = []
    for i in range(0, len(rx)-1, 16):
        if i != 1:
            flipped_rx.extend(1-rx[i:i+15])
        else:
            flipped_rx.extend(rx[i:i+15])
    return np.array(flipped_rx)

def check_with_orig(orig, rx_bits):
    errors = 0
    for i in range(len(rx_bits)):
        errors += (orig[i] != rx_bits[i])
    return errors

if __name__ == '__main__':
    t, rx_pre = read_floats('received.dat')
    t, rx = splice_digital_sig(t, rx_pre)
    synced_rx = sync_long_sig(rx, t)
    check_with_bits(synced_rx)

    orig_array = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]*1000
    orig_array = np.array(orig_array)
    ext_array = []
    for i in range(len(orig_array)):
        ext_array.extend([orig_array[i]]*(BIT_LENGTH-1))
    ext_array = np.array(ext_array)
    ext_array = 2*ext_array-1
    ext_array = 0.9*ext_array

    rx_bytes = sample_correct_bits(synced_rx.real)

    #checked_rx_bytes = check_with_bits(rx_bytes)
    #print check_with_orig(orig_array, checked_rx_bytes)

    #print read_bytes(-synced_rx.real)
    # find_phase_flips(np.ediff1d(np.angle(synced_rx)))
    # read_bytes(rx_pre, synced_rx, t, orig_array)
    plts = 2000
    plte = 3000
    # plt.plot(t[3001:10001], np.ediff1d(np.angle(synced_rx))[3000:10000])
    start = 0
    #plt.plot(t, synced_rx.imag)
    for i in range(4, 7):
        print rx_bytes[i*16: (i+1)*16]
        print orig_array[i*16: (i+1)*16]
        print
    print check_with_orig(orig_array, rx_bytes)
    #plt.plot(t, synced_rx.real)
    checks = [1 if orig_array[i] != rx_bytes[i] else 0 for i in range(len(rx_bytes))]
    checks = np.array(checks)
    plt.plot(checks)
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    # plt.plot(t, synced_rx.real)
    # plt.plot(t, rx)

    #plt.plot(checked_rx_bytes)
    #plt.plot(orig_array)
    plt.show()
