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

    # demodulate original signal by multiplying complex number for phase shift
    synced_z = z*np.exp(-1j*2*np.pi*(two_f_delta/2)*time)

    return synced_z

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

# def read_bytes(pre, signal, time, orig_array):
#     """
#     Find when value switches if it has not entered the zero threshold. Restart count at switched value
#     """
#     real_signal_diff = np.ediff1d(signal.real)
#     zeros_real = []
#     imag_signal_diff = np.ediff1d(signal.imag)
#     zeros_imag = []

#     """
#     Where the zeros are in the real 
#     normalizing real part of the signal
#     """
#     rx_real = pre.real[100:]
#     x_real = rx_real / rx_real.max()
#     max_t_real = max(x_real[0:300])
#     min_t_real = min(x_real[0:300])
#     r_thres = (min_t_real, max_t_real)

#     temp_real = 0
#     for i in range(len(real_signal_diff)):
#         if (real_signal_diff[i] > max_t_real-min_t_real):
#             if (i - temp_real > 4):
#                 zeros_real.append(i)
#                 temp_real = i


#     """
#     Where the zeros are in the imag 
#     normalizing imag part of the signal
#     """
#     rx_imag = pre.imag[100:]
#     x_imag = rx_imag / rx_imag.max()
#     max_t_imag = max(x_imag[0:300])
#     min_t_imag = min(x_imag[0:300])
#     i_thres = (min_t_imag, max_t_imag)

#     temp_imag = 0
#     for i in range(len(imag_signal_diff)):
#         if (imag_signal_diff[i] > max_t_imag-min_t_imag):
#             if (i - temp_imag > 4):
#                 zeros_imag.append(i)
#                 temp_imag = i

#     # print zeros_real
#     # print zeros_imag

#     """
#     Everytime you have zeroth element, Find the 25th element after it, and recount.
#     """
#     # signal = -(signal)
#     list_bytes = []
#     n = 25
#     num_error = 0

#     # 0 = real, 1 = imaginary
#     cur_ri = 1
#     cur_arrs = [signal.real, signal.imag]
#     #cur_thres = (r_thres, i_thres)
#     cur_thres = ( (-.6, .6), (-.6,.6) )
#     cur_zeros = (zeros_real, zeros_imag)
#     for j in range(len(zeros_imag)):
#         # if cur_arrs[cur_ri][n] > cur_thres[cur_ri][0] and cur_arrs[cur_ri][n] < cur_thres[cur_ri][1]:
#         #    cur_ri = (cur_ri+1)%2
#         #    prev_bit = cur_zeros[(cur_ri+1)%2][j-1]+25
#         #    print 'switching', n
#         #    if list_bytes[-1] == -(cur_arrs[cur_ri][n] > 0):
#         #        cur_arrs[cur_ri] = -cur_arrs[cur_ri]
#         bit = cur_arrs[cur_ri][n] > 0
#         list_bytes.append(bit)
#         num_error += bit == orig_array[j]
#         if ((n + 50) < cur_zeros[cur_ri][j]):
#             n = 50 + n
#         else:
#             n = cur_zeros[cur_ri][j] + 25

#     print list_bytes
#     list_bytes = np.array(list_bytes)
#     orig_array = np.array(orig_array)
#     print num_error
#     comp_arr = [ 1 if list_bytes[m] != orig_array[m] else 0 for m in range(len(list_bytes)) ]
#     #comp_arr = np.array(list_bytes != orig_array)
#     comp_arr = np.array(comp_arr)
#     plt.plot(comp_arr)
#     plt.ylim([-0.1, 1.1])
#     plt.show()

def read_bytes(rx):
    orig_array = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]*500
    bytes_array = []
    wrong = []
    count = 0
    for i in range(((len(rx)/50)+1)/16):
        count += 1
        # print count
        if (rx[25 + 50*16*i] >0 and rx[25 + 50*16*i + 50] > 0):
            # print "here"
            for k in range(16*i, 16*i+16):
                if (rx[25 + 50*k] < 0):
                    bytes_array.append(0)
                else:
                    bytes_array.append(1)
        elif (rx[25 + 50*16*i] < 0 and rx[25 + 50*16*i + 50] < 0):
            for k in range(16*i, 16*i+16):
                if (rx[25 + 50*k] < 0):
                    bytes_array.append(1)
                else:
                    bytes_array.append(0)
        else:
            for k in range(16*i, 16*i+16):
                if (rx[25+50*k] < 0):
                    bytes_array.append(0)
                else:
                    bytes_array.append(1)

    print bytes_array

    # for i in range((len(rx)/50)+1):
    #     if (rx[25+50*i] > 0):
    #         # if (25+50*i )
    #         bytes_array.append(1)
    #     else:
    #         bytes_array.append(0)

    for j in range(len(bytes_array)):
        if (orig_array[j] != bytes_array[j]):
            wrong.append(j)
    print len(wrong)
    # print bytes_array

# def find_phase_flips(rx) :
#     """
#     Finds the location of all of the phase flips

#     Input: Phase Angles

#     Output: Indeces of where in the original array, the phase flips
#     """
#     list_max = []
#     final_list_max = []
#     temp = []
#     final_list_max.append(0)

#     #Find the index of where the angle differences are the biggest (where the phase flips) 
#     phase_diff = np.ediff1d(rx)
#     for i in range(len(rx)):
#         if (rx[i] > 5):
#             list_max.append(i)

#     #Going through all of the indeces, we find the index that it matches to in an array that is every 50th bit, so 723 would
#     #would be closest to 700 (the closest number that is divisible by 700 and smaller than 723), and then divide that by 50 to
#     #find what bit that would match to in our original array of bits (that we had found the median bit of each padded signal)
#     # for i in range(len(list_max)):
#     #     temp.append(((list_max[i])))
#     #     final_list_max.append(((list_max[i] - (list_max[i]%50)))/50)

#     #Remove duplicates and sort the list:
#     # final_list_max = sorted(list(set(final_list_max)))
#     # print temp
#     # print len(final_list_max)
#     print len(list_max)
#     print  list_max


if __name__ == '__main__':
    t, rx_pre = read_floats('received.dat')
    t, rx = splice_digital_sig(t, rx_pre)
    synced_rx = sync_long_sig(rx, t)

    orig_array = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]*500
    ext_array = []
    for i in range(len(orig_array)):
        ext_array.extend([orig_array[i]]*(BIT_LENGTH-1))
    ext_array = np.array(ext_array)
    ext_array = 2*ext_array-1
    ext_array = 0.9*ext_array

    # print read_bytes(synced_rx.imag)
    # find_phase_flips(np.ediff1d(np.angle(synced_rx)))
    # read_bytes(rx_pre, synced_rx, t, orig_array)
    plts = 2000
    plte = 3000
    # plt.plot(t[3001:10001], np.ediff1d(np.angle(synced_rx))[3000:10000])
    plt.plot(t[1000:], synced_rx.imag[1000:])
    plt.plot(t[1000:], synced_rx.real[1000:])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.plot(t, synced_rx.real)
    # plt.plot(t, rx)
    plt.show()
