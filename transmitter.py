import numpy as np

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
    test_data = (np.arange(6), np.zeros(6))
    write_floats('tx_samp.dat', test_data)
