import numpy as np

def wr_dat(in_dat_list, out_file_path):
    out_file = open(out_file_path, 'wb')
    out_file.write(bytes(in_dat_list))
    out_file.close()

if __name__ == "__main__":
    data = np.sin(np.arange(10000))
    wr_dat(data, 'sent.dat')
