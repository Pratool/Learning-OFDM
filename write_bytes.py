def wr_dat(in_str, out_file_path):
    out_file = open(out_file_path, 'wb')
    out_file.write(bytes(in_str)).encode('ascii')

if __name__ == "__main__":
    wr_dat('hi, my name is pratool', 'send.dat')
