import struct

def read_bytes():
    f = open('send.dat', 'rb')
    bytes = []
    bytes.append(struct.unpack('i', f.read(4)))
    #for line in f:
    #    bytes.append(line)
    return bytes

if __name__ == '__main__':
    print(read_bytes())
