import zlib


def compress(b: bytes):
    b_compress = zlib.compress(b)
    print("Compress: ", len(b), "->", len(b_compress))
    return b_compress


def decompress(b: bytes):
    return zlib.decompress(b, bufsize=2**20)
