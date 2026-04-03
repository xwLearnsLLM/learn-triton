import random
from time import perf_counter
from contextlib import contextmanager
import torch
import triton
import triton.language as tl



@contextmanager
def show_performance (name, data_volumn=None, enable=True, cuda_sync=False) :
    if enable :
        start = perf_counter()
    yield
    if cuda_sync :
        torch.cuda.synchronize()
    if enable :
        elapsed = perf_counter() - start 
        if data_volumn is None :
            print(f"[{name}] {elapsed*1e3:.2f} ms")
        else :
            bandwidth_MBPS = data_volumn / 1e6 / elapsed
            print(f"[{name}] {elapsed*1e3:.2f} ms to transfers {data_volumn/1e6:.2f} MB, BW = {bandwidth_MBPS:.2f} MBps") 




@triton.jit
def kernel_TunstallDecode (
    ip_tunstall_lut,          # shape = (256,), int64
    ip_compressed_marks,      # shape = (bsz, N), int8
    op_decompressed,          # shape = (bsz, M), int8
    N_TILE: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr
) :
    batch_id = tl.program_id(0)            # 每个 block 处理一个 batch
    ip_compressed_marks += batch_id * N
    op_decompressed     += batch_id * M
    ip_tunstall_lut     += 128

    offs_n = tl.arange(0, N_TILE)                       # shape = (N_TILE, )
    for i_block in range(N // N_TILE) :
        offs_ip = i_block * N_TILE + offs_n
        marks = tl.load(ip_compressed_marks+offs_ip)    # shape = (N_TILE, ), int8
        items = tl.load(ip_tunstall_lut+marks)          # shape = (N_TILE, ), int64
        count = items.to(tl.int32) & 0x7                # shape = (N_TILE, ), int32
        cum_count = tl.cumsum(count)                    # shape = (N_TILE, ), int32
        ops = op_decompressed + (cum_count - count)     # shape = (N_TILE, ), pointer 
        #op_decompressed += cum_count.get_element((N_TILE-1,))  # move to next pointer
        op_decompressed += tl.sum(count)
        for _ in range(7) :
            items >>= 8
            bytes = items.to(tl.int8)
            tl.store(ops, bytes, mask=(bytes>=0))
            ops += 1



def triton_tunstall_decode (
    tunstall_lut,
    compressed_marks,
    decompressed,
    N_TILE = 16
) : 
    assert tunstall_lut.dtype     == torch.int64
    assert compressed_marks.dtype == torch.int8 
    assert decompressed.dtype     == torch.int8 
    BSZ, N = compressed_marks.shape
    _  , M = decompressed.shape
    assert (BSZ, M) == decompressed.shape 
    assert tunstall_lut.shape == (256,)
    assert N % N_TILE == 0
    kernel_TunstallDecode [(BSZ,)] (tunstall_lut, compressed_marks, decompressed, N_TILE, N, M)



def reference_tunstall_decode (tunstall_lut, compressed_marks, decompressed) :    # AI 写的 golden reference (经典又臭又长)
    BSZ, N = compressed_marks.shape
    _, M = decompressed.shape
    device = compressed_marks.device

    # Move LUT to CPU for indexing (assume it's on the same device as inputs)
    lut_cpu = tunstall_lut.cpu().numpy()  # shape (256,)

    # Convert compressed_marks to CPU numpy for iteration
    marks_cpu = compressed_marks.cpu().numpy()  # shape (BSZ, N)

    # Prepare output as numpy, then copy back at the end
    out_cpu = decompressed.cpu().numpy()  # shape (BSZ, M)
    out_cpu.fill(0)  # ensure zeros

    for b in range(BSZ):
        out_ptr = 0  # current write position in this batch
        for m in range(N):
            mark = marks_cpu[b, m]
            # Actual LUT index: 128 + mark (mark is int8, may be negative)
            idx = (128 + int(mark)) & 0xFF  # wrap to 0..255
            item = lut_cpu[idx]             # Python int (from np.int64)
            count = item & 0x7              # low 3 bits = number of valid bytes
            # Extract bytes: each byte in little-endian order, skip if 0xFF
            for k in range(count):
                byte_val = (item >> (8 * (k + 1))) & 0xFF  # k=0 -> bits 8..15? Wait careful.
                pass

        for m in range(N):
            mark = marks_cpu[b, m]
            idx = (128 + int(mark)) & 0xFF
            item = int(lut_cpu[idx])
            count = item & 0x7
            for shift in range(count):
                byte = (item >> (shift+1)*8) & 0xFF
                assert byte < 128
                if out_ptr >= M:
                    raise IndexError(f"Output buffer too small: out_ptr={out_ptr} >= M={M}")
                out_cpu[b, out_ptr] = byte if byte < 128 else byte - 256
                out_ptr += 1
        
        print(f'batch{b} decode done, get {out_ptr} bytes')

    # Copy back to original tensor (in-place)
    decompressed.copy_(torch.from_numpy(out_cpu).to(device=device, dtype=torch.int8))




def random_tunstall_lut_item () :
    x = random.randint(0,99)
    if   x < 10 : return 0x0706050403020107
    elif x < 20 : return 0xFF06050403020106
    elif x < 30 : return 0xFFFF050403020105
    elif x < 60 : return 0xFFFFFF0403020104
    elif x < 80 : return 0xFFFFFFFF03020103
    elif x < 90 : return 0xFFFFFFFFFF020102
    else        : return 0xFFFFFFFFFFFF0101



def random_tunstall_lut (lut_count, device) :
    tunstall_lut = [random_tunstall_lut_item() for _ in range(lut_count)]
    tunstall_lut = torch.tensor(tunstall_lut, dtype=torch.uint64, device=device).view(torch.int64)
    return tunstall_lut



def random_tunstall_marks (mark_shape, device) :
    return (torch.randn(mark_shape)*99999).to(dtype=torch.int8, device=device)



def simple_test_tunstall_decode (BSZ, N, N_TILE, device, check=False) :
    tunstall_lut = random_tunstall_lut(256, device)
    compressed_marks = random_tunstall_marks((BSZ, N), device)
    decompressed = torch.zeros((BSZ, N*8), dtype=torch.int8, device=device)
    decompressed_ref = decompressed.clone()
    with show_performance(f'decompress shape = {tuple(compressed_marks.shape)}', data_volumn=compressed_marks.numel(), cuda_sync=True) :
        triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, N_TILE)
    if check :
        reference_tunstall_decode(tunstall_lut, compressed_marks, decompressed_ref)
        if (decompressed == decompressed_ref).all() :
            print('=== test passed ===')
        else :
            print('*** test failed ***')



if __name__ == '__main__' :
    simple_test_tunstall_decode(BSZ=4   , N=16384, N_TILE=4096, device='cuda', check=True)
    simple_test_tunstall_decode(BSZ=1024, N=16384, N_TILE=4096, device='cuda')
    simple_test_tunstall_decode(BSZ=1024, N=16384, N_TILE=4096, device='cuda')


