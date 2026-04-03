import random
from time import perf_counter
from contextlib import contextmanager
import torch
import triton
import triton.language as tl


def synchronize_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("npu"):
        torch.npu.synchronize()
    else:
        raise ValueError(f"unsupported device for synchronize: {device}")


@contextmanager
def show_performance(name, data_volumn=None, enable=True, sync=False, device="npu"):
    if enable:
        start = perf_counter()
    yield
    if sync:
        synchronize_device(device)
    if enable:
        elapsed = perf_counter() - start
        if data_volumn is None:
            print(f"[{name}] {elapsed*1e3:.2f} ms")
        else:
            bandwidth_MBPS = data_volumn / 1e6 / elapsed
            print(
                f"[{name}] {elapsed*1e3:.2f} ms to transfers {data_volumn/1e6:.2f} MB, BW = {bandwidth_MBPS:.2f} MBps"
            )


@triton.jit
def kernel_TunstallDecode_CUDA(
    ip_tunstall_lut,          # shape = (256,), int64
    ip_compressed_marks,      # shape = (bsz, N), int8
    op_decompressed,          # shape = (bsz, M), int8
    N_TILE: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    batch_id = tl.program_id(0)            # each block handles one batch
    ip_compressed_marks += batch_id * N
    op_decompressed += batch_id * M
    ip_tunstall_lut += 128

    offs_n = tl.arange(0, N_TILE)          # shape = (N_TILE, )
    for i_block in range(N // N_TILE):
        offs_ip = i_block * N_TILE + offs_n
        marks = tl.load(ip_compressed_marks + offs_ip)     # shape = (N_TILE, ), int8
        items = tl.load(ip_tunstall_lut + marks)           # shape = (N_TILE, ), int64
        count = items.to(tl.int32) & 0x7                   # shape = (N_TILE, ), int32
        cum_count = tl.cumsum(count)                       # shape = (N_TILE, ), int32
        ops = op_decompressed + (cum_count - count)        # shape = (N_TILE, ), pointer
        op_decompressed += tl.sum(count)
        for _ in range(7):
            items >>= 8
            bytes_ = items.to(tl.int8)
            tl.store(ops, bytes_, mask=(bytes_ >= 0))
            ops += 1


@triton.jit
def kernel_TunstallCount_NPU(
    ip_tunstall_lut,          # shape = (256,), int64
    ip_compressed_marks,      # shape = (bsz, N), int8
    op_counts,                # shape = (bsz, N), int32
    N_TILE: tl.constexpr,
    N: tl.constexpr,
):
    batch_id = tl.program_id(0)            # each block handles one batch
    ip_compressed_marks += batch_id * N
    op_counts += batch_id * N
    ip_tunstall_lut += 128

    offs_n = tl.arange(0, N_TILE)          # shape = (N_TILE, )
    for i_block in range(N // N_TILE):
        offs_ip = i_block * N_TILE + offs_n
        marks = tl.load(ip_compressed_marks + offs_ip)     # shape = (N_TILE, ), int8
        items = tl.load(ip_tunstall_lut + marks)           # shape = (N_TILE, ), int64
        count = items.to(tl.int32) & 0x7                   # shape = (N_TILE, ), int32
        tl.store(op_counts + offs_ip, count)


@triton.jit
def kernel_TunstallScatter_NPU(
    ip_tunstall_lut,          # shape = (256,), int64
    ip_compressed_marks,      # shape = (bsz, N), int8
    ip_starts,                # shape = (bsz, N), int32
    op_decompressed,          # shape = (bsz, M), int8
    N_TILE: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    batch_id = tl.program_id(0)            # each block handles one batch
    ip_compressed_marks += batch_id * N
    ip_starts += batch_id * N
    op_decompressed += batch_id * M
    ip_tunstall_lut += 128

    offs_n = tl.arange(0, N_TILE)          # shape = (N_TILE, )
    for i_block in range(N // N_TILE):
        offs_ip = i_block * N_TILE + offs_n
        marks = tl.load(ip_compressed_marks + offs_ip)     # shape = (N_TILE, ), int8
        starts = tl.load(ip_starts + offs_ip)              # shape = (N_TILE, ), int32
        items = tl.load(ip_tunstall_lut + marks)           # shape = (N_TILE, ), int64
        count = items.to(tl.int32) & 0x7                   # shape = (N_TILE, ), int32
        for i in range(7):
            items >>= 8
            bytes_ = items.to(tl.int8)
            tl.store(op_decompressed + starts + i, bytes_, mask=(i < count) & (starts + i < M))


def triton_tunstall_decode(
    tunstall_lut,
    compressed_marks,
    decompressed,
    N_TILE=128,
):
    assert tunstall_lut.dtype == torch.int64
    assert compressed_marks.dtype == torch.int8
    assert decompressed.dtype == torch.int8
    BSZ, N = compressed_marks.shape
    _, M = decompressed.shape
    assert (BSZ, M) == decompressed.shape
    assert tunstall_lut.shape == (256,)
    assert N % N_TILE == 0
    device_type = compressed_marks.device.type

    if device_type == "cuda":
        # Fast path on CUDA: single kernel decode.
        kernel_TunstallDecode_CUDA[(BSZ,)](tunstall_lut, compressed_marks, decompressed, N_TILE, N, M)
        return

    # Portable/stable path for NPU-like backends: two-stage decode.
    counts = torch.empty((BSZ, N), dtype=torch.int32, device=compressed_marks.device)
    kernel_TunstallCount_NPU[(BSZ,)](tunstall_lut, compressed_marks, counts, N_TILE, N)
    starts = torch.cumsum(counts, dim=1, dtype=torch.int32) - counts
    max_out = int((starts[:, -1] + counts[:, -1]).max().item())
    if max_out > M:
        raise ValueError(f"output buffer too small: required={max_out}, provided={M}")

    kernel_TunstallScatter_NPU[(BSZ,)](
        tunstall_lut, compressed_marks, starts, decompressed, N_TILE, N, M
    )


def reference_tunstall_decode(tunstall_lut, compressed_marks, decompressed):
    BSZ, N = compressed_marks.shape
    _, M = decompressed.shape
    device = compressed_marks.device

    lut_cpu = tunstall_lut.cpu().numpy()
    marks_cpu = compressed_marks.cpu().numpy()

    out_cpu = decompressed.cpu().numpy()
    out_cpu.fill(0)

    for b in range(BSZ):
        out_ptr = 0
        for m in range(N):
            mark = marks_cpu[b, m]
            idx = (128 + int(mark)) & 0xFF
            item = int(lut_cpu[idx])
            count = item & 0x7
            for shift in range(count):
                byte = (item >> ((shift + 1) * 8)) & 0xFF
                assert byte < 128
                if out_ptr >= M:
                    raise IndexError(f"Output buffer too small: out_ptr={out_ptr} >= M={M}")
                out_cpu[b, out_ptr] = byte if byte < 128 else byte - 256
                out_ptr += 1

        print(f"batch{b} decode done, get {out_ptr} bytes")

    decompressed.copy_(torch.from_numpy(out_cpu).to(device=device, dtype=torch.int8))


def random_tunstall_lut_item():
    x = random.randint(0, 99)
    if x < 10:
        return 0x0706050403020107
    elif x < 20:
        return 0xFF06050403020106
    elif x < 30:
        return 0xFFFF050403020105
    elif x < 60:
        return 0xFFFFFF0403020104
    elif x < 80:
        return 0xFFFFFFFF03020103
    elif x < 90:
        return 0xFFFFFFFFFF020102
    else:
        return 0xFFFFFFFFFFFF0101


def random_tunstall_lut(lut_count, device):
    tunstall_lut = [random_tunstall_lut_item() for _ in range(lut_count)]
    tunstall_lut = torch.tensor(tunstall_lut, dtype=torch.uint64, device=device).view(torch.int64)
    return tunstall_lut


def random_tunstall_marks(mark_shape, device):
    return (torch.randn(mark_shape) * 99999).to(dtype=torch.int8, device=device)


def simple_test_tunstall_decode(BSZ, N, N_TILE, device, check=False):
    tunstall_lut = random_tunstall_lut(256, device)
    compressed_marks = random_tunstall_marks((BSZ, N), device)
    decompressed = torch.zeros((BSZ, N * 8), dtype=torch.int8, device=device)
    decompressed_ref = decompressed.clone()
    with show_performance(
        f"decompress shape = {tuple(compressed_marks.shape)}",
        data_volumn=compressed_marks.numel(),
        sync=True,
        device=device,
    ):
        triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, N_TILE)
    if check:
        reference_tunstall_decode(tunstall_lut, compressed_marks, decompressed_ref)
        if (decompressed == decompressed_ref).all():
            print("=== test passed ===")
        else:
            print("*** test failed ***")


def estimate_io_bytes(tunstall_lut, compressed_marks):
    # LUT low 3 bits store decoded-byte count for each symbol.
    count_lut = (tunstall_lut.to(torch.int64) & 0x7).to(torch.int32)
    lut_indices = (compressed_marks.to(torch.int16) + 128).to(torch.int64)
    output_bytes = int(count_lut[lut_indices].sum().item())

    input_bytes = compressed_marks.numel() * compressed_marks.element_size()
    lut_bytes = tunstall_lut.numel() * tunstall_lut.element_size()
    total_bytes = input_bytes + output_bytes + lut_bytes
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "lut_bytes": lut_bytes,
        "total_bytes": total_bytes,
    }


def benchmark_tunstall_decode(
    BSZ,
    N,
    device,
    n_tile_candidates=(64, 128, 256, 512, 1024, 2048, 4096),
    warmup=5,
    repeat=20,
    check=False,
    use_cuda_event=False,
):
    tunstall_lut = random_tunstall_lut(256, device)
    compressed_marks = random_tunstall_marks((BSZ, N), device)
    decompressed = torch.zeros((BSZ, N * 8), dtype=torch.int8, device=device)
    decompressed_ref = decompressed.clone() if check else None

    io_info = estimate_io_bytes(tunstall_lut, compressed_marks)
    print(
        f"benchmark shape={(BSZ, N)}, warmup={warmup}, repeat={repeat}, "
        f"estimated IO={io_info['total_bytes']/1e6:.2f} MB/run, "
        f"timer={'cuda_event' if use_cuda_event else 'perf_counter'}"
    )

    valid_tiles = [x for x in n_tile_candidates if N % x == 0]
    if len(valid_tiles) == 0:
        raise ValueError(f"no valid N_TILE in {n_tile_candidates} for N={N}")

    best = None
    for n_tile in valid_tiles:
        for _ in range(warmup):
            triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, n_tile)
        synchronize_device(device)

        elapsed_ms = []
        for _ in range(repeat):
            if use_cuda_event:
                if device.startswith("cuda"):
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                elif device.startswith("npu"):
                    ev_start = torch.npu.Event(enable_timing=True)
                    ev_end = torch.npu.Event(enable_timing=True)
                else:
                    raise ValueError(f"unsupported device for event timing: {device}")
                ev_start.record()
                triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, n_tile)
                ev_end.record()
                synchronize_device(device)
                elapsed_ms.append(ev_start.elapsed_time(ev_end))
            else:
                start = perf_counter()
                triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, n_tile)
                synchronize_device(device)
                elapsed_ms.append((perf_counter() - start) * 1e3)

        avg_ms = sum(elapsed_ms) / len(elapsed_ms)
        min_ms = min(elapsed_ms)
        max_ms = max(elapsed_ms)
        sorted_ms = sorted(elapsed_ms)
        p50_ms = sorted_ms[int((len(sorted_ms) - 1) * 0.50)]
        p90_ms = sorted_ms[int((len(sorted_ms) - 1) * 0.90)]
        p99_ms = sorted_ms[int((len(sorted_ms) - 1) * 0.99)]
        est_bw_gbps = io_info["total_bytes"] / 1e9 / (avg_ms / 1e3)
        print(
            f"  N_TILE={n_tile:4d} | avg={avg_ms:7.3f} ms | min={min_ms:7.3f} ms | "
            f"p50={p50_ms:7.3f} ms | p90={p90_ms:7.3f} ms | p99={p99_ms:7.3f} ms | "
            f"max={max_ms:7.3f} ms | est BW={est_bw_gbps:7.3f} GB/s"
        )

        if (best is None) or (avg_ms < best["avg_ms"]):
            best = {
                "n_tile": n_tile,
                "avg_ms": avg_ms,
                "est_bw_gbps": est_bw_gbps,
            }

    print(
        f"best N_TILE={best['n_tile']}, avg={best['avg_ms']:.3f} ms, "
        f"est BW={best['est_bw_gbps']:.3f} GB/s"
    )

    if check:
        triton_tunstall_decode(tunstall_lut, compressed_marks, decompressed, best["n_tile"])
        reference_tunstall_decode(tunstall_lut, compressed_marks, decompressed_ref)
        if (decompressed == decompressed_ref).all():
            print("=== correctness check passed ===")
        else:
            print("*** correctness check failed ***")

    return best


def benchmark_tunstall_decode_batch_sweep(
    bsz_list=(1, 4, 16, 64, 256, 1024),
    N=16384,
    device="npu",
    n_tile_candidates=(64, 128, 256, 512, 1024, 2048, 4096),
    warmup=5,
    repeat=20,
    use_cuda_event=False,
):
    print(f"batch sweep = {tuple(bsz_list)}, N={N}")
    sweep_results = []
    for bsz in bsz_list:
        print("\n" + "=" * 88)
        best = benchmark_tunstall_decode(
            BSZ=bsz,
            N=N,
            device=device,
            n_tile_candidates=n_tile_candidates,
            warmup=warmup,
            repeat=repeat,
            check=False,
            use_cuda_event=use_cuda_event,
        )
        best["bsz"] = bsz
        sweep_results.append(best)

    print("\n" + "=" * 88)
    print("batch sweep summary")
    for item in sweep_results:
        print(
            f"  BSZ={item['bsz']:4d} | best N_TILE={item['n_tile']:4d} | "
            f"avg={item['avg_ms']:7.3f} ms | est BW={item['est_bw_gbps']:7.3f} GB/s"
        )
    return sweep_results


if __name__ == "__main__" :
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = "npu"
    else:
        raise RuntimeError("no supported accelerator found: cuda/npu")
    simple_test_tunstall_decode(BSZ=4, N=16384, N_TILE=128, device=device, check=True)
    simple_test_tunstall_decode(BSZ=1024, N=16384, N_TILE=128, device=device)
    simple_test_tunstall_decode(BSZ=1024, N=16384, N_TILE=128, device=device)
    benchmark_tunstall_decode_batch_sweep(
        bsz_list=(1, 4, 16, 64, 256, 1024),
        N=16384,
        device=device,
        n_tile_candidates=(64, 128, 256, 512, 1024, 2048, 4096),
        warmup=5,
        repeat=20,
        use_cuda_event=False,
    )
