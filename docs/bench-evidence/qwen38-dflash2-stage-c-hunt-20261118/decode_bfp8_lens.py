#!/usr/bin/env python3
"""The W23 lens-arbitration decoder, rebuilt.

Decodes the RAW device bytes of a staged BFP8 weight under BOTH layouts
(plain row-major BFP8 vs the pin's BFP8_B tile/face-packed order, per
tt_metal/impl/data_format/bfloat8.cpp unpack_bfp8_tiles_into_float_vec),
compares against the METADATA decode (wmeta .f32) and the HOST bf16 source
(whost .b16), and emits a per-tile/face corruption map.

Face-packed BFP8_B tile (32x32, 4 faces 16x16):
  exp area: 16 uint32 LE words (64 B). Exp word k covers mantissa words
  16k..16k+15 (64 datums = 4 face rows = 4 blocks); byte b=(w>>2)&3 holds
  the shared biased-127 exponent for block b (16 datums = one face row).
  data area: 256 uint32 LE words (1024 B); word index within tile for datum
  (r,c): tr=r//16 tc=c//16 i=r%16 j=c%16 -> tr*64 + tc*16 + i*4 + j//4,
  byte lane j%4. mantissa: 7-bit normalized fraction.
  value = float bits: sign<<31 | exp<<23 | frac<<16 (denormals -> 0).
"""
import struct, sys, os
import numpy as np

TILE_H = TILE_W = 32    # the pin's tile shape (32x32, 4 faces of 16x16)
FACE_H = FACE_W = 16
EXP_WORDS = 16                       # round_up(64, l1_alignment=16)/4
TILE_EXP_BYTES = EXP_WORDS * 4       # 64
TILE_DATA_WORDS = 1024 // 4          # 256
TILE_BYTES = TILE_EXP_BYTES + TILE_DATA_WORDS * 4   # 1088


def face_packed_decode(raw, rows, cols):
    """raw: uint8 array of n_tiles*1088 (32x32 BFP8_B tiles). Returns [rows, cols].

    Word order per tile (blockfloat_common.cpp pack / bfloat8.cpp unpack):
    data word w = tr*128 + tc*64 + i*4 + j//4 for datum (tr*16+i, tc*16+j),
    exp word k = w>>4, byte (w>>2)&3 (one shared biased exponent per face row).
    """
    trows, tcols = (rows + 31) // 32, (cols + 31) // 32
    n_tiles = trows * tcols
    assert raw.size >= n_tiles * TILE_BYTES, (raw.size, n_tiles)
    tiles = raw[: n_tiles * TILE_BYTES].reshape(n_tiles, TILE_BYTES)
    expw = tiles[:, :TILE_EXP_BYTES].view(np.uint32).reshape(n_tiles, EXP_WORDS)
    data = tiles[:, TILE_EXP_BYTES:].view(np.uint32).reshape(n_tiles, TILE_DATA_WORDS)
    ti = np.arange(n_tiles)
    out = np.zeros((n_tiles, TILE_H * TILE_W), dtype=np.float32)
    for tr in range(2):
        for tc in range(2):
            for i in range(FACE_H):
                base = tr * 128 + tc * 64 + i * 4
                w = base + np.arange(4)
                lane = np.arange(4)
                diw = w.astype(np.int64)
                k = diw >> 4
                bb = (diw >> 2) & 3
                eword = expw[ti[:, None], k[None, :]].astype(np.uint32)[:, :, None]
                exps = (eword >> (8 * bb)[None, :, None]) & 0xFF
                dw = data[ti[:, None], w[None, :]].astype(np.uint32)[:, :, None]
                mans = ((dw >> (8 * lane)[None, None, :]) & 0xFF).astype(np.uint32)
                sign = (mans >> 7).astype(np.uint32)
                man = (mans & 0x7F).astype(np.uint32)
                # the unpack's re-normalization (bfloat8.cpp:166-199)
                sh = np.zeros(man.shape, dtype=np.uint32)
                mm = man.copy()
                for _ in range(7):
                    set6 = ((mm & 0x40) != 0) | (man == 0)
                    mm = np.where(set6, mm, (mm << 1) & 0xFF)
                    sh = np.where(set6, sh, sh + 1)
                man_n = ((mm << 1) & 0x7F).astype(np.uint32)
                E = (exps.astype(np.int32) - sh.astype(np.int32))
                denorm = (man == 0) | (E <= 0)
                Ef = np.where(denorm, 0, E).astype(np.uint32)
                bits = (sign << 31) | (Ef << 23) | (man_n << 16)
                vals = bits.astype(np.uint32).copy().view(np.float32)
                vals = vals.reshape(vals.shape[0], -1)
                vals[denorm.reshape(vals.shape)] = 0.0
                out[:, (tr * 16 + i) * TILE_W + tc * 16 + np.arange(16)] = vals
    img = out.reshape(trows, tcols, TILE_H, TILE_W).transpose(0, 2, 1, 3).reshape(trows * TILE_H, tcols * TILE_W)
    return img[:rows, :cols]


def plain_decode(raw, rows, cols):
    """The W23 'plain BFP8' arm: contiguous 32-byte blocks (16-byte exponent
    header + 16-byte mantissa) in row-major order over the RAW byte stream."""
    nblocks = raw.size // 32
    blk = raw[: nblocks * 32].reshape(nblocks, 32)
    exps = blk[:, :16].astype(np.uint32)
    mans = blk[:, 16:].astype(np.uint32)
    sign = ((mans >> 7) & 1).astype(np.uint32)
    frac = (mans & 0x7F).astype(np.uint32)
    bits = (sign << 31) | (exps << 23) | (frac << 16)
    vals = bits.copy().view(np.float32)
    vals[(mans & 0x7F) == 0] = 0.0
    flat = vals.reshape(-1)
    if flat.size < rows * cols:
        flat = np.concatenate([flat, np.zeros(rows * cols - flat.size, np.float32)])
    return flat[: rows * cols].reshape(rows, cols)


def decode_bf16_tiled(raw, rows, cols):
    trows, tcols = (rows + 31) // 32, (cols + 31) // 32
    n_tiles = trows * tcols
    tiles = raw[: n_tiles * 2048].reshape(n_tiles, 32, 32, 2)
    out = np.zeros((trows, tcols, 32, 32, 2), dtype=np.uint8)
    out[:, :, :16] = tiles[:, :16].reshape(trows, tcols, 16, 32, 2)
    out[:, :, 16:] = tiles[:, 16:].reshape(trows, tcols, 16, 32, 2)
    u16 = out.reshape(trows * 32, tcols * 32, 2).astype(np.uint32)
    bits = u16[..., 0] | (u16[..., 1] << 8)
    f = bits.astype(np.uint32).copy().view(np.float32)
    return f[:rows, :cols]


def bf16_to_f32(b16):
    b = np.ascontiguousarray(b16).astype(np.uint32)
    return b.copy().view(np.float32)


def asum(v):
    v = np.abs(v[np.isfinite(v)])
    return float(v.sum()) if v.size else 0.0


def main(dump_dir, sq, n):
    tag = lambda name, ext: os.path.join(dump_dir, f"{name}_sq{int(sq):03d}_N{n}.{ext}")
    wraw = np.fromfile(tag("wraw", "raw"), dtype=np.uint8)
    wmeta = np.fromfile(tag("wmeta", "f32"), dtype=np.float32)
    whost = bf16_to_f32(np.fromfile(tag("whost", "b16"), dtype=np.uint8))
    n = int(n)
    cols = 5120
    rows = whost.size // cols
    assert rows == n, (rows, n, whost.size)
    print(f"geometry N x K = {rows} x {cols}, wraw={wraw.size} B, tiles={(rows+31)//32 * (cols+31)//32}, tile_bytes={TILE_BYTES}")

    fp = face_packed_decode(wraw, rows, cols)
    pl = plain_decode(wraw, rows, cols)

    def stat(name, a, b):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rel = d / (np.abs(b) + 1e-30)
        bad = (rel > 0.05) & (d > 1e-3)
        print(f"{name}: n={a.size} diff_max={d.max():.4g} frac_bad={bad.mean():.6%} asum_a={asum(a):.6g} asum_b={asum(b):.6g}")
        return bad

    bad_meta = stat("face-packed vs METADATA", fp, wmeta.reshape(rows, cols))
    bad_host = stat("face-packed vs HOST bf16", fp, whost)
    try:
        pl_stat = stat("plain vs HOST bf16", pl, whost)
    except Exception as e:
        print("plain decode failed:", e)
        pl_stat = None

    # The corruption map: per tile, per face — where the device decode differs
    # from the host source (the actual corruption question), and where the two
    # device decodes differ (the layout question).
    trows, tcols = (rows + 31) // 32, (cols + 31) // 32
    tm = bad_meta.reshape(trows, tcols, 32, 32)
    map_lines = ["tile_row,tile_col,face_row,face_col,bad_elems,bad_frac"]
    for r in range(trows):
        for c in range(tcols):
            for fr in range(2):
                for fc in range(2):
                    face = tm[r, c, fr * 16:(fr + 1) * 16, fc * 16:(fc + 1) * 16]
                    nb = int(face.sum())
                    if nb:
                        map_lines.append(f"{r},{c},{fr},{fc},{nb},{nb/1024:.4f}")
    out_csv = os.path.join(dump_dir, f"corruption_map_sq{int(sq):03d}_N{n}.csv")
    with open(out_csv, "w") as f:
        f.write("\n".join(map_lines) + "\n")
    print(f"bad tiles/faces: {len(map_lines)-1} -> {out_csv}")
    # byte-level identical-tile count between device and host repack
    print("verdict: face-packed decode matches metadata:",
          bad_meta.mean() < 1e-6, "| face-packed matches host:", bad_host.mean() < 5e-3)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10240)
