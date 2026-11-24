#!/usr/bin/env python3
"""把 qwen35moe GGUF 里被错误量化的高精度张量（ssm_alpha/ssm_beta 等）无损转成 F32。

引擎的 qwen35 路径要求这些张量是 F32/BF16/F16/Q8_0（src/qwen35/q36_engine.cu:847 的 upload_f32），
但有些第三方量化（例如 Ornith 的 "CRACK-Q6_K"）把它们也压成了 Q6_K，导致启动即退出。

做法：反量化该张量的 Q6_K 值 → 以 F32 追加到文件末尾（32 字节对齐）→ 改写它的信息表项（type=0, offset）。
原有张量数据一个字节不动，文件只增长几 MB，数值无损。

  python tools/patch_q36_f32.py <model.gguf> [--dry-run]
"""
import struct, sys, shutil
import numpy as np
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

HIGH_OK = {0, 30, 1, 8}                      # F32 / BF16 / F16 / Q8_0
F32_FAMILIES = ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
                "ssm_norm.weight", "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "post_attention_norm.weight", "output_norm.weight", "ffn_gate_inp.weight",
                "ffn_gate_inp_shexp.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight")
VSIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
ALIGN = 32

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, vt):
    if vt in VSIZE: f.seek(VSIZE[vt], 1); return
    if vt == 8: read_str(f); return
    if vt == 9:
        et = struct.unpack("<I", f.read(4))[0]; cnt = struct.unpack("<Q", f.read(8))[0]
        if et in VSIZE: f.seek(VSIZE[et] * cnt, 1)
        elif et == 8:
            for _ in range(cnt): read_str(f)
        else: raise ValueError(f"array elem {et}")
        return
    raise ValueError(f"value type {vt}")

def main():
    path = sys.argv[1]; dry = "--dry-run" in sys.argv
    r = GGUFReader(path)
    data_start = min(int(t.data_offset) for t in r.tensors)
    targets = [t for t in r.tensors if t.name.split(".", 2)[-1] in F32_FAMILIES and int(t.tensor_type) not in HIGH_OK]
    if not targets:
        print("没有需要修补的张量"); return 0
    print(f"需要修补 {len(targets)} 个张量（{sorted({t.name.split('.',2)[-1] for t in targets})}）")
    if dry:
        for t in targets[:5]: print(f"  {t.name}  {t.tensor_type.name} → F32")
        print("（--dry-run，未写入）"); return 0
    shutil.copyfileobj(open(path, "rb"), open(path + ".header.bak", "wb"), length=8 << 20)  # 头部备份（可回滚）

    with open(path, "r+b") as f:
        f.seek(0); f.read(4)
        ver = struct.unpack("<I", f.read(4))[0]
        nt = struct.unpack("<Q", f.read(8))[0]; nkv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(nkv):
            read_str(f); skip_value(f, struct.unpack("<I", f.read(4))[0])
        entries = {}
        for _ in range(nt):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * nd, 1)
            tp = f.tell(); ty = struct.unpack("<I", f.read(4))[0]
            op = f.tell(); off = struct.unpack("<Q", f.read(8))[0]
            entries[name] = (tp, op, ty, off)
        # 自校验：我的头部解析必须与 gguf-py 读到的类型/偏移完全一致，否则不写
        bad_parse = [t.name for t in r.tensors
                     if entries[t.name][2] != int(t.tensor_type) or data_start + entries[t.name][3] != int(t.data_offset)]
        if bad_parse:
            print(f"解析自校验失败（{len(bad_parse)} 个不一致，例 {bad_parse[:3]}），未写入"); return 2
        print(f"解析自校验通过（{len(entries)} 个张量）")
        f.seek(0, 2); end = f.tell()
        added = 0
        for t in targets:
            tp, op, ty, off = entries[t.name]
            el = int(t.n_elements)
            nb = el // 256 * 210                                   # Q6_K 的字节数
            f.seek(data_start + off); raw = f.read(nb)
            vals = dequantize(np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType.Q6_K).astype(np.float32)
            new_abs = (end + ALIGN - 1) // ALIGN * ALIGN
            f.seek(new_abs); f.write(vals.tobytes())
            f.seek(tp); f.write(struct.pack("<I", 0))               # type = F32
            f.seek(op); f.write(struct.pack("<Q", new_abs - data_start))
            end = new_abs + el * 4; added += el * 4
        f.flush()
    print(f"已修补 {len(targets)} 个张量，追加 {added/1048576:.1f} MiB；头部备份 {path}.header.bak")
    r2 = GGUFReader(path)
    bad = [t.name for t in r2.tensors if t.name.split(".", 2)[-1] in F32_FAMILIES and int(t.tensor_type) not in HIGH_OK]
    print("复查：仍不合规", len(bad), "个" if not bad else f" → {bad[:3]}")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main())
