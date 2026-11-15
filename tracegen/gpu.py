"""GPU discovery and fit checks. Logs every GPU's name and free VRAM; never assumes the GPUs are identical."""
import logging
import subprocess

log = logging.getLogger("tracegen")
MIB = 1024 ** 2
OVERHEAD = 1.15            # weights → VRAM (KV cache, CUDA context) for a quantized Ollama model
CTX_BYTES = 2560 * MIB     # 16k context x 2 slots, q8 KV cache, for a ~8B GQA model
NLI_BYTES = 1800 * MIB     # deberta-large-mnli fp16/fp32 inference footprint, measured roughly


class FitError(SystemExit):
    pass


def probe():
    """[{index, name, total_mib, free_mib}] from nvidia-smi; [] when no NVIDIA GPU/driver is present."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, free = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "total_mib": int(total), "free_mib": int(free)})
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    for g in gpus:
        log.info("GPU %d: %s, %d MiB free of %d MiB", g["index"], g["name"], g["free_mib"], g["total_mib"])
    if not gpus:
        log.warning("no NVIDIA GPU detected (torch sees %d CUDA devices)", n)
    return gpus


def need_bytes(model_size):
    return int(model_size * OVERHEAD + CTX_BYTES)


def check_fit(gpus, devices, items):
    """items: [(label, bytes)] that must sit together on `devices` (summed free VRAM). Raises FitError."""
    by_idx = {g["index"]: g for g in gpus}
    missing = [d for d in devices if d not in by_idx]
    if missing:
        raise FitError("GPU(s) %s are configured but not present (found %s). Fix gpu.* in config.yaml."
                       % (missing, sorted(by_idx)))
    free = sum(by_idx[d]["free_mib"] for d in devices) * MIB
    need = sum(b for _, b in items)
    what = ", ".join("%s ~%.1f GB" % (lbl, b / 1e9) for lbl, b in items)
    if need > free:
        raise FitError(
            "Not enough VRAM on GPU(s) %s: need ~%.1f GB (%s), free %.1f GB.\n"
            "Options: use a smaller quant (e.g. a :q4_K_S / smaller-parameter tag), set gpu.mode: split to spread "
            "the generator over both GPUs, or use an API model (trace.agents entry with base_url)."
            % (devices, need / 1e9, what, free / 1e9))
    log.info("fit OK on GPU(s) %s: %s (free %.1f GB)", devices, what, free / 1e9)
