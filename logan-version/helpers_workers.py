# helpers_workers.py
import os, time, torch
from torch.utils.data import DataLoader

def recommend_num_workers(io="medium") -> int:
    """
    Return a reasonable num_workers based on SLURM CPUs and device.
    io ∈ {"light","medium","heavy"} describes your input pipeline cost.
    """
    # CPU budget: prefer SLURM's cpus-per-task if set; else machine total
    cpu_budget = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    reserve = 2                            # leave headroom for the main proc & BLAS
    budget = max(0, cpu_budget - reserve)

    # Device-specific baseline
    if torch.cuda.is_available():
        base = {"light": 2, "medium": 4, "heavy": 8}[io]
        return max(0, min(base, budget))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS tends to benefit less from many loader procs; keep it simple
        return 0
    # Pure CPU training
    return max(0, min(2, budget))

def dataloader_kwargs(device: torch.device, io="medium"):
    nw = recommend_num_workers(io)
    return dict(
        num_workers=nw,
        persistent_workers=bool(nw),
        pin_memory=(device.type == "cuda"),
        # modest prefetch; you can raise this if you have RAM to spare
        prefetch_factor=2 if nw else None
    )

# Optional: quick autotuner (tries a few candidates and picks the fastest)
def autotune_num_workers(dataset, batch_size, device, candidates=(0,1,2,4,8), seconds=3):
    # keep candidates within CPU budget
    cpu_budget = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    reserve = 2
    max_ok = max(0, cpu_budget - reserve)
    cands = [c for c in candidates if c <= max_ok]
    if not cands: cands = [0]

    best = None
    best_ips = -1.0
    for nw in cands:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=nw, persistent_workers=bool(nw),
            pin_memory=(device.type == "cuda"), prefetch_factor=(2 if nw else None)
        )
        it = iter(loader)
        # warmup one batch (handles CUDA graph/init etc.)
        try:
            _ = next(it)
        except StopIteration:
            return 0  # empty dataset
        start = time.perf_counter()
        n = 0
        while (time.perf_counter() - start) < seconds:
            try:
                _ = next(it); n += 1
            except StopIteration:
                break
        ips = n / max(1e-6, (time.perf_counter() - start))
        if ips > best_ips:
            best_ips, best = ips, nw
    return best