import os, json, time, hashlib, pathlib, typing as T
import torch

DEFAULT_DIR = os.path.expanduser(os.getenv("AUTOSAGE_CACHE_DIR", "~/.cache/autosage"))
SCHEDULES_FILE = os.path.join(DEFAULT_DIR, "schedules.jsonl")
LOG_FILE = os.path.join(DEFAULT_DIR, "probe_logs.jsonl")

def _ensure_dir():
    pathlib.Path(DEFAULT_DIR).mkdir(parents=True, exist_ok=True)

def _device_sig(device: T.Optional[int] = None) -> dict:
    dev = torch.device("cuda", device if device is not None else torch.cuda.current_device())
    p = torch.cuda.get_device_properties(dev)
    sig = {
        "name": p.name, "cc": [p.major, p.minor], "sms": p.multi_processor_count,
        "total_mem": int(p.total_memory), "cuda": torch.version.cuda, "torch": torch.__version__,
    }
    uuid = getattr(p, "uuid", None)
    if uuid: sig["uuid"] = str(uuid)
    return sig

def _graph_sig(crow: torch.Tensor, col: torch.Tensor, F: int, *, bins: bool = True) -> dict:
    stats = torch.ops.autosage.compute_graph_stats(crow, col).cpu().tolist()
    N, nnz, mean, mx, q50, q90, q99, heavy = stats
    q = (lambda x: int(round(float(x)))) if bins else (lambda x: float(x))
    sig = {
        "op": "spmm_csr", "F": int(F), "N": int(round(N)), "nnz": int(round(nnz)),
        "q50": q(q50), "q90": q(q90), "q99": q(q99), "heavy": float(f"{heavy:.2f}"),
    }
    try:
        h = hashlib.sha1()
        k = min(col.numel(), 1024)
        if k > 0:
            buf = torch.cat([col[:k], col[-k:]] if col.numel() > k else [col]).contiguous().cpu().numpy().tobytes()
            h.update(buf)
            sig["col_hash16"] = h.hexdigest()[:16]
    except Exception:
        pass
    return sig

class ScheduleCache:
    def __init__(self, path: str = SCHEDULES_FILE):
        _ensure_dir()
        self.path = path
        self._index = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        k = json.dumps({"device": rec.get("device"), "graph": rec.get("graph")}, sort_keys=True)
                        self._index[k] = rec
                    except Exception:
                        continue

    def lookup(self, device_sig: dict, graph_sig: dict):
        k = json.dumps({"device": device_sig, "graph": graph_sig}, sort_keys=True)
        return self._index.get(k)

    def store(self, device_sig: dict, graph_sig: dict, result: dict):
        rec = {"ts": int(time.time()), "device": device_sig, "graph": graph_sig, "result": result}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        k = json.dumps({"device": device_sig, "graph": graph_sig}, sort_keys=True)
        self._index[k] = rec

def log_probe(event: dict):
    _ensure_dir()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass
