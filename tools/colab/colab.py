# -*- coding: utf-8 -*-
"""PC side of the Colab worker queue.

  python tools/colab.py bundle                      -> packs the code into bundles/code_<sha>.tar.gz
  python tools/colab.py submit w01 JOB_ID "cmd" [--inputs data/a.tar ...] [--timeout 3600] [--bundle NAME]
  python tools/colab.py status
  python tools/colab.py fetch JOB_ID [--to DIR]     -> copies results/<job> to out/colab/<job>
Jobs run on the node with cwd = /content/work/<job> containing ./code, ./in and ./out.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time

sys.stdout.reconfigure(encoding="utf-8")
# The queue lives in a Google Drive folder that is synced to this PC (Drive for desktop) and mounted on
# Colab as /content/drive/MyDrive/<same path>. Set PAI_COLAB_BASE to the local path of that folder.
BASE = os.environ.get("PAI_COLAB_BASE", os.path.expanduser("~/GoogleDrive/colab_jobs/pai_battery"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIRS = ["sim", "scripts", "perception", "video", "tools"]
WORKERS = ["w01", "w02", "g01"]
LATEST = os.path.join(ROOT, "out", "colab", "latest_bundle.txt")


def ensure_dirs():
    for d in ["bundles", "data", "results"] + [f"workers/{w}/pending" for w in WORKERS]:
        os.makedirs(f"{BASE}/{d}", exist_ok=True)


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d in CODE_DIRS:
            p = os.path.join(ROOT, d)
            if not os.path.isdir(p):
                continue
            for dp, dns, fns in os.walk(p):
                dns[:] = sorted(x for x in dns if x != "__pycache__")
                for fn in sorted(fns):
                    if fn.endswith((".pyc",)) or fn.startswith("_scene_"):
                        continue
                    full = os.path.join(dp, fn)
                    info = t.gettarinfo(full, arcname=os.path.relpath(full, ROOT).replace("\\", "/"))
                    info.mtime = 0
                    with open(full, "rb") as f:
                        t.addfile(info, f)
    data = buf.getvalue()
    name = f"code_{hashlib.sha256(data).hexdigest()[:10]}.tar.gz"
    ensure_dirs()
    dst = f"{BASE}/bundles/{name}"
    if not os.path.exists(dst):
        with open(dst, "wb") as f:
            f.write(data)
    os.makedirs(os.path.dirname(LATEST), exist_ok=True)
    open(LATEST, "w").write(name)
    print(f"bundle {name} ({len(data) / 1e6:.1f} MB)")
    return name


def submit(worker, job_id, cmd, inputs, timeout, bundle_name, env):
    ensure_dirs()
    bundle_name = bundle_name or open(LATEST).read().strip()
    size = os.path.getsize(f"{BASE}/bundles/{bundle_name}")
    job = {"job_id": job_id, "bundle": bundle_name, "bundle_size": size, "cmd": cmd, "inputs": inputs,
           "timeout_sec": timeout, "env": env, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(f"{BASE}/results/{job_id}"):
        sys.exit(f"results/{job_id} already exists: use a new job id")
    p = f"{BASE}/workers/{worker}/pending/{job_id}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=1)
    print(f"submitted {job_id} -> {worker} (bundle {bundle_name})")


def status():
    now = time.time()
    for w in WORKERS:
        hb = f"{BASE}/workers/{w}/heartbeat.json"
        pend = len(os.listdir(f"{BASE}/workers/{w}/pending")) if os.path.isdir(f"{BASE}/workers/{w}/pending") else 0
        if os.path.exists(hb):
            h = json.load(open(hb))
            age = now - h.get("epoch", 0)
            print(f"{w}: beats={h['beats']} age={age:.0f}s gpu={h.get('gpu')} running={list(h['running'])} "
                  f"done={h['done']} failed={h['failed']} pending={pend} last_error={h['last_error']}")
        else:
            print(f"{w}: no heartbeat (pending={pend})")
    res = sorted(os.listdir(f"{BASE}/results")) if os.path.isdir(f"{BASE}/results") else []
    for r in res[-12:]:
        rj = f"{BASE}/results/{r}/result.json"
        if os.path.exists(rj):
            j = json.load(open(rj, encoding="utf-8"))
            print(f"  {r}: rc={j.get('rc')} {j.get('seconds')}s on {j.get('worker')}")
        else:
            print(f"  {r}: (in progress)")


def fetch(job_id, to):
    src = f"{BASE}/results/{job_id}"
    dst = to or os.path.join(ROOT, "out", "colab", job_id)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    j = json.load(open(f"{dst}/result.json", encoding="utf-8"))
    print(json.dumps({k: j[k] for k in j if k != "cmd"}, ensure_ascii=False)[:800])
    tail = open(f"{dst}/stdout.txt", encoding="utf-8", errors="replace").read().splitlines()[-25:]
    print("\n".join(tail))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("bundle")
    s = sp.add_parser("submit")
    s.add_argument("worker")
    s.add_argument("job_id")
    s.add_argument("command")
    s.add_argument("--inputs", nargs="*", default=[])
    s.add_argument("--timeout", type=int, default=3 * 3600)
    s.add_argument("--bundle", default=None)
    s.add_argument("--env", nargs="*", default=[], help="KEY=VALUE")
    sp.add_parser("status")
    f = sp.add_parser("fetch")
    f.add_argument("job_id")
    f.add_argument("--to", default=None)
    a = ap.parse_args()
    if a.cmd == "bundle":
        bundle()
    elif a.cmd == "submit":
        submit(a.worker, a.job_id, a.command, a.inputs, a.timeout, a.bundle,
               dict(kv.split("=", 1) for kv in a.env))
    elif a.cmd == "status":
        status()
    else:
        fetch(a.job_id, a.to)
