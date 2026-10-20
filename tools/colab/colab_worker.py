# -*- coding: utf-8 -*-
"""
Colab worker for the battery-picking project.

Pulls job files from Google Drive, runs them on this Colab VM, writes results back to Drive.
  Drive layout (BASE = $PAI_COLAB_BASE):
    bundles/code_<sha>.tar.gz                         <- project code, written by tools/colab.py on the PC
    data/<name>.tar                                   <- inputs shared between jobs (datasets, weights)
    workers/<wid>/pending/<job>.json -> running/ -> done/
    workers/<wid>/heartbeat.json                      <- 'beats' increases every 30 s
    results/<job>/ (stdout.txt, result.json, out/...) <- written as results/<job>.part then renamed
  Each job runs in /content/work/<job>/ with ./code (the bundle), ./in (the job's inputs, untarred) and ./out.
  Everything the job writes to ./out is copied to results/<job>/out/.
Usage (from the notebook): python colab_worker.py <worker_id> [lanes]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback

BASE = os.environ.get("PAI_COLAB_BASE", "/content/drive/MyDrive/colab_jobs/pai_battery")
WORK = "/content/work"
WID = sys.argv[1] if len(sys.argv) > 1 else "w01"
LANES = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) > 0 else 1
WDIR = f"{BASE}/workers/{WID}"
for sub in ("pending", "running", "done"):
    os.makedirs(f"{WDIR}/{sub}", exist_ok=True)
os.makedirs(WORK, exist_ok=True)

HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(
    ["nvidia-smi"], capture_output=True).returncode == 0
state = {"beats": 0, "running": {}, "done": 0, "failed": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "lanes": LANES, "cpus": os.cpu_count(), "gpu": HAS_GPU, "last_error": "", "version": "v1"}
lock = threading.Lock()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def heartbeat_loop():
    while True:
        with lock:
            state["beats"] += 1
            hb = dict(state, ts=time.strftime("%Y-%m-%d %H:%M:%S"), epoch=time.time(), worker=WID,
                      running=dict(state["running"]))
        try:
            tmp = f"{WDIR}/heartbeat.json.tmp"
            with open(tmp, "w") as f:
                json.dump(hb, f)
            os.replace(tmp, f"{WDIR}/heartbeat.json")
        except OSError as e:
            log(f"heartbeat write failed: {e}")
        time.sleep(30)


def wait_for(path, size=None, minutes=30):
    """Drive sync can lag: wait until the file exists (and has the expected size)."""
    deadline = time.time() + minutes * 60
    while True:
        if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
            return
        if time.time() > deadline:
            raise RuntimeError(f"{path} not available on Drive after {minutes} min")
        log(f"waiting for {path} ...")
        time.sleep(20)


def untar(src, dst):
    os.makedirs(dst, exist_ok=True)
    local = f"{dst}/.tmp_{os.path.basename(src)}"
    shutil.copyfile(src, local)                 # one sequential copy is much faster than tar reading FUSE
    with tarfile.open(local) as t:
        try:
            t.extractall(dst, filter="data")
        except TypeError:
            t.extractall(dst)
    os.remove(local)


def run_job(jpath):
    name = os.path.basename(jpath)
    job = json.load(open(jpath))
    jid = job["job_id"]
    t0 = time.time()
    res = {"job_id": jid, "worker": WID, "gpu": HAS_GPU, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cmd": job["cmd"]}
    out_part = f"{BASE}/results/{jid}.part"
    shutil.rmtree(out_part, ignore_errors=True)
    os.makedirs(out_part, exist_ok=True)
    root = f"{WORK}/{jid}"
    try:
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(f"{root}/out")
        b = f"{BASE}/bundles/{job['bundle']}"
        wait_for(b, job.get("bundle_size"))
        untar(b, f"{root}/code")
        for rel in job.get("inputs", []):        # tar files under BASE, unpacked into ./in
            p = f"{BASE}/{rel}"
            wait_for(p, None)
            untar(p, f"{root}/in")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", COLAB_WORKER=WID,
                   MUJOCO_GL="egl" if HAS_GPU else "osmesa", PYOPENGL_PLATFORM="egl" if HAS_GPU else "osmesa",
                   JOB_ID=jid, **job.get("env", {}))
        with open(f"{root}/stdout.txt", "w") as so:
            p = subprocess.Popen(["bash", "-c", job["cmd"]], cwd=root, env=env, stdout=so,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            with lock:
                state["running"][jid] = {"since": time.strftime("%H:%M:%S"), "pgid": p.pid}
            try:
                rc = p.wait(timeout=job.get("timeout_sec", 3 * 3600))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 15)
                time.sleep(5)
                try:
                    os.killpg(p.pid, 9)
                except ProcessLookupError:
                    pass
                rc = "timeout"
        res["rc"] = rc
        shutil.copyfile(f"{root}/stdout.txt", f"{out_part}/stdout.txt")
        shutil.copytree(f"{root}/out", f"{out_part}/out", dirs_exist_ok=True)
        res["outputs"] = sorted(os.path.relpath(p, f"{root}/out")
                                for p in glob.glob(f"{root}/out/**", recursive=True) if os.path.isfile(p))[:200]
    except Exception:
        res["rc"] = "worker_error"
        res["error"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.time() - t0, 1)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{out_part}/result.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    final = f"{BASE}/results/{jid}"
    shutil.rmtree(final, ignore_errors=True)
    os.rename(out_part, final)
    shutil.move(jpath, f"{WDIR}/done/{name}")
    if job.get("cleanup", True):
        shutil.rmtree(root, ignore_errors=True)
    with lock:
        state["running"].pop(jid, None)
        state["done"] += 1
        if res.get("rc") != 0:
            state["failed"] += 1
            state["last_error"] = f"{jid}: rc={res.get('rc')}"
    log(f"job {jid} finished rc={res.get('rc')} in {res['seconds']} s")


def main():
    log(f"worker {WID} lanes={LANES} cpus={os.cpu_count()} gpu={HAS_GPU} base={BASE}")
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    for p in glob.glob(f"{WDIR}/running/*.json"):          # put back jobs from a previous session
        shutil.move(p, f"{WDIR}/pending/{os.path.basename(p)}")
    threads = {}
    while True:
        for jid in [j for j, t in threads.items() if not t.is_alive()]:
            threads.pop(jid)
        if os.path.exists(f"{WDIR}/STOP"):
            if not threads:
                log("STOP file found and no job running: exiting")
                return
        elif len(threads) < LANES:
            pend = sorted(glob.glob(f"{WDIR}/pending/*.json"))
            if pend:
                src = pend[0]
                dst = f"{WDIR}/running/{os.path.basename(src)}"
                try:
                    shutil.move(src, dst)
                    jid = json.load(open(dst))["job_id"]
                except Exception as e:
                    log(f"could not claim {src}: {e}")
                    time.sleep(10)
                    continue
                t = threading.Thread(target=run_job, args=(dst,), daemon=True)
                t.start()
                threads[jid] = t
                log(f"job {jid} started ({len(threads)}/{LANES} lanes busy)")
                continue
        time.sleep(15)


if __name__ == "__main__":
    main()
