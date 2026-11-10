#!/usr/bin/env python3
"""
slurmdash: a live terminal dashboard for Slurm fairshare, job priority,
GPU availability and your own queue.

Written for Killarney (Vector Institute / Alliance Canada), but it only calls
stock Slurm commands (sshare, squeue, sprio, scontrol), so it should work on
most Slurm clusters that have GPU partitions.

  python3 slurmdash.py              live dashboard, Ctrl-C to quit
  python3 slurmdash.py --once       print one frame and exit
  python3 slurmdash.py collect      record one history sample and exit
  python3 slurmdash.py backfill     rebuild usage history from sacct (runs once automatically)
  python3 slurmdash.py --demo       synthetic data, to preview the layout anywhere

History (bottom charts and the "when to submit" banner) is appended to
~/.cache/slurmdash/history-$USER.jsonl every 5 minutes while it runs. On first
launch the usage chart is also rebuilt from sacct, since Slurm keeps job records
but no fairshare history.

Requires Python 3.8+ and rich (pip install rich).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("slurmdash needs the 'rich' package:  pip install --user rich")

# ----------------------------------------------------------------- constants
BLUE, CYAN, GREEN, YELLOW = "#5f87ff", "#5fafff", "#5fd75f", "#d7af00"
RED, MAGENTA, VIOLET, OLIVE = "#e5534b", "#d75fd7", "#af87ff", "#a8a832"
DIM, MUTED, EMPTY, FRAME, REFLINE = "grey50", "grey70", "grey23", "grey35", "grey46"
ISO = "%Y-%m-%dT%H:%M:%S"

# Pending reasons that mean "will not start soon no matter how free the cluster is".
BLOCKED_RE = re.compile(
    r"Dependency|Held|BeginTime|Limit|Max|Grp|PartitionDown|PartitionInactive|"
    r"NodeDown|ReqNodeNotAvail|Licenses|Reservation")
# Node state flags that make a node's idle GPUs unusable.
BAD_STATES = ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING", "NO_RESPOND", "INVALID",
              "MAINT", "REBOOT", "POWER", "FUTURE", "UNKNOWN")
GPU_RE = re.compile(r"gpu(?::(?P<t>[A-Za-z][\w.-]*))?[:=](?P<n>\d+)")

# Satire knobs. Energy assumes GPUs at full TDP plus their share of the host, times a
# datacenter PUE of 1.3, so read it as an upper bound.
FUN = True                                        # --serious turns emojis and jokes off
KWH_PER_GPU_H = {"l40s": (0.35 + 0.13) * 1.3, "h100": (0.70 + 0.07) * 1.3}
HOME_KWH_MONTH = 750                              # Ontario Energy Board's typical household
KETTLE_KWH = 0.1                                  # boiling one full kettle
# Cheapest AWS on-demand list price per GPU-hour (us-east-1): g6e.xlarge for one L40S,
# p5.48xlarge / 8 for an H100. Mid-2026 prices, so the bill is if anything an underestimate.
AWS_USD_PER_GPU_H = {"l40s": 1.861, "h100": 55.04 / 8}
FAIL_STATES = {"FAILED": "crash", "OUT_OF_MEMORY": "OOM"}     # the user's fault, mostly
SHAME_DAYS = 7
PODIUM = ("😈", "👹", "👺")
GPU_EMOJI = {"l40s": "🐴", "h100": "🚀"}
GPU_BASES = ("mi300a", "h200", "h100", "a100", "l40s", "a40", "v100", "p100", "t4", "l4")
GPU_MEM = {"h100": 80, "h200": 141, "a100": 40}   # memory of the full GPU, for MIG slice fractions


def E(s):
    return s if FUN else ""


def canon_type(t):
    """One name per GPU kind. Clusters accept several spellings: 'nvidia_h100_80gb_hbm3_1g.10gb',
    'h100_1g.10gb', 'h100_1.10' and 'h100_10gb' all become 'h100_1g.10gb', and
    'nvidia_h100_80gb_hbm3' or 'h100_80gb' become 'h100'."""
    t = (t or "").lower()
    base = next((b for b in GPU_BASES if b in t), None)
    if base is None:
        return t or None
    m = re.search(r"(\d)g\.(\d+)gb", t) or re.search(rf"{base}_(\d)\.(\d+)$", t)
    if m:
        return f"{base}_{m.group(1)}g.{m.group(2)}gb"
    m = re.search(rf"{base}_(\d+)gb$", t)
    if m and int(m.group(1)) < GPU_MEM.get(base, 999):
        mem = int(m.group(1))
        g = {10: 1, 20: 2, 40: 3}.get(mem)
        return f"{base}_{g}g.{mem}gb" if g else base
    return base


def is_mig(t):
    return bool(t and re.search(r"\dg\.\d+gb", t))


def gpu_frac(t):
    """Share of a full GPU. The Alliance bills a MIG slice as max(compute/7, memory/full)."""
    m = re.search(r"(\d)g\.(\d+)gb", t or "")
    if not m:
        return 1.0
    return max(int(m.group(1)) / 7, int(m.group(2)) / GPU_MEM.get(t.split("_")[0], 80))


def pretty_type(t):
    if not t:
        return "GPU"
    m = re.search(r"\dg\.\d+gb", t)
    return f"{t.split('_')[0].upper()} {m.group(0)} MIG" if m else t.upper()


def short_type(t):
    """Compact label for tight columns: 'H100/1g' for a MIG slice, 'L40S' for a full GPU."""
    m = re.search(r"(\d)g\.\d+gb", t or "")
    return f"{t.split('_')[0].upper()}/{m.group(1)}g" if m else (t or "gpu").upper()


def gpu_emoji(t):
    return "🍕" if is_mig(t) else GPU_EMOJI.get(t, "🎮")     # a slice of a GPU


def kwh_rate(gpu_type):
    return KWH_PER_GPU_H.get((gpu_type or "").split("_")[0], 0.65) * gpu_frac(gpu_type)


def usd_rate(gpu_type):
    base = (gpu_type or "").split("_")[0]
    return AWS_USD_PER_GPU_H.get(base, AWS_USD_PER_GPU_H["l40s"]) * gpu_frac(gpu_type)


def fmt_usd(x):
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    if x >= 1e4:
        return f"${x / 1e3:.0f}k"
    return f"${x / 1e3:.1f}k" if x >= 1e3 else f"${x:.0f}"


LabJob = namedtuple("LabJob", "user start end gpus type submit state array")


SSHARE_ME = "--format=Account,User,RawUsage"
SSHARE_LAB = "--format=Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare,LevelFS"
SSHARE_ALL = "--format=Account,User,NormShares,EffectvUsage,FairShare"
SQUEUE_FMT = "%i|%u|%a|%P|%t|%Q|%L|%b|%D|%S|%M|%l|%R|%j"   # name last: it may contain '|'
SPRIO_FMT = "%i|%Y|%A|%F|%J|%P|%Q|%T"


# ------------------------------------------------------------------- helpers
def fnum(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def fmt_int(x):
    return "-" if x is None else f"{int(round(x)):,}"


def fmt_k(x):
    if x is None:
        return "-"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def fmt_hours(h):
    if h is None:
        return "-"
    if h < 1:
        return f"{h * 60:.0f}m"
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def parse_hours(s):
    """Slurm duration ('1-02:03:04', '2:03:04', '3:04') to hours, or None."""
    s = (s or "").strip()
    if not s or not s[0].isdigit():
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        if ":" not in s:
            return days * 24 + int(s)
    parts = [int(p) for p in s.split(":")]
    parts = [0] * (3 - len(parts)) + parts
    h, m, sec = parts[-3:]
    return days * 24 + h + m / 60 + sec / 3600


def hms(h):
    s = int(max(h, 0) * 3600)
    d, s = divmod(s, 86400)
    hh, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{d}-{hh:02d}:{m:02d}:{s:02d}" if d else f"{hh}:{m:02d}:{s:02d}"


def gpus_in(s):
    """(count, type) from a gres/TRES string like 'gres/gpu:l40s=2' or 'gpu:h100:8(S:0-1)'."""
    if not s or s in ("N/A", "(null)"):
        return 0, None
    untyped, typed, typ = [], 0, None
    for m in GPU_RE.finditer(s):
        n = int(m.group("n"))
        if m.group("t"):
            typed += n
            typ = typ or canon_type(m.group("t"))
        else:
            untyped.append(n)
    return (max(untyped) if untyped else typed), typ


def summ(vals):
    vals = [v for v in vals if v is not None and not math.isinf(v)]
    return (statistics.median(vals), min(vals), max(vals)) if vals else None


def short_part(p, types):
    """'gpubase_l40s_b1' -> 'l40s_b1' (cut everything before the GPU type)."""
    out = []
    for q in p.split(","):
        low = q.lower()
        hits = [low.find(t) for t in types if t in low]
        if hits and min(hits) > 0:
            out.append(q[min(hits):])
        else:                                   # Alliance: 'gpubase_bygpu_b1' -> 'bygpu_b1'
            out.append(re.sub(r"^(gpu|cpu)base_", "", q))
    return ",".join(out)


def hour_range(h):
    a, b = h % 12 or 12, (h + 1) % 12 or 12
    return f"{a}-{b}{'am' if (h + 1) % 24 < 12 else 'pm'}"


def wait_text(h):
    if h is None:
        return "unknown", DIM
    if h < 0.25:
        return "<15 min", GREEN
    if h < 1:
        return "<1 h", GREEN
    if h < 2:
        return "~1–2 h", YELLOW
    if h < 6:
        return "~2–6 h", YELLOW
    if h < 24:
        return "6–24 h", RED
    return f"~{h / 24:.0f} day{'s' if h >= 36 else ''}", RED


def xs_style(x):
    return GREEN if x < 0.5 else (YELLOW if x <= 1.0 else RED)


# -------------------------------------------------------------- slurm access
class Slurm:
    """Runs Slurm CLI commands. FakeSlurm stands in for it in --demo."""

    def run(self, args, timeout=60):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise RuntimeError(f"{args[0]} not found (run this on a Slurm login node)")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{args[0]} timed out after {timeout}s")
        if r.returncode != 0:
            raise RuntimeError(f"{args[0]} failed: {r.stderr.strip()[:160]}")
        return r.stdout


class Collector:
    """Pulls everything the dashboard needs, caching the slow-changing parts."""

    def __init__(self, slurm, user, account=None, sshare_every=120, node_every=30, usage_every=600):
        self.slurm, self.user, self.account = slurm, user, account
        self.sshare_every, self.node_every, self.usage_every = sshare_every, node_every, usage_every
        self._cache = {}
        # lab GPU jobs from sacct, kept between polls: jobid -> (user, start, end or None, gpus)
        self._usage_jobs, self._usage_since, self._usage_ts = {}, None, 0.0
        self._usage_ok, self._usage_err, self._usage_busy = False, None, False
        self._usage_lock = threading.Lock()
        self._usage_users = set()
        self.sync_usage = False                  # --once / collect: poll sacct inline
        self._bgstate = {}

    def _cached(self, key, every, fn, default=None):
        ts, val = self._cache.get(key, (0.0, None))
        if val is not None and time.time() - ts < every:
            return val
        try:
            val = fn()
        except RuntimeError:
            if val is not None:
                return val                      # keep showing the last good value
            if default is not None:
                return default
            raise
        self._cache[key] = (time.time(), val)
        return val

    # -- individual queries ---------------------------------------------------
    def detect_account(self):
        out = self.slurm.run(["sshare", "-P", "-n", "-U", "-u", self.user, SSHARE_ME])
        rows = [[x.strip() for x in line.split("|")] for line in out.splitlines()]
        accts = [(r[0], fnum(r[2])) for r in rows if len(r) >= 3 and r[1] == self.user]
        if not accts:
            raise RuntimeError(f"no fairshare association for {self.user}; pass --account")
        env = os.environ.get("SLURM_ACCOUNT") or os.environ.get("SBATCH_ACCOUNT")
        names = [a for a, _ in accts]
        for cand in ([f"{env}_gpu", env] if env else []):
            if cand in names:
                return cand
        gpu = [a for a in accts if a[0].endswith("_gpu")]     # Alliance: def-lab_cpu and def-lab_gpu
        return max(gpu or accts, key=lambda a: a[1])[0]

    def lab_shares(self):
        out = self.slurm.run(["sshare", "-P", "-n", "-a", "-A", self.account, SSHARE_LAB])
        acct, users = None, []
        for line in out.splitlines():
            f = [x.strip() for x in line.split("|")]
            if len(f) < 8 or f[0] != self.account:
                continue
            row = dict(user=f[1], norm=fnum(f[3]), eff=fnum(f[5]), fs=fnum(f[6]),
                       levelfs=fnum(f[7], None))
            row["xs"] = row["eff"] / row["norm"] if row["norm"] > 0 else 0.0
            if f[1]:
                users.append(row)
            else:
                acct = row
        users.sort(key=lambda r: (-r["fs"], r["user"]))
        return acct, users

    def cluster_shares(self):
        """Every user (account, user, fairshare, xS) and every account's xS, cluster-wide."""
        out = self.slurm.run(["sshare", "-P", "-n", "-a", SSHARE_ALL], timeout=90)
        users, accts = [], {}
        for line in out.splitlines():
            f = [x.strip() for x in line.split("|")]
            if len(f) < 5:
                continue
            norm = fnum(f[2])
            xs = fnum(f[3]) / norm if norm > 0 else 0.0
            if f[1]:
                users.append((f[0], f[1], fnum(f[4]), xs))
            else:
                accts[f[0]] = xs
        return {"users": users, "accts": accts}

    def lab_gpuh(self):
        """GPU-hours per account over the last 30 days. sreport reads Slurm's hourly
        rollups, so it is far cheaper than a cluster-wide sacct."""
        now = time.time()
        out = self.slurm.run(["sreport", "-P", "-t", "Hours", "-T", "gres/gpu", "cluster",
                              "AccountUtilizationByUser",
                              f"start={datetime.fromtimestamp(now - BOARD_DAYS * 86400):%Y-%m-%dT%H:00:00}",
                              f"end={datetime.fromtimestamp(now):%Y-%m-%dT%H:00:00}"], timeout=180)
        hdr, res = None, {}
        for line in out.splitlines():
            f = [x.strip() for x in line.split("|")]
            if hdr is None:
                if "Account" in f and "Used" in f:
                    hdr = {k: i for i, k in enumerate(f)}
                continue
            if len(f) < len(hdr):
                continue
            if not (f[hdr["Login"]] if "Login" in hdr else ""):        # account rows only
                res[f[hdr["Account"]]] = res.get(f[hdr["Account"]], 0.0) + fnum(f[hdr["Used"]])
        if hdr is None:
            raise RuntimeError("unexpected sreport output")
        return res

    def _bg(self, key, every, fn):
        """Like _cached, but the query runs in a thread so the UI never waits on it."""
        st = self._bgstate.setdefault(key, {"ts": 0.0, "val": None, "err": None, "busy": False})
        if not st["busy"] and time.time() - st["ts"] >= every:
            st["busy"] = True

            def work():
                try:
                    st["val"], st["err"] = fn(), None
                except Exception as e:
                    st["err"] = str(e)[:100]
                finally:
                    st["ts"], st["busy"] = time.time(), False
            if self.sync_usage:
                work()
            else:
                threading.Thread(target=work, daemon=True).start()
        return st["val"], st["err"]

    def config(self):
        out = self.slurm.run(["scontrol", "show", "config"])
        return dict(re.findall(r"^(\w+)\s*=\s*(.*?)\s*$", out, re.M))

    def nodes(self):
        out = self.slurm.run(["scontrol", "show", "node", "--oneliner"])
        nodes = []
        for line in out.splitlines():
            if not line.startswith("NodeName="):
                continue
            kv = {}
            for k, v in re.findall(r"(\w+)=(\S*)", line):
                kv.setdefault(k, v)
            total, typ = gpus_in(kv.get("CfgTRES", ""))
            g_total, g_typ = gpus_in(kv.get("Gres", ""))
            total, typ = total or g_total, typ or g_typ
            if not total:
                continue                         # CPU-only node
            alloc, _ = gpus_in(kv.get("AllocTRES", ""))
            if not alloc and "GresUsed" in kv:
                alloc, _ = gpus_in(kv["GresUsed"])
            nodes.append(dict(name=kv["NodeName"], type=typ or "gpu", total=total,
                              alloc=min(alloc, total), state=kv.get("State", "").upper(),
                              parts=[p for p in kv.get("Partitions", "").split(",") if p]))
        return nodes

    def jobs(self):
        out = self.slurm.run(["squeue", "-h", "-r", "-t", "PD,R,CF", "-o", SQUEUE_FMT], timeout=90)
        jobs = []
        for line in out.splitlines():
            f = line.split("|", 13)
            if len(f) < 14:
                continue
            jid, user, acct, part, st, prio, left, tres, nn, start, used, limit, reason, name = \
                [x.strip() for x in f]
            g, rtype = gpus_in(tres)
            jobs.append(dict(id=jid, user=user, acct=acct, part=part, st=st, prio=fnum(prio), rtype=rtype,
                             left_h=parse_hours(left), gpus=g * max(1, int(fnum(nn, 1))),
                             start=start, used=used, limit=limit, reason=reason.strip("()"),
                             name=name))
        return jobs

    def my_alloc(self):
        """Allocated GPUs of your running jobs, keyed like squeue's %i ('123' or '123_4').
        %b misses GPUs requested with --gpus, and CPU-only jobs in GPU partitions would
        otherwise be guessed as 1 GPU."""
        out = self.slurm.run(["squeue", "-h", "-u", self.user, "-t", "R",
                              "-O", "ArrayJobID:20,ArrayTaskID:20,tres-alloc:600"])
        res = {}
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 2:
                jid = p[0] if p[1] in ("N/A", "") else f"{p[0]}_{p[1]}"
                res[jid] = gpus_in(p[2] if len(p) > 2 else "")
        return res

    def sprio(self):
        out = self.slurm.run(["sprio", "-h", "-u", self.user, "-o", SPRIO_FMT])
        res = {}
        for line in out.splitlines():
            f = [x.strip() for x in line.split("|")]
            if len(f) < 8:
                continue
            tres = sum(fnum(v) for v in re.findall(r"=([\d.]+)", f[7])) if "=" in f[7] else fnum(f[7])
            d = dict(total=fnum(f[1]), age=fnum(f[2]), fair=fnum(f[3]), size=fnum(f[4]),
                     part=fnum(f[5]), qos=fnum(f[6]), tres=tres)
            if f[0] not in res or d["total"] > res[f[0]]["total"]:
                res[f[0]] = d
        return res

    def _refresh_usage(self):
        """Poll sacct for lab GPU jobs. The first poll covers 30 days, later ones only
        the time since the previous poll (plus whatever is still running)."""
        try:
            now = time.time()
            since = (self._usage_since or now - BOARD_DAYS * 86400) - 3600
            out = sacct_range(self.slurm, ["-a", "-X", "-n", "-P", "-A", self.account, "-o",
                                           "JobIDRaw,JobID,User,Submit,Start,End,AllocTRES,State"], since)
            new = {}
            for line in out.splitlines():
                f = [x.strip() for x in line.split("|")]
                if len(f) >= 8:
                    self._usage_users.add(f[2])  # any job, GPU or not: used to spot hidden labmates
                if len(f) < 8 or parse_ts(f[4]) is None:
                    continue                     # never started
                g, typ = gpus_in(f[6])
                if g:
                    new[f[0]] = LabJob(f[2], parse_ts(f[4]), parse_ts(f[5]), g, typ, parse_ts(f[3]),
                                       f[7].split()[0] if f[7] else "", "_" in f[1])
            cutoff = now - BOARD_DAYS * 86400
            with self._usage_lock:
                self._usage_jobs.update(new)
                self._usage_jobs = {k: v for k, v in self._usage_jobs.items() if v.end is None or v.end > cutoff}
                self._usage_since, self._usage_ok, self._usage_err = now, True, None
        except Exception as e:
            self._usage_err = f"sacct failed: {e}"[:120]
        finally:
            self._usage_ts, self._usage_busy = time.time(), False

    def usage_boards(self):
        if not self._usage_busy and time.time() - self._usage_ts >= self.usage_every:
            self._usage_busy = True
            if self.sync_usage:
                self._refresh_usage()
            else:                                # a 30-day sacct can take a while: don't block the UI
                threading.Thread(target=self._refresh_usage, daemon=True).start()
        boards = shame = None
        with self._usage_lock:
            if self._usage_ok:
                boards = compute_boards(self._usage_jobs, time.time())
                shame = compute_shame(self._usage_jobs, time.time())
        return boards, shame, self._usage_err

    # -- one full snapshot -----------------------------------------------------
    def snapshot(self):
        if not self.account:
            self.account = self.detect_account()
        acct, board = self._cached("lab", self.sshare_every, self.lab_shares)
        clu = self._cached("clu", self.sshare_every * 3, self.cluster_shares,
                           default={"users": [], "accts": {}})
        cfg = self._cached("config", 3600, self.config, default={})
        weights = {k[14:].lower(): fnum(v) for k, v in cfg.items()
                   if k.startswith("PriorityWeight") and v.isdigit()}
        nodes = self._cached("nodes", self.node_every, self.nodes)
        jobs = self.jobs()
        try:
            alloc = self.my_alloc()
        except RuntimeError:
            alloc = {}                           # older squeue: fall back to the %b guess
        for j in jobs:
            if j["user"] == self.user and j["id"] in alloc:
                j["gpus"], j["exact"] = alloc[j["id"]][0], True
                j["rtype"] = alloc[j["id"]][1] or j.get("rtype")
        sprio = {}
        if any(j["user"] == self.user and j["st"] == "PD" for j in jobs):
            try:
                sprio = self.sprio()
            except RuntimeError:
                pass
        snap = build_snapshot(self.user, self.account, acct, board, clu, weights, nodes, jobs, sprio)
        snap["ts"] = time.time()
        snap["halflife_h"] = parse_hours(cfg.get("PriorityDecayHalfLife", ""))
        snap["boards"], snap["shame"], snap["boards_note"] = self.usage_boards()
        if snap["boards"] is not None and self._usage_users <= {self.user} and \
                any(r["eff"] > 0 for r in board if r["user"] != self.user):
            snap["boards_note"] = "sacct only shows your own jobs on this cluster"
        gpuh, err = self._bg("sreport", 1800, self.lab_gpuh)
        our_kwh = snap["boards"]["month"]["kwh"] if snap["boards"] else None
        our_usd = snap["boards"]["month"]["usd"] if snap["boards"] else None
        snap["labs"] = build_labs(gpuh, clu, snap["acct_now"], self.account, our_kwh, snap["kwh_rate"],
                                  our_usd, snap["usd_rate"])
        snap["labs_note"] = (None if gpuh is not None else
                             f"sreport failed ({err}), ranked by decayed usage" if err else "querying sreport…")
        return snap


def gpu_util(slurm, jobid):
    """nvidia-smi inside a running job (first node only): [(util %, mem GB, total GB)]."""
    out = slurm.run(["srun", f"--jobid={jobid}", "--overlap", "--ntasks=1", "--quiet",
                     "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], timeout=25)
    vals = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3:
            vals.append((fnum(p[0]), fnum(p[1]) / 1024, fnum(p[2]) / 1024))
    return vals


# --------------------------------------------------------------- the model
def gpu_summary(nodes):
    types = {}
    for n in nodes:
        T = types.setdefault(n["type"], dict(total=0, used=0, free=0, dead=0, dead_nodes=0, resv=0))
        idle = n["total"] - n["alloc"]
        T["total"] += n["total"]
        T["used"] += n["alloc"]
        if any(b in n["state"] for b in BAD_STATES):
            T["dead"] += idle
            T["dead_nodes"] += 1 if idle else 0
        elif "RESERVED" in n["state"]:
            T["resv"] += idle
        else:
            T["free"] += idle
    return types


def partition_types(nodes):
    votes = {}
    for n in nodes:
        for p in n["parts"]:
            votes.setdefault(p, Counter())[n["type"]] += n["total"] * gpu_frac(n["type"])
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def estimate_wait(free, need, running, capacity):
    """(hours, GPU starts per hour) until `need` GPUs could start for you.

    Uses the start rate seen over the last 3 h (running jobs' elapsed times), which
    undercounts short jobs that already finished, so it leans pessimistic. Falls back
    to an upper bound from time limits. Ignores backfill: short jobs often start sooner.
    """
    if free >= need:
        return 0.0, None
    recent = [j for j in running if (h := parse_hours(j["used"])) is not None and h <= 3]
    rate = sum(j["gpus"] for j in recent) / 3.0
    if rate > 0:
        return (need - free) / rate, rate
    have, last = free, 0.0
    for left, g in sorted((j["left_h"], j["gpus"]) for j in running if j["left_h"] is not None):
        have, last = have + g, left
        if have >= need:
            return left, None
    limits = [h for h in (parse_hours(j["limit"]) for j in running) if h]
    if not limits or capacity <= 0:
        return None, None
    return last + (need - have) / (capacity / statistics.mean(limits)), None


def build_snapshot(user, account, acct_row, board, clu_rows, weights, nodes, jobs, sprio):
    types = gpu_summary(nodes)
    order = sorted(types, key=lambda t: -types[t]["total"] * gpu_frac(t))   # full GPUs before MIG slices
    p2t = partition_types(nodes)
    for j in jobs:                              # partitions can mix GPU kinds: trust the request first
        j["type"] = j.get("rtype") if j.get("rtype") in types else \
            next((p2t[p] for p in j["part"].split(",") if p in p2t), None)
        if j["type"] and not j["gpus"] and not j.get("exact"):
            j["gpus"] = 1                       # e.g. --gpus=N jobs whose %b shows N/A

    me = next((r for r in board if r["user"] == user), None)
    fs = me["fs"] if me else 0.0
    est_new = weights.get("fairshare", 0.0) * fs     # what a fresh job gets from fairshare alone
    mine = [j for j in jobs if j["user"] == user]

    for t in order:
        T = types[t]
        pend = [j for j in jobs if j["type"] == t and j["st"] == "PD"]
        run = [j for j in jobs if j["type"] == t and j["st"] in ("R", "CF")]
        mp = [j["prio"] for j in pend if j["user"] == user and not BLOCKED_RE.search(j["reason"])]
        mp = mp or [j["prio"] for j in pend if j["user"] == user]
        ref = max(mp) if mp else est_new
        T.update(pending=len(pend), my_prio=ref, est=not mp,
                 max_prio=max((j["prio"] for j in pend), default=0.0))
        T["hist"] = [sum(j["prio"] > 2 * ref for j in pend),
                     sum(1.5 * ref < j["prio"] <= 2 * ref for j in pend),
                     sum(ref < j["prio"] <= 1.5 * ref for j in pend),
                     sum(j["prio"] <= ref for j in pend)]
        T["ahead"] = sum(T["hist"][:3])
        need = 1 + sum(j["gpus"] for j in pend
                       if j["prio"] > ref and not BLOCKED_RE.search(j["reason"]))
        T["wait_h"], T["rate"] = estimate_wait(T["free"], need, run, T["total"] - T["dead"])

    # priority breakdown of your best pending job (or an estimate for a new one)
    pd_mine = sorted((j for j in mine if j["st"] == "PD"), key=lambda j: -j["prio"])
    top = next((j for j in pd_mine if not BLOCKED_RE.search(j["reason"])),
               pd_mine[0] if pd_mine else None)
    comps = ("fair", "age", "part", "qos", "tres", "size")
    if top:
        parts = sprio.get(top["id"]) or sprio.get(top["id"].split("_")[0]) or {}
        prio = {k: parts.get(k, 0.0) for k in comps}
        prio.update(total=parts.get("total") or top["prio"], job=top["id"], est=False,
                    type=top["type"], breakdown=bool(parts))
    else:
        prio = {k: 0.0 for k in comps}
        prio.update(total=est_new, fair=est_new, job=None, est=True, breakdown=True, type=None)
    prio["type"] = prio["type"] if prio["type"] in types else (order[0] if order else None)
    prio["max"] = types[prio["type"]]["max_prio"] if prio["type"] else 0.0

    acct_row = acct_row or dict(xs=0.0, norm=0.0, eff=0.0)
    fs_info = dict(factor=fs, xs=me["xs"] if me else 0.0, eff=me["eff"] if me else 0.0,
                   norm=me["norm"] if me else 0.0,
                   levelfs=me["levelfs"] if me else None, missing=me is None)
    pd_all = [j for j in jobs if j["st"] == "PD" and j["type"]]
    acct_now = Counter()
    for j in jobs:
        if j["st"] in ("R", "CF") and j["type"]:
            acct_now[j["acct"]] += j["gpus"]
    cap = sum(types[t]["total"] for t in order) or 1
    return dict(
        user=user, account=account, fs=fs_info, acct=acct_row, board=board,
        rank=board.index(me) + 1 if me else None,
        types=types, type_order=order, prio=prio,
        myjobs=sorted(mine, key=lambda j: (j["st"] == "PD", -j["prio"])),
        acct_now=dict(acct_now), kwh_rate=sum(types[t]["total"] * kwh_rate(t) for t in order) / cap,
        usd_rate=sum(types[t]["total"] * usd_rate(t) for t in order) / cap,
        lab=dict(fs=summ([r["fs"] for r in board]), xs=summ([r["xs"] for r in board]),
                 prio=summ([j["prio"] for j in pd_all if j["acct"] == account])),
        clu=dict(fs=summ([r[2] for r in clu_rows["users"]]), xs=summ([r[3] for r in clu_rows["users"]]),
                 prio=summ([j["prio"] for j in pd_all])),
    )


# ------------------------------------------------------------------ history
def history_sample(snap):
    return {"ts": round(snap["ts"]), "fs": round(snap["fs"]["factor"], 5),
            "xs": round(snap["fs"]["xs"], 4), "prio": round(snap["prio"]["total"]),
            "types": {t: {"p": v["pending"], "a": v["ahead"], "f": v["free"],
                          "w": None if v["wait_h"] is None else round(v["wait_h"], 2)}
                      for t, v in snap["types"].items()}}


class History:
    def __init__(self, path, keep_days=180):
        self.path = Path(path).expanduser() if path else None
        self.keep_s = keep_days * 86400
        self.samples, self._mtime = [], 0.0
        self.lock = threading.Lock()
        self.reload()

    def reload(self):
        if not self.path or not self.path.exists():
            return
        cutoff, out, n = time.time() - self.keep_s, [], 0
        with self.path.open() as f:
            for line in f:
                n += 1
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if s.get("ts", 0) >= cutoff:
                    out.append(s)
        out.sort(key=lambda s: s["ts"])
        self.samples, self._mtime = out, self.path.stat().st_mtime
        if n - len(out) > 5000:                  # drop expired lines from the file
            self._write_all()

    def _write_all(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(s, separators=(",", ":")) + "\n" for s in self.samples))
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    def has_backfill(self):
        return any(s.get("bf") for s in self.samples)

    def set_backfill(self, bf):
        """Replace reconstructed samples, keeping only those older than the first live one."""
        with self.lock:
            if self.path and self.path.exists() and self.path.stat().st_mtime > self._mtime:
                self.reload()
            live = [s for s in self.samples if not s.get("bf")]
            first = live[0]["ts"] if live else float("inf")
            self.samples = sorted(live + [s for s in bf if s["ts"] < first], key=lambda s: s["ts"])
            if self.path:
                self._write_all()

    def maybe_add(self, sample, every):
        with self.lock:
            if self.path and self.path.exists() and self.path.stat().st_mtime > self._mtime:
                self.reload()                    # another slurmdash or `collect` wrote meanwhile
            last = next((x["ts"] for x in reversed(self.samples) if not x.get("bf")), None)
            if last is not None and sample["ts"] - last < every:
                return False
            self.samples = self.samples + [sample]
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                self._mtime = self.path.stat().st_mtime
            return True


SACCT_FMT = "JobIDRaw,User,Start,End,AllocTRES,Priority"
_SACCT_SPAN = [None]          # widest date range this cluster's sacct has accepted, in seconds


def sacct_range(slurm, args, t_from, timeout=600):
    """sacct from t_from to now, split into smaller windows when slurmdbd refuses a wide
    one ("Too wide of a date range": its MaxQueryTimeRange). A job spanning two windows
    comes back twice, so callers dedupe by JobIDRaw."""
    now = time.time()
    span = _SACCT_SPAN[0] or (now - t_from)
    while True:
        try:
            parts, a = [], t_from
            while a < now:
                b = min(now, a + span)
                if now - b < 3600:              # don't leave a sliver of a window at the end
                    b = now
                parts.append(slurm.run(["sacct", *args, "-S", datetime.fromtimestamp(a).strftime(ISO),
                                        "-E", "now" if b >= now else datetime.fromtimestamp(b).strftime(ISO)],
                                       timeout=timeout))
                a = b
            return "".join(parts)
        except RuntimeError as e:
            if "too wide" not in str(e).lower() or span <= 86400:
                raise
            span = max(86400.0, span / 2)
            _SACCT_SPAN[0] = span


BOARD_DAYS = 30
BOARDS = (("month", "PAST MONTH", 30), ("week", "PAST WEEK", 7), ("day", "PAST DAY", 1))


def compute_boards(jobs, now):
    """GPU-hours per user inside each rolling window (biggest user first), plus the
    lab's total GPU-hours and estimated energy for that window."""
    out = {}
    for key, _, days in BOARDS:
        t0, tot, kwh, usd_by = now - days * 86400, {}, 0.0, {}
        for j in jobs.values():
            hi, lo = min(j.end or now, now), max(j.start, t0)
            if hi > lo:
                h = j.gpus * (hi - lo) / 3600
                tot[j.user] = tot.get(j.user, 0.0) + h
                usd_by[j.user] = usd_by.get(j.user, 0.0) + h * usd_rate(j.type)
                kwh += h * kwh_rate(j.type)
        out[key] = dict(rows=sorted(tot.items(), key=lambda x: (-x[1], x[0])),
                        total=sum(tot.values()), kwh=kwh, usd=sum(usd_by.values()), usd_by=usd_by)
    return out


def compute_shame(jobs, now, days=SHAME_DAYS):
    """Wall of failure (crashed and OOM-killed jobs, ranked by GPU-hours that produced
    nothing) and achievement badges, over the past week."""
    t0 = now - days * 86400
    wall, arrays, owls, timeouts, patience = {}, Counter(), Counter(), Counter(), {}
    for j in jobs.values():
        ended_here = j.end is not None and j.end >= t0
        cause = FAIL_STATES.get(j.state)
        if cause and ended_here:
            h = j.gpus * (j.end - j.start) / 3600
            w = wall.setdefault(j.user, dict(n=0, gpuh=0.0, usd=0.0, causes=Counter()))
            w["n"] += 1
            w["gpuh"] += h
            w["usd"] += h * usd_rate(j.type)
            w["causes"][cause] += 1
        if j.state == "TIMEOUT" and ended_here:
            timeouts[j.user] += 1
        if j.submit is not None and j.submit >= t0:
            arrays[j.user] += j.array
            owls[j.user] += 2 <= datetime.fromtimestamp(j.submit).hour < 5
        if j.submit is not None and j.start >= t0 and j.start - j.submit > patience.get(j.user, 0.0):
            patience[j.user] = j.start - j.submit
    wall_rows = sorted(wall.items(), key=lambda kv: (-kv[1]["gpuh"], kv[0]))

    def winner(counts, describe):
        counts = {u: n for u, n in counts.items() if n}
        if not counts:
            return None, None
        u = max(counts, key=lambda k: (counts[k], k))
        return u, describe(counts[u])

    badges = [("🏹", "Array Lord", *winner(arrays, lambda n: f"{n:,} array tasks")),
              ("🦉", "Night Owl", *winner(owls, lambda n: f"{n:,} jobs at 2–5 am")),
              ("🎰", "Gambler", *winner(timeouts, lambda n: f"{n:,} timeouts")),
              ("🐌", "Saint of Patience", *winner(patience, lambda sec: f"waited {fmt_hours(sec / 3600)} once"))]
    return dict(wall=wall_rows, badges=badges,
                wasted_gpuh=sum(w["gpuh"] for _, w in wall_rows), wasted_usd=sum(w["usd"] for _, w in wall_rows))


def redemption_days(eff, norm, halflife_h):
    """Days of not submitting until usage decays back to your share (xS = 1), assuming
    everyone else's usage stays where it is."""
    if norm <= 0 or eff <= norm:
        return 0.0
    if eff >= 1 or norm >= 1:
        return halflife_h / 24 * math.log2(eff / norm)
    return halflife_h / 24 * math.log2((1 - norm) * eff / (norm * (1 - eff)))


def build_labs(gpuh, clu, acct_now, ours, our_kwh, rate, our_usd=None, usd_per_gpuh=0.0):
    """One row per lab (an account that has users): 30-day GPU-hours from sreport, the
    members' median fairshare and the account's usage multiple from sshare, and GPUs in
    use right now from squeue. Worst (most GPU-hours) first."""
    fs_by = {}
    for acct, _, fs, _ in clu["users"]:
        fs_by.setdefault(acct, []).append(fs)
    fs_by.setdefault(ours, [])
    gpu_only = ours.endswith("_gpu")           # Alliance clusters split each lab into _cpu and _gpu
    labs = []
    for a, fss in fs_by.items():
        if gpu_only and not a.endswith("_gpu"):
            continue
        h = gpuh.get(a, 0.0) if gpuh is not None else None
        labs.append(dict(acct=a, gpuh=h, now=acct_now.get(a, 0), xs=clu["accts"].get(a),
                         fair=statistics.median(fss) if fss else None,
                         kwh=our_kwh if a == ours and our_kwh else (h or 0.0) * rate,
                         usd=our_usd if a == ours and our_usd else (h or 0.0) * usd_per_gpuh))
    if gpuh is not None:
        labs.sort(key=lambda r: (-r["gpuh"], r["fair"] if r["fair"] is not None else 1.0, r["acct"]))
    else:
        labs.sort(key=lambda r: (-(r["xs"] or 0.0), r["acct"]))
    return labs


def parse_ts(s):
    try:
        return datetime.strptime(s.strip(), ISO).timestamp()
    except ValueError:
        return None


def backfill(slurm, account, user, board, halflife_h, days=60, step_h=1.0):
    """Rebuild usage history from the lab's job records.

    Slurm's fairshare usage is billing-seconds decayed with PriorityDecayHalfLife, so
    replaying every lab job reproduces each user's share of the lab (xS) over time.
    The fairshare factor itself cannot be rebuilt: it depends on every account on the
    cluster and Slurm stores no history of it. sacct's Priority column (priority when
    a job started) gives sparse points for the priority chart.
    Returns (samples, message).
    """
    norms = {r["user"]: r["norm"] for r in board}
    if not norms.get(user):
        raise RuntimeError(f"{user} not found in {account}'s sshare")
    hl = 168.0 if halflife_h is None else halflife_h      # Slurm's default is 7 days
    t_end = time.time()
    t_plot = t_end - days * 86400
    t0 = t_plot - (4 * hl if hl else 0) * 3600              # warm-up so the decayed sums settle
    cmd = ["-a", "-X", "-n", "-P", "-A", account, "-o"]
    try:
        out, has_prio = sacct_range(slurm, cmd + [SACCT_FMT], t0, timeout=900), True
    except RuntimeError as e:                               # older sacct without Priority
        if "too wide" in str(e).lower():
            raise
        out, has_prio = sacct_range(slurm, cmd + [SACCT_FMT.rsplit(",", 1)[0]], t0, timeout=900), False
    rows = {}
    for line in out.splitlines():                           # windows overlap: one row per job
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 5:
            rows[f[0]] = f

    step = step_h * 3600
    n = int((t_end - t0) / step) + 1
    contrib, prio_pts, njobs = {}, [], 0
    for f in rows.values():
        s0 = parse_ts(f[2])
        if s0 is None:
            continue                                        # never started
        if has_prio and len(f) > 5 and f[1] == user and s0 >= t_plot and fnum(f[5]) > 0:
            prio_pts.append({"ts": round(s0), "prio": round(fnum(f[5])), "bf": 1})
        e = parse_ts(f[3]) or t_end                         # still running
        s, e = max(s0, t0), min(e, t_end)
        if e <= s:
            continue
        m = re.search(r"billing=(\d+)", f[4]) or re.search(r"cpu=(\d+)", f[4])
        bill = float(m.group(1)) if m else 1.0
        c = contrib.setdefault(f[1], [0.0] * n)
        k = int((s - t0) // step)
        while k < n and t0 + k * step < e:
            lo, hi = max(s, t0 + k * step), min(e, t0 + (k + 1) * step)
            c[k] += bill * max(0.0, hi - lo)
            k += 1
        njobs += 1
    if not contrib:
        raise RuntimeError(f"sacct returned no jobs for {account}")
    if set(contrib) <= {user} and any(r["eff"] > 0 for r in board if r["user"] != user):
        raise RuntimeError("sacct only shows your own jobs here, so lab totals can't be rebuilt")

    d = 0.5 ** (step_h / hl) if hl else 1.0
    usage = {}
    for u, c in contrib.items():
        acc, arr = 0.0, [0.0] * n
        for k in range(n):
            acc = acc * d + c[k]
            arr[k] = acc
        usage[u] = arr
    total = [sum(arr[k] for arr in usage.values()) for k in range(n)]
    mine = usage.get(user, [0.0] * n)
    samples = []
    for k in range(n):
        t = min(t_end, t0 + (k + 1) * step)
        if t >= t_plot and total[k] > 0:
            samples.append({"ts": round(t), "xs": round(mine[k] / total[k] / norms[user], 4), "bf": 1})
    if len(prio_pts) >= 3:
        samples += prio_pts
    rebuilt = mine[-1] / total[-1] / norms[user] if total[-1] else 0.0
    current = next(r["xs"] for r in board if r["user"] == user)
    msg = (f"usage rebuilt from {njobs:,} jobs over {days}d: "
           f"xS now {rebuilt:.2f} vs sshare {current:.2f}")
    return samples, msg


def submit_hint(samples, snap):
    """Banner text: warn when this hour is historically bad for a GPU type."""
    samples = [s for s in samples if "types" in s]          # live samples only
    span = samples[-1]["ts"] - samples[0]["ts"] if samples else 0
    if span < 3 * 86400:
        return E("🧠 ") + f"Learning queue patterns ({span / 86400:.1f}/3 days)", "grey30"
    hour, best = datetime.now().hour, None
    for t in snap["type_order"]:
        by_hour, allw = {}, []
        for s in samples:
            v = s.get("types", {}).get(t)
            if not v:
                continue
            w = 72.0 if v.get("w") is None else min(v["w"], 72.0)
            by_hour.setdefault(datetime.fromtimestamp(s["ts"]).hour, []).append(w)
            allw.append(w)
        cur = by_hour.get(hour, [])
        if len(cur) < 5:
            continue
        m, overall = statistics.median(cur), statistics.median(allw)
        if m >= max(4.0, 1.5 * overall):
            return E("🚫 ") + f"Avoid now: {hour_range(hour)} {pretty_type(t)} backlog (median ~{m:.0f}h)", "#c0392b"
        meds = {h: statistics.median(v) for h, v in by_hour.items() if len(v) >= 5}
        if best is None and meds:
            bh = min(meds, key=meds.get)
            best = (E("✅ ") + f"{pretty_type(t)} now: median wait ~{fmt_hours(m)} at this hour, "
                    f"quietest {hour_range(bh)} (~{fmt_hours(meds[bh])})")
    return (best, "#2e6b34") if best else ("Not enough samples for this hour yet", "grey30")


# --------------------------------------------------------------- braille chart
DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def line_pts(x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def fmt_big(v, d=0):
    if abs(v) >= 1e6:
        return f"{v / 1e6:.{max(1, d)}f}M"
    return f"{v / 1e3:.{d}f}k" if abs(v) >= 1e3 else f"{v:.{d}f}"


def nice_labels(vals, big=False):
    """Fewest decimals that keep every tick label distinct."""
    labs = []
    for d in range(7):
        labs = [fmt_big(v, d) if big else f"{v:.{d}f}" for v in vals]
        if len(set(labs)) == len(labs):
            break
    return [("0" if lab.strip("-0.") == "" and not big else lab) for lab in labs]


class BrailleChart:
    """Line chart drawn with braille dots; sizes itself to the box rich gives it."""

    def __init__(self, ts, ys, color, ref=None, big=False, gap_s=3 * 3600, link=None):
        self.ts, self.ys, self.color, self.ref, self.big, self.gap_s = ts, ys, color, ref, big, gap_s
        self.link = link or [True] * len(ts)    # False: draw as a dot, don't join to the previous point
        self._cache = None

    def __rich_console__(self, console, options):
        size = (options.max_width, options.height or 8)
        if not self._cache or self._cache[0] != size:
            self._cache = (size, self._render(*size))
        yield from self._cache[1]

    def _render(self, W, H):
        if len(self.ts) < 2:
            pad = [Text("") for _ in range(max(0, H // 2 - 1))]
            return pad + [Text("collecting history (keep slurmdash running)", style=DIM,
                               justify="center", no_wrap=True, overflow="ellipsis")]
        now = time.time()
        xs = [(t - now) / 3600 for t in self.ts]
        lo, hi = min(self.ys), max(self.ys)
        # only show the lab reference line when it doesn't squash your own series flat
        ref = self.ref if self.ref is not None and lo - (hi - lo) <= self.ref <= hi + (hi - lo) else None
        if ref is not None:
            lo, hi = min(lo, ref), max(hi, ref)
        pad = (hi - lo) * 0.06 or (abs(hi) * 0.1 or 1.0)
        lo, hi = (lo - pad if lo - pad >= 0 or lo < 0 else 0.0), hi + pad   # no negative axis for >=0 data
        show_unit = H >= 6
        ph = max(2, H - (2 if show_unit else 1))
        nt = min(5, ph)
        tick_rows = sorted({round(i * (ph - 1) / (nt - 1)) for i in range(nt)})
        labels = dict(zip(tick_rows, nice_labels(
            [hi - (r * 4 + 1.5) / (ph * 4 - 1) * (hi - lo) for r in tick_rows], self.big)))
        lw = max(len(s) for s in labels.values())
        pw = max(4, W - lw - 1)
        PX, PY = pw * 2, ph * 4
        x0, x1 = min(xs), max(max(xs), 0.0)
        if x1 - x0 < 1e-9:
            x0 = x1 - 1
        data = [[0] * pw for _ in range(ph)]
        refg = [[0] * pw for _ in range(ph)]

        def px_of(x):
            return min(PX - 1, max(0, int((x - x0) / (x1 - x0) * (PX - 1) + 0.5)))

        def py_of(y):
            return min(PY - 1, max(0, int((hi - y) / (hi - lo) * (PY - 1) + 0.5)))

        def dot(g, px, py):
            g[py // 4][px // 2] |= DOTS[py % 4][px % 2]

        cols = {}                               # pixel column -> [min, max, first, last, t_first, t_last, link]
        for t, x, y, lk in zip(self.ts, xs, self.ys, self.link):
            px, py = px_of(x), py_of(y)
            c = cols.get(px)
            if c is None:
                cols[px] = [py, py, py, py, t, t, lk]
            elif lk:
                c[0], c[1], c[3], c[5] = min(c[0], py), max(c[1], py), py, t
            else:
                dot(data, px, py)
        prev = None
        for px in sorted(cols):
            mn, mx, first, last, tf, tl, lk = cols[px]
            for py in range(mn, mx + 1):
                dot(data, px, py)
            if prev and lk and tf - prev[2] <= self.gap_s:
                for qx, qy in line_pts(prev[0], prev[1], px, first):
                    dot(data, qx, qy)
            prev = (px, last, tl)
        if ref is not None:
            ry = py_of(ref)
            for px in range(0, PX, 3):
                dot(refg, px, ry)

        out = []
        for r in range(ph):
            line = Text(no_wrap=True, overflow="crop")
            line.append(labels.get(r, "").rjust(lw), style=MUTED)
            line.append("┤" if r in labels else "│", style=FRAME)
            buf, cur = [], None
            for c in range(pw):
                if data[r][c]:
                    ch, st = chr(0x2800 + data[r][c]), self.color
                elif refg[r][c]:
                    ch, st = chr(0x2800 + refg[r][c]), REFLINE
                else:
                    ch, st = " ", ""
                if st != cur and buf:
                    line.append("".join(buf), style=cur or None)
                    buf = []
                cur = st
                buf.append(ch)
            line.append("".join(buf), style=cur or None)
            out.append(line)

        span = x1 - x0
        unit, div, word = (("m", 1 / 60, "minutes") if span <= 2 else
                           ("h", 1.0, "hours") if span <= 72 else ("d", 24.0, "days"))
        nx = max(2, min(6, pw // 12))
        labs = [lab + unit for lab in nice_labels([(x0 + span * i / (nx - 1)) / div for i in range(nx)])]
        if abs(x1) < max(span * 0.02, 1 / 60):
            labs[-1] = "now"
        axis = [" "] * pw
        for i, lab in enumerate(labs):
            c = round(i * (pw - 1) / (nx - 1))
            s = min(max(0, c - len(lab) // 2), pw - len(lab))
            if all(ch == " " for ch in axis[max(0, s - 1):s + len(lab) + 1]):
                axis[s:s + len(lab)] = list(lab)
        out.append(Text(" " * (lw + 1) + "".join(axis), style=MUTED, no_wrap=True, overflow="crop"))
        if show_unit:
            short = now - self.ts[0] < 86400
            out.append(Text(f"{word} ago" + ("  ·  history grows while slurmdash runs" if short else ""),
                            style=DIM, justify="center", no_wrap=True, overflow="ellipsis"))
        return out


# ----------------------------------------------------------------- panels
def title(label, color, extra=""):
    return Text.assemble((label, f"bold {color}"), (extra, "grey62"))


def p_header(snap, state, hint):
    left = Text.assemble((E("🦥 ") + "slurmdash ", f"bold {CYAN}"), (state.user, "bold"), (" @ ", DIM),
                         (snap["account"] if snap else "…", ""))
    msg, bg = hint
    mid = Align.center(Text(msg, style="bold white", no_wrap=True, overflow="ellipsis"), style=f"on {bg}")
    age = time.time() - state.ok_ts if state.ok_ts else None
    if age is None:
        dot = (" ● error ", f"bold black on {RED}") if state.err else (" ● loading ", "bold black on grey50")
    elif age > 3 * state.refresh:
        dot = (f" ● stale {fmt_hours(age / 3600)} ", f"bold black on {RED if state.err else YELLOW}")
    else:
        dot = (" ● live ", f"bold black on {GREEN}")
    right = Text.assemble((datetime.now().strftime("%a %H:%M:%S") + "  ", "bold"), dot)
    g = Table.grid(expand=True, padding=(0, 2))
    g.add_column(ratio=2, no_wrap=True)
    g.add_column(ratio=5)
    g.add_column(ratio=2, justify="right", no_wrap=True)
    g.add_row(left, mid, right)
    return Panel(g, border_style=BLUE, padding=(0, 1))


def p_fairshare(snap, bar_w=14):
    f, a = snap["fs"], snap["acct"]
    if f["missing"]:
        body = Text(f"{snap['user']} has no association in {snap['account']} (try --account)", style=RED)
        return Panel(body, title=title(E("🥧 ") + "FAIRSHARE", YELLOW), border_style=YELLOW)
    g = Table.grid(padding=(0, 1))
    g.add_column(style=MUTED, no_wrap=True)
    g.add_column(no_wrap=True)
    n = int(round(min(1.0, f["factor"]) * bar_w))
    g.add_row("factor", Text.assemble((f"{f['factor']:.4f} ", f"bold {YELLOW}"),
                                      ("█" * n, OLIVE), ("█" * (bar_w - n), EMPTY)))
    g.add_row("usage", Text(f"{f['xs']:.2f}x your share{E(hog(f['xs']))}", style=xs_style(f["xs"])))
    g.add_row("EffUsage", f"{f['eff'] * 100:.2f}% of lab")
    lfs = f["levelfs"]
    g.add_row("LevelFS", "-" if lfs is None else ("∞ (no usage)" if math.isinf(lfs) else f"{lfs:.2f}"))
    g.add_row("lab acct", Text(f"{a['xs']:.1f}x share · {a['norm'] * 100:.1f}% of cluster{E(hog(a['xs']))}",
                               style=xs_style(a["xs"])))
    hl = snap["halflife_h"] or 168.0
    d_me, d_lab = redemption_days(f["eff"], f["norm"], hl), redemption_days(a["eff"], a["norm"], hl)
    if d_me <= 0:
        redeem = Text("you're within your share" + E(" 😇"), style=GREEN)
    else:
        back = datetime.now() + timedelta(days=d_me)
        redeem = Text.assemble(("stop submitting and you're back to 1.0x in ", MUTED),
                               (f"{d_me:.1f} d", f"bold {YELLOW}"), (f" ({back:%b} {back.day})", MUTED),
                               (f" · lab {d_lab:.1f} d" if d_lab > 0 else "", MUTED), (E(" 🛌"), ""))
    g.add_row("redeem", redeem)
    b = (snap["boards"] or {}).get("month")
    if b:
        g.add_row("AWS bill", Text.assemble((fmt_usd(b["usd_by"].get(snap["user"], 0.0)), f"bold {RED}"),
                                            (" you · ", MUTED), (fmt_usd(b["usd"]), f"bold {RED}"),
                                            (" lab, past 30 d at on-demand prices", MUTED), (E(" 💸"), "")))
    else:
        g.add_row("AWS bill", Text("waiting for sacct…", style=DIM))
    return Panel(g, title=title(E("🥧 ") + "FAIRSHARE", YELLOW), border_style=YELLOW)


def hog(xs):
    return " 🐷🐷" if xs > 2 else (" 🐷" if xs > 1 else (" 😇" if xs <= 0.5 else ""))


def p_priority(snap, width):
    p = snap["prio"]
    head = Text(fmt_int(p["total"]), style=f"bold {CYAN}")
    head.append("   new-job estimate (nothing pending)" if p["est"] else f"   job {p['job']}", style=DIM)
    if p["max"]:
        head.append(f" · top pending {fmt_k(p['max'])}", style=DIM)
    comps = [(n, p.get(n) or 0.0, c) for n, c in (("fair", CYAN), ("age", GREEN), ("part", YELLOW),
                                                  ("qos", MAGENTA), ("tres", VIOLET), ("size", MUTED))]
    shown = [c for c in comps if c[1] > 0] or comps[:1]
    bw = max(10, width)
    filled = max(1, int(round(bw * min(1.0, p["total"] / p["max"])))) if p["max"] else bw
    bar, used, tot = Text(no_wrap=True), 0, sum(v for _, v, _ in shown) or 1.0
    for i, (_, v, c) in enumerate(shown):
        n = filled - used if i == len(shown) - 1 else int(round(filled * v / tot))
        n = max(0, min(n, filled - used))
        bar.append("█" * n, style=c)
        used += n
    bar.append("█" * (bw - used), style=EMPTY)
    legend = Table.grid(expand=True)
    for _ in shown:
        legend.add_column(ratio=1, no_wrap=True)
    if p["breakdown"]:
        legend.add_row(*[Text.assemble(("■", c), (f"{n} {fmt_int(v)}", c)) for n, v, c in shown])
    else:
        legend.add_row(Text("sprio breakdown unavailable for this job", style=DIM))
    return Panel(Group(head, bar, legend), title=title(E("🎫 ") + "PRIORITY", CYAN), border_style=CYAN)


def fmt_gpuh(h):
    return f"{h:,.0f}" if h >= 10 else f"{h:.1f}"


def fmt_kwh(k):
    return f"{k / 1000:.1f} MWh" if k >= 1000 else f"{k:.0f} kWh"


def fair_style(fs):
    return RED if fs < 0.25 else (YELLOW if fs < 0.5 else GREEN)


def p_compute(snap, key, label, width=40):
    """Top ten lab users by GPU-hours in one window, with each user's fairshare factor,
    usage multiple (xS) and share of the lab's fairshare usage (eff%), all from sshare."""
    b, me, note = (snap["boards"] or {}).get(key), snap["user"], snap["boards_note"]
    if b is None or (note and "own jobs" in note):
        body = Text(note or "querying sacct…", style=YELLOW if note else DIM)
        return Panel(body, title=title(label, MUTED), border_style=FRAME)
    rows, total = b["rows"], b["total"]
    share = {r["user"]: r for r in snap["board"]}
    compact = width < 28                       # very narrow: short numbers
    # podium + user (8 chars) + GPU-h always; the rest only if they fit, most important first
    avail, cols = width - (2 + 1 + 8 + 1 + 5), []
    for name, w in (("fair", 4 if compact else 6), ("xS", 5), ("eff%", 4), ("#", 2)):
        if avail >= w + 1:
            cols.append(name)
            avail -= w + 1
    t = Table(box=None, expand=True, padding=(0, 0, 0, 1), pad_edge=False, show_edge=False,
              header_style=f"bold {MUTED}")
    if "#" in cols:
        t.add_column("#", justify="right", width=2, no_wrap=True)
    t.add_column("", width=2, no_wrap=True)
    t.add_column("user", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("GPU-h", justify="right", no_wrap=True)
    for name in ("fair", "xS", "eff%"):
        if name in cols:
            t.add_column(name, justify="right", no_wrap=True)
    for i, (u, h) in enumerate(rows[:10]):
        hl = f"bold {CYAN}" if u == me else ""
        r = share.get(u)
        cells = [Text(str(i + 1), style=hl or DIM)] if "#" in cols else []
        cells += [E(PODIUM[i]) if i < 3 else "", Text(u, style=hl or None),
                  Text(f"{h / 1000:.1f}k" if compact and h >= 1000 else fmt_gpuh(h),
                       style=hl or (f"bold {RED}" if i == 0 else None))]
        if "fair" in cols:
            fz = "-" if r is None else (f"{r['fs']:.3f}".lstrip("0") if compact else f"{r['fs']:.4f}")
            cells.append(Text(fz, style=hl or (DIM if r is None else fair_style(r["fs"]))))
        if "xS" in cols:
            xz = "-" if r is None else (f"{r['xs']:.2f}x" if r["xs"] < 10 else f"{r['xs']:.0f}x")
            cells.append(Text(xz, style=hl or (DIM if r is None else xs_style(r["xs"]))))
        if "eff%" in cols:
            ez = "-" if r is None else (f"{r['eff'] * 100:.1f}" if r["eff"] < 0.9995 else "100")
            cells.append(Text(ez, style=hl or DIM))
        t.add_row(*cells)
    if not rows:
        t.add_row(*([""] if "#" in cols else []), "", Text("no GPU jobs", style=DIM), "",
                  *[""] * sum(c in cols for c in ("fair", "xS", "eff%")))
    pos = next((i for i, (u, _) in enumerate(rows) if u == me), None)
    you = f"  you #{pos + 1}" if pos is not None else "  you: 0 GPU-h"
    sub = Text(f"lab {fmt_gpuh(total)} GPU-h{E(' ⚡')} ≈{fmt_kwh(b['kwh'])}" + (" (stale)" if note else ""),
               style=DIM)
    return Panel(t, title=title(label, MUTED, you), subtitle=sub, border_style=FRAME)


def p_labs(snap, height):
    """Cluster-wide lab ranking: the heaviest GPU users, plus our lab wherever it lands."""
    labs, ours, note = snap["labs"], snap["account"], snap["labs_note"]
    head = E("🏭 ") + ("LAB HALL OF SHAME" if FUN else "LAB LEADERBOARD")
    if not labs:
        return Panel(Text(note or "no labs found", style=DIM), title=title(head, RED), border_style=RED)
    mine = next((i for i, r in enumerate(labs) if r["acct"] == ours), None)
    avail = max(3, height - 3)
    idx = list(range(min(avail, len(labs))))
    if mine is not None and mine >= avail:
        idx = idx[:avail - 2] + [None, mine]   # None = "…" separator row
    t = Table(box=None, expand=True, padding=(0, 0, 0, 1), pad_edge=False, show_edge=False,
              header_style=f"bold {MUTED}")
    t.add_column("#", justify="right", width=3, no_wrap=True)
    t.add_column("", width=2, no_wrap=True)
    t.add_column("lab", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("GPU-h", justify="right", no_wrap=True)
    t.add_column(E("⚡") + "MWh", justify="right", no_wrap=True)
    t.add_column(E("💸") + "AWS", justify="right", no_wrap=True)
    t.add_column("GPUs now", justify="right", no_wrap=True)
    t.add_column("fair", justify="right", no_wrap=True)
    t.add_column("xS", justify="right", no_wrap=True)
    for i in idx:
        if i is None:
            t.add_row("", "", Text("…", style=DIM), "", "", "", "", "", "")
            continue
        r, us = labs[i], labs[i]["acct"] == ours
        hl = f"bold {CYAN}" if us else ""
        t.add_row(Text(str(i + 1), style=hl or DIM), E(PODIUM[i]) if i < 3 else "",
                  Text(r["acct"] + (E(" 👈") if us else ""), style=hl or None),
                  Text("-" if r["gpuh"] is None else fmt_gpuh(r["gpuh"]),
                       style=hl or (f"bold {RED}" if i == 0 else None)),
                  Text(f"{r['kwh'] / 1000:.1f}" if r["kwh"] else "-", style=hl or DIM),
                  Text(fmt_usd(r["usd"]) if r["usd"] else "-", style=hl or DIM),
                  Text(str(r["now"]), style=hl or None),
                  Text("-" if r["fair"] is None else f"{r['fair']:.3f}",
                       style=hl or (DIM if r["fair"] is None else fair_style(r["fair"]))),
                  Text("-" if r["xs"] is None else f"{r['xs']:.1f}x",
                       style=hl or (DIM if r["xs"] is None else xs_style(r["xs"]))))
    extra = f"  past 30 d · us #{mine + 1} of {len(labs)}" if mine is not None else "  past 30 d"
    if note:
        extra += f" · {note}"
    sub = None
    if mine is not None and labs[mine]["kwh"]:
        k = labs[mine]["kwh"]
        homes = k / HOME_KWH_MONTH
        sub = Text(f"us: ≈{fmt_kwh(k)}" + (f" ⚡ ≈ {homes:.0f} Ontario homes for a month 🏠 ≈ {k / KETTLE_KWH:,.0f} kettles ☕"
                                         if FUN else " at full GPU power"), style=DIM)
    return Panel(t, title=title(head, RED, extra), subtitle=sub, border_style=RED)


def p_gpu(t, T, width=100):
    compact = width < 60                       # side-by-side panels get a shorter wait note
    gap = "  " if compact else "   "
    g = Table.grid(padding=(0, 2))
    g.add_column(style=MUTED, no_wrap=True)
    g.add_column(no_wrap=True)
    g.add_row("slices" if is_mig(t) else "GPUs", Text.assemble((f"{T['free']} free{'' if T['free'] else E(' 😭')}",
                                     f"bold {GREEN if T['free'] else RED}"),
                                    f"{gap}{T['used']} used{gap}", (f"{T['total']} total", "bold")))
    g.add_row("dead", Text(f"{T['dead']} on {T['dead_nodes']} nodes{E(' 💀') if T['dead'] else ''}",
                           style=YELLOW if T["dead"] else DIM))
    if T["resv"]:
        g.add_row("resv", Text(f"{T['resv']} idle in reservations", style=MUTED))
    g.add_row("pending", f"{T['pending']:,} jobs")
    g.add_row("ahead", Text.assemble((f"{T['ahead']:,} jobs", RED if T["ahead"] else GREEN),
                                     (("  (new job)" if compact else "  (vs a new job)") if T["est"] else "", DIM)))
    wt, wc = wait_text(T["wait_h"])
    note = ""
    if T["wait_h"] and compact:
        note = f"  ~{fmt_hours(T['wait_h'])}, {T['rate']:.0f}/h" if T["rate"] else f"  ≤{fmt_hours(T['wait_h'])}"
    elif T["wait_h"]:
        note = (f"   ~{fmt_hours(T['wait_h'])} at {T['rate']:.0f} GPU starts/h, ignores backfill"
                if T["rate"] else f"   ≤{fmt_hours(T['wait_h'])} from time limits")
    mood = "👴" if (T["wait_h"] or 0) >= 24 else {GREEN: "🏃", YELLOW: "☕", RED: "🐌"}.get(wc, "")
    g.add_row("wait", Text.assemble((wt + E(" " + mood if mood else ""), f"bold {wc}"), (note, DIM)))
    return Panel(g, title=title(E(gpu_emoji(t) + " ") + pretty_type(t), GREEN), border_style=GREEN)


def p_achievements(snap):
    sh, me = snap["shame"], snap["user"]
    head = E("🏆 ") + "ACHIEVEMENTS"
    if sh is None:
        return Panel(Text(snap["boards_note"] or "querying sacct…", style=DIM), title=title(head, YELLOW),
                     border_style=YELLOW)
    g = Table.grid(padding=(0, 1), expand=True)
    g.add_column(no_wrap=True)
    g.add_column(no_wrap=True, style="bold")
    g.add_column(no_wrap=True, overflow="ellipsis")
    g.add_column(ratio=1, no_wrap=True, overflow="ellipsis", style=DIM)
    for icon, name, user, detail in sh["badges"]:
        if user is None:
            g.add_row(E(icon), name, Text("nobody", style=DIM), "a wholesome week")
        else:
            g.add_row(E(icon), name, Text(user + (" (you!)" if user == me else ""),
                                          style=f"bold {CYAN}" if user == me else YELLOW), detail)
    return Panel(g, title=title(head, YELLOW, f"  past {SHAME_DAYS} d"), border_style=YELLOW)


def p_wall(snap, height):
    sh, me = snap["shame"], snap["user"]
    head = E("💥 ") + "WALL OF FAILURE"
    if sh is None:
        return Panel(Text(snap["boards_note"] or "querying sacct…", style=DIM), title=title(head, MAGENTA),
                     border_style=MAGENTA)
    t = Table(box=None, expand=True, padding=(0, 0, 0, 1), pad_edge=False, show_edge=False,
              header_style=f"bold {MUTED}")
    t.add_column("#", justify="right", width=2, no_wrap=True)
    t.add_column("", width=2, no_wrap=True)
    t.add_column("user", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("fails", justify="right", no_wrap=True)
    t.add_column("GPU-h", justify="right", no_wrap=True)
    t.add_column("AWS", justify="right", no_wrap=True)
    t.add_column("mostly", no_wrap=True)
    for i, (u, w) in enumerate(sh["wall"][:max(1, height - 3)]):
        hl = f"bold {CYAN}" if u == me else ""
        t.add_row(Text(str(i + 1), style=hl or DIM), E("🤡") if i == 0 else "", Text(u, style=hl or None),
                  Text(str(w["n"]), style=hl or None), Text(fmt_gpuh(w["gpuh"]), style=hl or f"bold {MAGENTA}"),
                  Text(fmt_usd(w["usd"]), style=hl or DIM), Text(w["causes"].most_common(1)[0][0], style=hl or MUTED))
    if not sh["wall"]:
        t.add_row("", "", Text("no crashes this week" + E(" 🧘"), style=GREEN), "", "", "", "")
    sub = Text(f"lab wasted {fmt_gpuh(sh['wasted_gpuh'])} GPU-h ≈ {fmt_usd(sh['wasted_usd'])}" + E(" 🔥"),
               style=DIM) if sh["wall"] else None
    return Panel(t, title=title(head, MAGENTA, f"  past {SHAME_DAYS} d"), subtitle=sub, border_style=MAGENTA)


def group_running(jobs):
    """Your running GPU jobs, with each job array collapsed into one row. Biggest first."""
    groups = {}
    for j in jobs:
        base, _, idx = j["id"].partition("_")
        g = groups.setdefault(base if idx else j["id"], dict(key=base if idx else j["id"], idx=[], jobs=[]))
        g["jobs"].append(j)
        if idx:
            g["idx"].append(idx)
    out = []
    for g in groups.values():
        js = sorted(g["jobs"], key=lambda j: parse_hours(j["used"]) or 0.0)
        nums = sorted(int(i) for i in g["idx"] if i.isdigit())
        if len(nums) > 1:
            label = f"{g['key']}_[{nums[0]}-{nums[-1]}]"
        elif g["idx"]:
            label = f"{g['key']}_{g['idx'][0]}"
        else:
            label = g["key"]
        nodes = sorted({j["reason"] for j in js})
        used = js[0]["used"] if js[0]["used"] == js[-1]["used"] else f"{js[0]['used']}–{js[-1]['used']}"
        out.append(dict(label=label, name=js[0]["name"], gpus=sum(j["gpus"] for j in js), n=len(js),
                        type=js[0]["type"], where=nodes[0] if len(nodes) == 1 else f"{len(nodes)} nodes",
                        used=used, limit=js[0]["limit"], ids=[j["id"] for j in js]))
    out.sort(key=lambda g: (-g["gpus"], g["label"]))
    return out


def my_running(snap):
    return [j for j in snap["myjobs"] if j["st"] in ("R", "CF") and j["gpus"]]


def p_mygpus(snap, util, rows=8):
    run = my_running(snap)
    groups = group_running(run)
    n_gpus = sum(j["gpus"] for j in run)
    rows = max(1, rows)
    if len(groups) <= rows:
        shown, more_row = groups, False
    elif rows >= 3:                             # room for a "+N more" line
        shown, more_row = groups[:rows - 1], True
    else:                                       # tiny panel: biggest groups only, count in the title
        shown, more_row = groups[:rows], False
    hidden = groups[len(shown):]
    extra = f"  {n_gpus} GPU{'s' if n_gpus != 1 else ''} · {len(run)} job{'s' if len(run) != 1 else ''}"
    if hidden and not more_row:
        extra += f" · +{sum(g['n'] for g in hidden)} more"
    head = title(E("🔥 ") + "MY GPUs", MUTED, extra if run else "")
    if not run:
        return Panel(Text("no running GPU jobs" + E(" 🧊 the GPUs miss you"), style=DIM), title=head,
                     border_style=FRAME)
    body = Table.grid(padding=(0, 2), expand=True)
    for ratio in (None, 1, None, None, None, None):
        body.add_column(no_wrap=True, ratio=ratio, overflow="ellipsis")
    for g in shown:
        u = [x for i in g["ids"] for x in util.get(i, [])]
        ut = Text("")
        if u:
            avg = sum(x[0] for x in u) / len(u)
            ut = Text(f"{avg:.0f}% util" + E(" 🧟" if avg < 5 else ""),
                      style=GREEN if avg >= 50 else (YELLOW if avg >= 10 else RED))
        name = g["name"] + (f" ×{g['n']}" if g["n"] > 1 else "")
        body.add_row(Text(g["label"], style="bold"), name,
                     Text(f"{g['gpus']}×{short_type(g['type'])}", style=GREEN),
                     Text(g["where"], style=MUTED), f"{g['used']} / {g['limit']}", ut)
    if more_row:
        body.add_row(Text(f"… +{sum(g['n'] for g in hidden)} more jobs", style=DIM),
                     Text(f"{sum(g['gpus'] for g in hidden)} GPUs", style=DIM), "", "", "", "")
    return Panel(body, title=head, border_style=FRAME)


def p_qvy(snap, width):
    t = snap["prio"]["type"]
    if not t:
        return Panel(Text("no GPU partitions found", style=DIM), border_style=FRAME)
    T = snap["types"][t]
    bw = max(5, width - 4 - 10 - 8)
    mx = max(T["hist"]) or 1
    g = Table.grid(expand=True, padding=(0, 1))
    g.add_column(width=10, style=MUTED, no_wrap=True)
    g.add_column(ratio=1, no_wrap=True)
    g.add_column(justify="right", width=7, no_wrap=True)
    for lab, n, c in zip((">2x yours", "1.5–2x", "1–1.5x", "below you"), T["hist"], (RED, RED, RED, GREEN)):
        g.add_row(lab, Text("█" * (max(1, round(bw * n / mx)) if n else 0), style=c), f"{n:,}")
    sub = Text(f"your prio {fmt_int(T['my_prio'])}{' (new-job estimate)' if T['est'] else ''}", style=DIM)
    return Panel(g, title=title(E("🐌 ") + f"QUEUE vs YOU ({pretty_type(t)})", MUTED), subtitle=sub,
                 border_style=FRAME)


def start_text(s):
    try:
        dt = datetime.strptime(s, ISO)
    except ValueError:
        return Text("computing…", style=DIM)
    mins = (dt - datetime.now()).total_seconds() / 60
    if mins < 0:
        return Text("any moment", style=f"bold {GREEN}")
    when = dt.strftime("%H:%M") if mins < 20 * 60 else dt.strftime("%a %H:%M")
    rel = f"{mins:.0f}m" if mins < 90 else fmt_hours(mins / 60)
    return Text(f"{when} (in {rel})", style=f"bold {GREEN}" if mins < 120 else YELLOW)


def p_queue(snap, rows, width=120):
    jobs, types = snap["myjobs"], snap["type_order"]
    t = Table(box=None, expand=True, padding=(0, 0, 0, 1), pad_edge=False, show_edge=False,
              header_style=f"bold {MUTED}")
    ids, parts = [j["id"] for j in jobs] or [""], [short_part(j["part"], types) for j in jobs] or [""]
    t.add_column("job", no_wrap=True, min_width=max(len(x) for x in ids))        # never truncate ids,
    t.add_column("name", no_wrap=True, overflow="ellipsis", max_width=24 if width >= 100 else 14, min_width=6)
    t.add_column("part", no_wrap=True, min_width=max(len(x) for x in parts))      # partitions or prios
    t.add_column("st", no_wrap=True, min_width=2)
    t.add_column("prio", justify="right", no_wrap=True, min_width=9)
    t.add_column("start / runtime", no_wrap=True, overflow="ellipsis")
    t.add_column("why", ratio=1, no_wrap=True, overflow="ellipsis", min_width=4)
    run_gpu = [j for j in jobs if j["st"] in ("R", "CF") and j["gpus"]]
    rest = sorted((j for j in jobs if j not in run_gpu), key=lambda j: (j["st"] != "PD", -j["prio"]))
    rows = max(1, rows)                         # running GPU jobs are listed in MY GPUs instead
    shown = rest if len(rest) <= rows else rest[:rows - 1]
    for j in shown:
        if j["st"] == "PD":
            st, when, why = Text("PD", style=f"bold {YELLOW}"), start_text(j["start"]), Text(j["reason"], style=MUTED)
        else:
            st, when, why = (Text(j["st"], style=f"bold {GREEN}"),
                             Text(f"{j['used']} / {j['limit']}", style=GREEN), Text(j["reason"]))
        t.add_row(j["id"], j["name"], short_part(j["part"], types), st, fmt_int(j["prio"]), when, why)
    if len(rest) > len(shown):
        t.add_row(Text(f"… +{len(rest) - len(shown)} more", style=DIM), *[""] * 6)
    if not rest:
        t.add_row(Text("nothing pending" + (E(" 🎉") if run_gpu else ""), style=DIM), *[""] * 6)
    n_pd = sum(j["st"] == "PD" for j in jobs)
    extra = f"  {n_pd} pending · {len(jobs) - n_pd} running"
    if run_gpu:
        extra += f" ({len(run_gpu)} on GPUs, in MY GPUs)"
    return Panel(t, title=title(E("⏳ ") + "MY QUEUE", BLUE, extra), border_style=BLUE)


def make_charts(snap, samples):
    ts = [s["ts"] for s in samples]
    lab, clu = snap["lab"], snap["clu"]
    f3, f2 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}")
    panels = []
    for name, key, you, fmt, color in ((E("📉 ") + "FairShare", "fs", snap["fs"]["factor"], f3, GREEN),
                                        (E("📈 ") + "Usage (xshare)", "xs", snap["fs"]["xs"], f2, YELLOW),
                                        (E("🎫 ") + "Priority", "prio", snap["prio"]["total"], fmt_k, CYAN)):
        pts = [(t, s.get(key), not s.get("bf") or key != "prio")      # sacct priorities are dots
               for t, s in zip(ts, samples) if s.get(key) is not None]
        sub = [Text.assemble((f"{n} ", MUTED), (fmt(v[0]), "bold"), (f" [{fmt(v[1])}–{fmt(v[2])}]", DIM))
               for n, v in (("lab", lab.get(key)), ("clu", clu.get(key))) if v]
        chart = BrailleChart([p[0] for p in pts], [p[1] for p in pts], color,
                             ref=lab[key][0] if lab.get(key) else None, big=key == "prio",
                             link=[p[2] for p in pts])
        panels.append(Panel(chart, title=Text.assemble((name, "bold"), ("   you ", MUTED), (fmt(you), f"bold {color}")),
                            subtitle=Text("   ").join(sub) if sub else None,
                            border_style=color, padding=(0, 1)))
    return panels


# ------------------------------------------------------------------ layout
class State:
    def __init__(self, user, refresh):
        self.user, self.refresh = user, refresh
        self.snap = self.err = self.ok_ts = self.charts = None
        self.hint = ("Querying Slurm…", "grey30")
        self.note = None
        self.util = {}
        self.lock = threading.Lock()


SHAME_H = 8                                     # achievements + wall of failure row
MYGPU_MAX_ROWS = 8                              # job rows in MY GPUs before "+N more"


def build_layout(console, state):
    W, H = console.size
    with state.lock:
        snap, err, charts, util, hint = state.snap, state.err, state.charts, state.util, state.hint
    if state.note and hint[1] == "grey30":
        hint = (f"{hint[0]}  ·  {state.note}", hint[1])
    if err:
        hint = (f"Slurm query failed: {err}", "#7a5c00")
    root = Layout()
    if snap is None:
        body = Panel(Align.center(Text(err or "querying Slurm…", style=RED if err else DIM), vertical="middle"),
                     border_style=FRAME)
        root.split_column(Layout(p_header(None, state, hint), size=3), Layout(body))
        return root

    running_gpu = my_running(snap)
    nq = len(snap["myjobs"]) - len(running_gpu)
    q_h = max(min(max(nq, 1), 10) + (1 if nq > 10 else 0) + 3, 13)   # 13 fits the lab board
    n_groups = len(group_running(my_running(snap)))
    left_w = W * 3 // 5                         # 60%: room for the boards' fair / xS / eff% columns
    right_w = W - left_w
    types = snap["type_order"]

    def gpu_h(t):
        return 7 + bool(snap["types"][t]["resv"])

    per_row = max(1, min(len(types), right_w // 40)) if types else 1      # e.g. L40S | H100 side by side
    type_rows = [types[i:i + per_row] for i in range(0, len(types), per_row)]
    types_h = sum(max(map(gpu_h, r)) for r in type_rows)
    left_need = 9 + 5 + 13                     # fairshare, priority, top-ten boards
    stack = right_w < 110                      # too narrow for achievements and wall side by side
    shame_h = SHAME_H + 6 if stack else SHAME_H
    fixed_right = types_h + shame_h
    mg_need = min(max(n_groups, 1), MYGPU_MAX_ROWS) + 2
    # MY GPUs grows into the space the left column leaves, plus any spare rows the
    # charts don't need, but never past MYGPU_MAX_ROWS
    spare = max(0, H - 3 - q_h - max(left_need, fixed_right + 3) - 14)
    mg_h = max(3, min(mg_need, max(left_need - fixed_right, 3) + spare))
    top_h = max(left_need, fixed_right + mg_h)
    qvy_h = top_h - fixed_right - mg_h
    if qvy_h < 6 and H - 3 - q_h - top_h - 6 + qvy_h >= 12:
        top_h += 6 - qvy_h                     # room to spare: keep QUEUE vs YOU visible
        qvy_h = 6
    chart_h = H - 3 - q_h - top_h
    if chart_h < 10:                            # small terminal: squeeze the top instead
        chart_h = max(8, min(16, (H - 3 - q_h) // 3))
        top_h = max(20, H - 3 - q_h - chart_h)
        mg_h = max(3, min(mg_h, top_h - fixed_right))
        qvy_h = top_h - fixed_right - mg_h

    left, right = Layout(size=left_w), Layout()
    left.split_column(Layout(p_fairshare(snap), size=9),
                      Layout(p_priority(snap, left_w - 4), size=5),
                      Layout(name="boards"))
    bw = left_w // len(BOARDS) - 4
    left["boards"].split_row(*[Layout(p_compute(snap, key, label, bw)) for key, label, _ in BOARDS])
    kids = []
    for r in type_rows:
        row = Layout(size=max(map(gpu_h, r)))
        row.split_row(*[Layout(p_gpu(t, snap["types"][t], right_w // len(r) - 4)) for t in r])
        kids.append(row)
    shame = Layout(size=shame_h)
    if stack:
        shame.split_column(Layout(p_achievements(snap), size=6), Layout(p_wall(snap, SHAME_H)))
    else:
        shame.split_row(Layout(p_achievements(snap), ratio=4), Layout(p_wall(snap, SHAME_H), ratio=3))
    kids += [Layout(p_mygpus(snap, util, mg_h - 2), size=mg_h), shame]
    if qvy_h >= 6:                              # QUEUE vs YOU is the first thing to go when space is tight
        kids.append(Layout(p_qvy(snap, right_w - 4)))
    elif qvy_h > 0:                             # (never a zero-height region: rich draws it anyway)
        kids.append(Layout(Text(" "), size=qvy_h))
    right.split_column(*kids)
    top, middle, bottom = Layout(size=top_h), Layout(size=q_h), Layout(size=chart_h)
    top.split_row(left, right)
    middle.split_row(Layout(p_labs(snap, q_h), size=left_w), Layout(p_queue(snap, q_h - 3, right_w - 4)))
    bottom.split_row(*[Layout(c) for c in (charts or [])])
    root.split_column(Layout(p_header(snap, state, hint), size=3), top, middle, bottom)
    return root


def refresh_state(col, state, hist, args, util_due):
    try:
        snap = col.snapshot()
        hist.maybe_add(history_sample(snap), args.sample_every)
        util = state.util
        if args.gpu_util and util_due:
            util = {}
            for g in group_running(my_running(snap))[:MYGPU_MAX_ROWS]:   # one task per array, ≤8 sruns
                try:
                    util[g["ids"][0]] = gpu_util(col.slurm, g["ids"][0])
                except RuntimeError:
                    pass
        samples = hist.samples
        charts, hint = make_charts(snap, samples), submit_hint(samples, snap)
        with state.lock:
            state.snap, state.err, state.ok_ts, state.util = snap, None, time.time(), util
            state.charts, state.hint = charts, hint
    except Exception as e:                      # keep the UI alive whatever goes wrong
        with state.lock:
            state.err = f"{type(e).__name__}: {e}"[:200]


def run_backfill(col, state, hist, days):
    with state.lock:
        state.note, snap = "rebuilding usage history from sacct…", state.snap
    try:
        bf, note = backfill(col.slurm, col.account, col.user, snap["board"], snap["halflife_h"], days)
        hist.set_backfill(bf)
    except Exception as e:
        note = f"usage backfill failed ({e})"[:160]
    charts = make_charts(snap, hist.samples)
    with state.lock:
        state.note, state.charts = note, charts


def fetch_loop(col, state, hist, args, stop):
    last_util, backfilling = 0.0, False
    while not stop.is_set():
        due = time.time() - last_util >= 60
        refresh_state(col, state, hist, args, due)
        if due:
            last_util = time.time()
        if not backfilling and state.snap and args.backfill_days > 0 and not hist.has_backfill():
            backfilling = True                  # one attempt per launch, in the background
            threading.Thread(target=run_backfill, args=(col, state, hist, args.backfill_days),
                             daemon=True).start()
        stop.wait(args.refresh)


# -------------------------------------------------------------------- demo
class FakeSlurm:
    """A synthetic Killarney-shaped cluster that answers the same CLI calls (--demo)."""
    W_FS = 5_000_000

    def __init__(self, user):
        rng = random.Random(42)
        self.user, self.acct = user, "aip-demo"
        lab = [f"labmate{i:02d}" for i in range(1, 20)]
        effs = [0.0] * 11 + [0.0035, 0.012, 0.0155, 0.031, 0.0465, 0.0715, 0.091, 0.55]
        self.eff = dict(zip(lab, effs))
        self.eff[user] = 0.1885
        ranked = sorted(self.eff, key=lambda u: (self.eff[u], u))
        self.fs = {u: 0.0670 - 0.00023 * i for i, u in enumerate(ranked)}
        # 40 other labs of 10 users; heavier labs get lower fairshare
        self.heavy = {k: rng.uniform(0, 4.5) for k in range(40)}
        self.user_acct = {f"user{i:03d}": f"aip-lab{i % 40:02d}" for i in range(400)}
        self.others = {u: max(0.02, min(1.0, 1 - 0.23 * self.heavy[i % 40] + rng.uniform(-0.05, 0.05)))
                       for i, u in enumerate(self.user_acct)}
        # 168 nodes x 4 L40S (16 drained), 10 nodes x 8 H100 (2 down)
        self.nodes = [[f"kn{i + 1:03d}", "l40s", 4, 0 if i < 16 else 4,
                       "IDLE+DRAIN" if i < 16 else "ALLOCATED"] for i in range(168)]
        self.nodes += [[f"kn{169 + i:03d}", "h100", 8, 0 if i < 2 else 8,
                        "DOWN+NOT_RESPONDING" if i < 2 else "ALLOCATED"] for i in range(10)]
        for k, a in ((40, 3), (77, 3), (130, 3), (170, 5), (171, 6)):
            self.nodes[k][3], self.nodes[k][4] = a, "MIXED"
        limits = {1: 3, 2: 12, 3: 24, 4: 72, 5: 168}
        users, jid, self.jobs = list(self.others) + lab, 6_300_000, []
        for name, t, _tot, alloc, _st in self.nodes:
            while alloc > 0:
                g = min(alloc, rng.choice([1, 1, 1, 2, 4]))
                b = rng.choice([1, 1, 2, 2, 3, 4, 5])
                jid += 1
                self.jobs.append(dict(id=str(jid), user=rng.choice(users), t=t, part=f"gpubase_{t}_b{b}",
                                      st="R", prio=rng.uniform(1e5, 2e6), gpus=g, lim=limits[b],
                                      used=rng.uniform(0, limits[b] * 0.95), start=None, reason=name, name="train"))
                alloc -= g
        for t, ref, counts in (("l40s", 317_304, (388, 370, 345, 4117)),
                               ("h100", self.W_FS * self.fs[user], (60, 40, 31, 115))):
            for n, (lo, hi) in zip(counts, ((2 * ref + 1, 6 * ref), (1.5 * ref + 1, 2 * ref),
                                            (ref + 1, 1.5 * ref), (0.3 * ref, ref - 1))):
                for _ in range(n):
                    jid += 1
                    b = rng.choice([1, 2, 2, 3, 4])
                    self.jobs.append(dict(
                        id=str(jid), user=rng.choice(users), t=t, part=f"gpubase_{t}_b{b}", st="PD",
                        prio=rng.uniform(lo, hi), gpus=rng.choice([1] * 7 + [2, 2, 4]), lim=limits[b],
                        used=0, start=None, name="job",
                        reason=rng.choices(["Priority", "Resources", "Dependency", "QOSMaxGRESPerUser"],
                                           [80, 5, 8, 7])[0]))
        mine = [("6460811", "train-baseline", 317_304, "Priority", 53),
                ("6492988", "train-baseline-retry", 317_048, "Priority", None)]
        mine += [(str(6460812 + i), n, 316_946, "Dependency", None)
                 for i, n in enumerate(["eval-scores", "sweep-optuna", "sweep-25", "sweep-50", "sweep-100"])]
        for i, name, prio, reason, start in mine:
            self.jobs.append(dict(id=i, user=user, t="l40s", part="gpubase_l40s_b1", st="PD", prio=prio,
                                  gpus=1, lim=3, used=0, start=start, reason=reason, name=name))
        singles = [j for j in self.jobs if j["st"] == "R" and j["t"] == "l40s" and j["gpus"] == 1]
        singles[0].update(user=user, name="debug-notebook", part="gpubase_l40s_b1", lim=3, used=1.2)
        for k, j in enumerate(singles[1:23]):
            j.update(user=user, id=f"6544972_{k}", name="A_joint0", part="gpubase_l40s_b1", lim=0.67,
                     used=0.6 - 0.025 * k)
        for k, j in enumerate(singles[23:26]):
            j.update(user=user, id=str(6544828 + k), name=f"B_s2_L{k}", part="gpubase_l40s_b1", lim=0.75, used=0.24)
        self.jobs.append(dict(id="6543200", user=user, t="l40s", part="gpubase_l40s_b2", st="R", prio=5e5,
                              gpus=0, lim=8, used=1.93, start=None, reason="kn037", name="vscode-cpu"))

    def run(self, args, timeout=60):
        cmd = args[0]
        if cmd == "sshare":
            if "-U" in args:
                return f"{self.acct}|{self.user}|75400000\n"
            return self._sshare_lab() if "-A" in args else self._sshare_all()
        if cmd == "squeue" and "-O" in args:
            out = []
            for j in self.jobs:
                if j["user"] == self.user and j["st"] == "R":
                    base, _, task = j["id"].partition("_")
                    gres = f",gres/gpu={j['gpus']},gres/gpu:{j['t']}={j['gpus']}" if j["gpus"] else ""
                    out.append(f"{base:<20}{task or 'N/A':<20}cpu=12,mem=60G,node=1,billing=4{gres}")
            return "\n".join(out) + "\n"
        if cmd == "squeue":
            return self._squeue()
        if cmd == "sreport":
            return self._sreport()
        if cmd == "sprio":
            return self._sprio()
        if cmd == "scontrol" and "node" in args:
            return self._nodes()
        if cmd == "scontrol":
            return ("PriorityWeightAge = 10000\nPriorityWeightFairShare = 5000000\n"
                    "PriorityWeightJobSize = 0\nPriorityWeightPartition = 2500\n"
                    "PriorityWeightQOS = 0\nPriorityWeightTRES = CPU=0,GRES/gpu=36\n"
                    "PriorityDecayHalfLife   = 7-00:00:00\n")
        if cmd == "sacct":
            return self._sacct(args)
        if cmd == "srun":
            return "87, 31211, 46068\n"
        raise RuntimeError(f"demo has no answer for {cmd}")

    def _sshare_lab(self):
        j = 1 + 0.01 * math.sin(time.time() / 60)
        lines = [f" {self.acct}||1|0.014000|987654321|{0.045 * j:.6f}||0.31"]
        for u, e in self.eff.items():
            lfs = "inf" if e == 0 else f"{0.05 / e:.6f}"
            lines.append(f"  {self.acct}|{u}|1|0.050000|{int(e * 4e8)}|{e * j:.6f}|{self.fs[u]:.6f}|{lfs}")
        return "\n".join(lines) + "\n"

    def _sshare_all(self):
        lines = [f"{self.acct}|{u}|0.050000|{e:.6f}|{self.fs[u]:.6f}" for u, e in self.eff.items()]
        lines += [f"{self.user_acct[u]}|{u}|0.100000|{max(0.0, 1 - fs) * 0.12:.6f}|{fs:.6f}"
                  for u, fs in self.others.items()]
        lines += [f" {self.acct}||0.014000|0.045000|"]
        lines += [f" aip-lab{k:02d}||0.020000|{0.02 * h:.6f}|" for k, h in self.heavy.items()]
        return "\n".join(lines) + "\n"

    def _sreport(self):
        now, ours = time.time(), 0.0
        for row in self._sacct(["-o", "JobIDRaw,User,Start,End,AllocTRES"]).splitlines():
            f = row.split("|")
            s, e = parse_ts(f[2]), parse_ts(f[3]) or now
            ours += max(0.0, e - max(s, now - 30 * 86400)) / 3600
        lines = ["-" * 60, "Cluster/Account/User Utilization (demo)", "Usage reported in TRES Hours", "-" * 60,
                 "Cluster|Account|Login|Proper Name|TRES Name|Used", "killarney|root|||gres/gpu|480000",
                 f"killarney|{self.acct}|||gres/gpu|{ours:.0f}", f"killarney|{self.acct}|{self.user}||gres/gpu|900"]
        lines += [f"killarney|aip-lab{k:02d}|||gres/gpu|{h * 3600 + 200:.0f}" for k, h in self.heavy.items()]
        return "\n".join(lines) + "\n"

    def _squeue(self):
        now, out = datetime.now(), []
        for j in self.jobs:
            if j["st"] == "R":
                left, used = hms(j["lim"] - j["used"]), hms(j["used"])
                start, reason = (now - timedelta(hours=j["used"])).strftime(ISO), j["reason"]
            else:
                left, used, reason = hms(j["lim"]), "0:00", f"({j['reason']})"
                start = (now + timedelta(minutes=j["start"])).strftime(ISO) if j["start"] else "N/A"
            acct = self.acct if j["user"] in self.eff else self.user_acct.get(j["user"], "aip-misc")
            out.append("|".join([j["id"], j["user"], acct, j["part"], j["st"], str(int(j["prio"])), left,
                                 f"gres/gpu:{j['t']}:{j['gpus']}" if j["gpus"] else "N/A", "1", start,
                                 used, hms(j["lim"]),
                                 reason, j["name"]]))
        return "\n".join(out) + "\n"

    def _sacct(self, args):
        rng, now, out, jid = random.Random(3), time.time(), [], 5_000_000
        fields = args[args.index("-o") + 1].split(",")
        iso = lambda t: datetime.fromtimestamp(t).strftime(ISO)       # noqa: E731
        for u, e in self.eff.items():
            arrayish = 0.6 if u in ("labmate19", "labmate16") else 0.08
            for k in range(int(e * 3000) + rng.randint(1, 40)):
                s = now - rng.uniform(0, 90 * 86400) * (rng.random() ** (2 if u == self.user else 0.5))
                end = s + rng.uniform(0.2, 12) * 3600
                sub = s - rng.expovariate(1 / 7200) * (12 if rng.random() < 0.03 else 1)
                if rng.random() < (0.25 if u == "labmate17" else 0.03):          # a night owl
                    sub = min(datetime.fromtimestamp(sub).replace(hour=rng.randint(2, 4)).timestamp(), s - 60)
                state = "RUNNING" if end > now else rng.choices(
                    ["COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED by 4242"],
                    [70, 8 if u != "labmate18" else 30, 5, 9 if u != "labmate15" else 40, 7])[0]
                vals = dict(JobIDRaw=str(jid), JobID=f"{jid - k}_{k}" if rng.random() < arrayish else str(jid),
                            User=u, Submit=iso(sub), Start=iso(s), End="Unknown" if end > now else iso(end),
                            AllocTRES="billing=4,cpu=12,gres/gpu=1,mem=60G,node=1", State=state,
                            Priority=str(int(rng.uniform(2e5, 1.2e6))))
                out.append("|".join(vals[f] for f in fields))
                jid += 1
        return "\n".join(out) + "\n"

    def _sprio(self):
        fair = int(self.W_FS * self.fs[self.user])
        return "".join(f"{j['id']}|{int(j['prio'])}|{max(0, int(j['prio']) - fair - 2536)}|{fair}|0|2500|0|gres/gpu=36\n"
                       for j in self.jobs if j["user"] == self.user and j["st"] == "PD")

    def _nodes(self):
        out = []
        for name, t, tot, alloc, state in self.nodes:
            parts = ",".join(f"gpubase_{t}_b{b}" for b in range(1, 6))
            out.append(f"NodeName={name} Arch=x86_64 CoresPerSocket=32 Gres=gpu:{t}:{tot}(S:0-1) "
                       f"State={state} Partitions={parts} CfgTRES=cpu=64,mem=500G,gres/gpu={tot},gres/gpu:{t}={tot} "
                       f"AllocTRES=cpu={alloc * 12},mem={alloc * 100}G,gres/gpu={alloc},gres/gpu:{t}={alloc} "
                       f"Reason={'Not responding' if 'DOWN' in state else ''}")
        return "\n".join(out) + "\n"


def demo_history():
    """130 days of hourly synthetic samples. The H100 backlog peaks at the current
    hour so the banner has something to warn about."""
    rng, now, peak, out = random.Random(1), time.time(), datetime.now().hour, []

    def sig(x):
        return 1 / (1 + math.exp(-max(-50.0, min(50.0, x))))

    for i in range(130 * 24):
        t = now - (130 * 24 - i) * 3600
        d = i / 24
        fs = (0.09 + 0.27 * math.exp(-((d - 33) / 3) ** 2) + 0.13 * sig((d - 58) / 2) * sig((122 - d) / 2)
              - 0.03 * sig((d - 125) / 1.5) + rng.gauss(0, 0.002))
        xs = max(0.0, 2.6 * math.exp(-d / 20) * (1 + 0.4 * math.sin(d / 3)) + 3.7 * sig((d - 127) / 0.8)
                 + rng.gauss(0, 0.03))
        hd = abs(datetime.fromtimestamp(t).hour - peak)
        hd = min(hd, 24 - hd)
        types = {"l40s": dict(p=5000, a=1000, f=3, w=round(3 + 1.5 * math.exp(-hd ** 2 / 2) + rng.uniform(0, 1), 2)),
                 "h100": dict(p=240, a=130, f=5, w=round(1.5 + 35 * math.exp(-hd ** 2 / 2) + rng.uniform(0, 1), 2))}
        out.append(dict(ts=round(t), fs=fs, xs=xs, prio=round(5e6 * fs + 2500), types=types))
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live Slurm fairshare / priority / GPU dashboard.")
    ap.add_argument("mode", nargs="?", default="live", choices=["live", "collect", "backfill"],
                    help="'collect' records one history sample and exits (for cron/scrontab); "
                         "'backfill' rebuilds usage history from sacct and exits")
    ap.add_argument("-u", "--user", default=os.environ.get("USER", ""))
    ap.add_argument("-A", "--account", help="Slurm account, e.g. aip-yourpi (default: auto-detect)")
    ap.add_argument("--refresh", type=float, default=30,
                    help="seconds between squeue polls (default 30, minimum 10; be kind to slurmctld)")
    ap.add_argument("--sshare-every", type=float, default=120, help="seconds between sshare polls")
    ap.add_argument("--sample-every", type=float, default=300, help="seconds between history samples")
    ap.add_argument("--history", help="history file (default ~/.cache/slurmdash/history-USER.jsonl)")
    ap.add_argument("--keep-days", type=float, default=180, help="history retention")
    ap.add_argument("--backfill-days", type=int, default=60,
                    help="days of usage history to rebuild from sacct on first launch (0 disables)")
    ap.add_argument("--gpu-util", action="store_true",
                    help="poll nvidia-smi inside your running jobs every 60 s (srun --overlap)")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no Slurm needed")
    ap.add_argument("--serious", action="store_true", help="no emojis or jokes (for when your PI walks by)")
    args = ap.parse_args()
    global FUN
    FUN = not args.serious
    args.refresh = max(10.0, args.refresh)
    user = args.user or "you"

    if args.demo:
        slurm, hist = FakeSlurm(user), History(None)
        hist.samples = demo_history()
    else:
        slurm = Slurm()
        hist = History(args.history or f"~/.cache/slurmdash/history-{user}.jsonl", args.keep_days)
    col = Collector(slurm, user, args.account, args.sshare_every)

    if args.mode == "backfill":
        snap = col.snapshot()
        print("querying sacct (can take a minute for a busy lab)…", flush=True)
        bf, msg = backfill(slurm, col.account, user, snap["board"], snap["halflife_h"],
                           max(1, args.backfill_days))
        hist.set_backfill(bf)
        print(f"{msg} (decay half-life {fmt_hours(snap['halflife_h'] or 168)})"
              + (f"\nsaved to {hist.path}" if hist.path else ""))
        return

    if args.mode == "collect":
        snap = col.snapshot()
        added = hist.maybe_add(history_sample(snap), args.sample_every / 2)
        summary = ", ".join(f"{t} free={v['free']} pend={v['pending']} wait={fmt_hours(v['wait_h'])}"
                            for t, v in snap["types"].items())
        print(f"{datetime.now():%F %T} {'recorded' if added else 'skipped (recent sample exists)'}: "
              f"fs={snap['fs']['factor']:.4f} xs={snap['fs']['xs']:.2f} {summary}")
        return

    console, state = Console(), State(user, args.refresh)
    if args.once:
        col.sync_usage = True
        refresh_state(col, state, hist, args, util_due=True)
        console.print(build_layout(console, state))
        return

    stop = threading.Event()
    threading.Thread(target=fetch_loop, args=(col, state, hist, args, stop), daemon=True).start()
    try:
        with Live(build_layout(console, state), console=console, screen=True, auto_refresh=False) as live:
            while True:
                live.update(build_layout(console, state), refresh=True)
                time.sleep(1)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()