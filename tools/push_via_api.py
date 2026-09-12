#!/usr/bin/env python3
"""Push the tracked files of this repository through the GitHub REST API.

Why: on some networks plain `git push` over HTTPS is blocked or stalls on an
intermediate proxy, while `api.github.com` still works.  This script recreates
the current `git ls-files` snapshot as a single commit using the Git Data API,
so the repository can be published anyway.

    set GITHUB_TOKEN=ghp_xxx            (or a github_pat_ fine-grained token
                                         with Contents: Read and write)
    python tools/push_via_api.py --owner AaronShawn --repo blackhole-simulator

The commit is created with the given branch (default: main).  Existing history
is preserved: when the branch already exists the new commit is parented on it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"


def request(url: str, token: str, method: str = "GET", payload=None, proxy: str | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "blackhole-simulator-publisher")
    if data:
        req.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else urllib.request.ProxyHandler({})
    )
    try:
        with opener.open(req, timeout=120) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        print(f"  [!] {method} {url} -> {exc.code} {detail}", file=sys.stderr)
        raise


def tracked_files() -> list[str]:
    # core.quotePath=false keeps non-ASCII paths (e.g. 一键发布.bat) readable
    out = subprocess.run(["git", "-c", "core.quotePath=false", "ls-files"],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="Schwarzschild black hole simulator v1.0.0")
    ap.add_argument("--author-name", default=None)
    ap.add_argument("--author-email", default=None)
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    ap.add_argument("--orphan", action="store_true",
                    help="replace the branch with a single root commit (fresh repository)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set", file=sys.stderr)
        return 2

    files = tracked_files()
    print(f"[1/5] {len(files)} tracked files")

    if args.dry_run:
        for f in files:
            print("   ", f)
        return 0

    base = f"{API}/repos/{args.owner}/{args.repo}/git"
    me = request(f"{API}/user", token, proxy=args.proxy)
    author_name = args.author_name or me.get("login") or "unknown"
    author_email = args.author_email or f"{author_name}@users.noreply.github.com"
    print(f"[2/5] authenticated as {author_name}")

    entries = []
    for i, path in enumerate(files, 1):
        with open(path, "rb") as fh:
            raw = fh.read()
        blob = request(f"{base}/blobs", token, "POST",
                       {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"},
                       args.proxy)
        entries.append({"path": path.replace("\\", "/"), "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})
        print(f"      [{i}/{len(files)}] {path}")

    tree = request(f"{base}/trees", token, "POST", {"tree": entries}, args.proxy)
    print("[3/5] tree", tree["sha"][:10])

    parents = []
    existing = False
    try:
        ref = request(f"{base}/ref/heads/{args.branch}", token, proxy=args.proxy)
        existing = True
        parents = [ref["object"]["sha"]]
        print("      existing branch head:", parents[0][:10])
    except urllib.error.HTTPError as exc:
        # 404 = no such ref, 409 = the repository has no commits yet
        if exc.code not in (404, 409):
            raise
        print("      branch does not exist yet")
    if args.orphan and existing:
        print("      --orphan: dropping previous history")
        parents = []

    commit = request(f"{base}/commits", token, "POST",
                     {"message": args.message, "tree": tree["sha"], "parents": parents,
                      "author": {"name": author_name, "email": author_email},
                      "committer": {"name": author_name, "email": author_email}},
                     args.proxy)
    print("[4/5] commit", commit["sha"][:10])

    if existing:
        request(f"{base}/refs/heads/{args.branch}", token, "PATCH",
                {"sha": commit["sha"], "force": bool(args.orphan)}, args.proxy)
    else:
        request(f"{base}/refs", token, "POST",
                {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]}, args.proxy)
    print(f"[5/5] {args.branch} -> {commit['sha'][:10]}")
    print(f"done: https://github.com/{args.owner}/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
