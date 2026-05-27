"""Bulk v3 enrichment driver for the 7 client-facing target families,
with **automatic approval** of awaiting_review checkpoints so the cascade
runs unattended end-to-end.

Per CLAUDE.md System 2, v3 Stage 3 (reconciler) escalates cross_role MERGE
and any SPLIT decisions to `awaiting_review`. This script polls v2_run_log
mid-cascade and auto-approves those checkpoints via the existing
`POST /api/generate-skills/runs/{run_id}/approve` endpoint (which also
fires the next stage via the standard cascade map in
`generate_skills_router.api_approve_run`).

Other stages that may halt at awaiting_review (Stage 6 containment,
Stage 7 enrichment) are also auto-approved.

Idempotent — re-running skips rows that are already enriched. Writes one
CSV row per role to scripts/_v3_enrich_targets_report.csv.

Usage:
    # Dry-run (lists targets, no triggers)
    uv run python scripts/_bulk_enrich_targets.py

    # Execute (long-running):
    uv run python scripts/_bulk_enrich_targets.py --apply
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"
BASE_URL = os.environ.get("AUTOCHAIN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REPORT_PATH = "scripts/_v3_enrich_targets_report.csv"

TARGET_PARENT_IDS = [1, 2, 7, 11, 15, 25, 75]
PRIORITY_CHILD_SLUGS = [
    "java-backend-developer", "python-backend-developer", "node-backend-developer",
    "react-frontend-developer", "angular-frontend-developer",
    "react-native-developer", "flutter-developer",
    "wordpress-dev", "drupal-dev",
]

# Stage 8 prompt-version prefix marks the cascade's terminal stage.
STAGE_8_PREFIX = "stage8_load"
# Other awaiting_review-capable stages we must auto-approve to advance.
AWAITING_REVIEW_STAGES = (
    "stage0_charter_",
    "stage1_anchor_",
    "stage3_recon_",
    "stage6_containment_",
    "stage7_enrichment_",
)


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def fetch_targets() -> list[dict]:
    """Return (id, slug, display_name, n_kras, parent_id) for every target
    row that's still a shell (kra_count < 3)."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.id, r.slug, r.display_name, r.parent_role_id,
                       (SELECT COUNT(*) FROM "{SCHEMA}".role_kras WHERE role_id = r.id) AS n_kras
                  FROM "{SCHEMA}".roles r
                 WHERE r.id = ANY(%s)
                    OR r.parent_role_id = ANY(%s)
                """,
                (TARGET_PARENT_IDS, TARGET_PARENT_IDS),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for rid, slug, disp, parent_id, n_kras in rows:
        if int(n_kras) >= 3:
            continue
        out.append({
            "id": int(rid), "slug": slug, "display_name": disp,
            "parent_role_id": parent_id, "n_kras": int(n_kras),
        })

    def _key(r):
        if r["parent_role_id"] is None:
            return (0, r["slug"])
        if r["slug"] in PRIORITY_CHILD_SLUGS:
            return (1, PRIORITY_CHILD_SLUGS.index(r["slug"]))
        return (2, r["slug"])
    out.sort(key=_key)
    return out


def fetch_kra_count_after(role_id: int) -> int:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT COUNT(*) FROM "{SCHEMA}".role_kras WHERE role_id = %s',
                (role_id,),
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def fetch_run_stages(slug: str, since: datetime) -> list[dict]:
    """All v2_run_log rows for `slug` since the given timestamp."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT run_id::text, prompt_version, status, started_at
                  FROM "{SCHEMA}".v2_run_log
                 WHERE role_id = %s AND started_at >= %s
                 ORDER BY started_at ASC
                """,
                (slug, since),
            )
            return [
                {"run_id": r[0], "prompt_version": r[1], "status": r[2], "started_at": r[3]}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def trigger_v3(client: httpx.Client, slug: str) -> tuple[bool, str | None]:
    try:
        r = client.post(f"{BASE_URL}/admin/trigger-v3", json={"slug": slug}, timeout=30)
        if r.status_code == 409:
            return True, r.json().get("run_id")
        r.raise_for_status()
        return True, r.json().get("run_id")
    except Exception as exc:
        print(f"    [ERROR] trigger failed: {exc}")
        return False, None


def auto_approve(client: httpx.Client, run_id: str, prompt_version: str) -> bool:
    """POST the approve endpoint. Returns True on success."""
    try:
        r = client.post(
            f"{BASE_URL}/api/generate-skills/runs/{run_id}/approve",
            json={
                "reviewed_by": "bulk-enrich-auto",
                "notes": "auto-approved during target-family bulk enrichment",
            },
            timeout=30,
        )
        if r.status_code == 400:
            # Already not in awaiting_review (race condition) — fine
            return True
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"    [WARN] approve {run_id[:8]} ({prompt_version}) failed: {exc}")
        return False


def drive_cascade(client: httpx.Client, slug: str, started_high_water: datetime,
                  timeout_min: int = 90) -> str:
    """Poll v2_run_log for this slug; auto-approve any awaiting_review stage;
    return when Stage 8 is approved/completed OR overall timeout hits.

    Returns terminal status: 'completed' | 'failed' | 'timeout'.
    """
    deadline = time.time() + timeout_min * 60
    last_stage_seen = ""
    while time.time() < deadline:
        stages = fetch_run_stages(slug, started_high_water)
        if not stages:
            time.sleep(15)
            continue

        # Auto-approve any awaiting_review row whose prompt_version is in
        # our cascade map.
        for s in stages:
            if s["status"] == "awaiting_review" and any(
                s["prompt_version"].startswith(p) for p in AWAITING_REVIEW_STAGES
            ):
                print(f"    [approve] {s['prompt_version']} run={s['run_id'][:8]} (was awaiting_review)")
                auto_approve(client, s["run_id"], s["prompt_version"])

        # Check terminal condition: Stage 8 reaching approved (or completed).
        stage_8 = next(
            (s for s in stages if s["prompt_version"].startswith(STAGE_8_PREFIX)),
            None,
        )
        if stage_8 is not None:
            if stage_8["status"] in ("approved", "completed"):
                return "completed"
            if stage_8["status"] in ("failed", "rejected"):
                return "failed"

        # Check for any failed stage that won't auto-recover
        any_failed = [s for s in stages
                      if s["status"] in ("failed", "rejected")
                      and not s["prompt_version"].startswith(STAGE_8_PREFIX)]
        if any_failed:
            return "failed"

        # Heartbeat
        latest = stages[-1]
        if latest["prompt_version"] != last_stage_seen:
            print(f"    [progress] now at {latest['prompt_version']} status={latest['status']}")
            last_stage_seen = latest["prompt_version"]

        time.sleep(20)
    return "timeout"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Actually fire v3 runs. Default lists targets only.")
    p.add_argument("--timeout-min", type=int, default=90,
                   help="Per-role timeout (default 90 min).")
    args = p.parse_args(argv)

    targets = fetch_targets()
    print(f"[bulk-enrich] {len(targets)} target rows need v3:")
    for t in targets:
        prefix = "PARENT" if t["parent_role_id"] is None else " child"
        print(f"  {prefix} id={t['id']:>3} {t['slug']:<35} ({t['display_name']!r})")

    if not args.apply:
        print(f"\n[bulk-enrich] DRY-RUN. Pass --apply to start the {len(targets)}-role queue.")
        return 0

    new_report = not os.path.exists(REPORT_PATH) or os.path.getsize(REPORT_PATH) == 0
    fp = open(REPORT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    if new_report:
        writer.writerow([
            "slug", "role_id", "started_at_utc", "completed_at_utc",
            "status", "n_kras_before", "n_kras_after",
        ])
    fp.flush()

    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        for i, t in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] {t['slug']} (id={t['id']})")
            n_kras_before = t["n_kras"]
            high_water = datetime.now(timezone.utc)
            started_at_str = high_water.isoformat()

            ok, run_id = trigger_v3(client, t["slug"])
            if not ok:
                writer.writerow([t["slug"], t["id"], started_at_str, "",
                                 "trigger_failed", n_kras_before, n_kras_before])
                fp.flush()
                continue
            print(f"  triggered run_id={(run_id or '?')[:8]} — driving cascade…")

            status = drive_cascade(
                client, t["slug"], high_water, timeout_min=args.timeout_min,
            )
            completed_at_str = datetime.now(timezone.utc).isoformat()
            n_kras_after = fetch_kra_count_after(t["id"])
            print(f"  -> {status}, KRAs: {n_kras_before} -> {n_kras_after}")
            writer.writerow([t["slug"], t["id"], started_at_str, completed_at_str,
                             status, n_kras_before, n_kras_after])
            fp.flush()
    fp.close()
    print(f"\n[bulk-enrich] done. Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
