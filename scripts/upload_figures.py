#!/usr/bin/env python3
"""Upload rendered figure PNGs to Supabase Storage and populate figures.image_url.

This is the DATA half of "make figures display": it (1) uploads each rendered PNG
to a PUBLIC Storage bucket named "figures" (object key = the filename, so it's
derivable from image_path), and (2) UPDATEs the figures table, setting image_url to
the object's public URL. A SEPARATE task wires the app renderer to use image_url.

Which figures? The figures TABLE is authoritative - we read `figure_id, image_path`
from it (via $env:SUPABASE_DB_URL, same connection the loader uses) and upload/populate
only rows that were actually loaded. (If SUPABASE_DB_URL is unset, --dry-run falls back
to scanning the local figures dir for a preview.) Use --bucket-only to push every local
PNG to the bucket WITHOUT any DB I/O, for staging figures before their rows are loaded.

Connection / credentials (read from env, never hardcoded):
  SUPABASE_URL               https://<ref>.supabase.co
                             (Dashboard -> Project Settings -> API -> Project URL)
  SUPABASE_SERVICE_ROLE_KEY  the *service_role* secret (NOT the anon key) - needed
                             for bucket admin + uploads.
                             (Dashboard -> Project Settings -> API -> service_role)
  SUPABASE_DB_URL            postgresql://postgres:<pwd>@<host>.pooler.supabase.com:5432/postgres
                             (Dashboard -> Project Settings -> Database -> Connection string)

Public URL format: <SUPABASE_URL>/storage/v1/object/public/figures/<key>

Idempotent: uploads use x-upsert (create-or-overwrite); the UPDATE writes the same
deterministic URL, so re-running is safe.

Usage:
    python scripts/upload_figures.py --dry-run     # show what WOULD happen
    python scripts/upload_figures.py               # upload + populate image_url

Requires: httpx, psycopg (both already in .venv).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

try:
    import psycopg
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "runs" / "beta_bank_v1" / "figures"
BUCKET = "figures"


# --- small helpers -----------------------------------------------------------

def env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def key_from_path(image_path: str | None, figure_id: str) -> str:
    """Object key = the bare filename from image_path ("figures/X.png" -> "X.png")."""
    if image_path:
        return image_path.replace("\\", "/").split("/")[-1]
    return f"{figure_id}.png"


def public_url(supabase_url: str, obj_key: str) -> str:
    return f"{supabase_url}/storage/v1/object/public/{BUCKET}/{obj_key}"


def _err_body(resp: httpx.Response) -> str:
    try:
        return resp.text.strip()[:300]
    except Exception:  # noqa: BLE001
        return f"HTTP {resp.status_code}"


# --- storage (service role) --------------------------------------------------

def storage_headers(svc: str) -> dict:
    return {"Authorization": f"Bearer {svc}", "apikey": svc}


def ensure_public_bucket(client: httpx.Client, base: str, svc: str) -> None:
    """Create the bucket if missing; ensure it is public. Needs the service role."""
    r = client.get(f"{base}/bucket/{BUCKET}", headers=storage_headers(svc))
    if r.status_code == 200:
        if not r.json().get("public"):
            up = client.put(
                f"{base}/bucket/{BUCKET}",
                headers=storage_headers(svc),
                json={"public": True},
            )
            if up.status_code >= 400:
                raise RuntimeError(f"could not set bucket public: {_err_body(up)}")
            print(f"  bucket '{BUCKET}': existed, switched to PUBLIC")
        else:
            print(f"  bucket '{BUCKET}': already exists and is public")
        return

    if r.status_code == 404:
        cr = client.post(
            f"{base}/bucket",
            headers=storage_headers(svc),
            json={"id": BUCKET, "name": BUCKET, "public": True},
        )
        if cr.status_code < 400:
            print(f"  bucket '{BUCKET}': created (PUBLIC)")
            return
        # Race / already-exists: fall through to making it public.
        if "exist" in _err_body(cr).lower():
            client.put(f"{base}/bucket/{BUCKET}", headers=storage_headers(svc),
                       json={"public": True})
            print(f"  bucket '{BUCKET}': already existed; ensured PUBLIC")
            return
        raise RuntimeError(f"could not create bucket: {_err_body(cr)}")

    raise RuntimeError(f"bucket check failed (HTTP {r.status_code}): {_err_body(r)}")


def upload_png(client: httpx.Client, base: str, svc: str, obj_key: str,
               data: bytes) -> None:
    """POST with x-upsert -> create or overwrite the object."""
    headers = storage_headers(svc)
    headers["Content-Type"] = "image/png"
    headers["x-upsert"] = "true"
    headers["cache-control"] = "3600"
    r = client.post(f"{base}/object/{BUCKET}/{obj_key}", headers=headers, content=data)
    if r.status_code >= 400:
        raise RuntimeError(f"upload failed (HTTP {r.status_code}): {_err_body(r)}")


# --- figure list (authoritative = DB) ----------------------------------------

def fetch_db_figures(dsn: str) -> list[dict]:
    with psycopg.connect(dsn) as conn:  # type: ignore[union-attr]
        with conn.cursor() as cur:
            cur.execute("select figure_id, image_path, image_url from figures "
                        "order by figure_id")
            return [{"figure_id": r[0], "image_path": r[1], "image_url": r[2]}
                    for r in cur.fetchall()]


def scan_local_figures(figdir: Path) -> list[dict]:
    """Fallback when no DB connection: derive a figure list from the PNG files.
    image_url unknown (treated as None)."""
    out = []
    for png in sorted(figdir.glob("*.png")):
        out.append({"figure_id": png.stem, "image_path": f"figures/{png.name}",
                    "image_url": None})
    return out


# --- bucket-only (no DB) -----------------------------------------------------

def _bucket_only(figdir: Path) -> int:
    """Upload every local PNG to the Storage bucket, without reading or writing the
    figures table. Idempotent (x-upsert)."""
    supabase_url = env("SUPABASE_URL")
    svc = env("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url:
        print("ERROR: set SUPABASE_URL.", file=sys.stderr)
        return 1
    if not svc:
        print("ERROR: set SUPABASE_SERVICE_ROLE_KEY (service_role secret).",
              file=sys.stderr)
        return 1

    # Source of truth is "what rendered" = every PNG in figdir. The loader
    # (load_to_supabase.py) creates a figure row for every passage figure, so each
    # rendered PNG needs a bucket object or its row would be a broken ref.
    pngs = sorted(figdir.glob("*.png"))
    print(f"Bucket-only upload (NO DB writes)")
    print(f"Local figures: {figdir}")
    print(f"Bucket:        {BUCKET}  (public)")
    print(f"To upload:     {len(pngs)} PNG(s) (no exclusions)\n")

    base = f"{supabase_url}/storage/v1"
    uploaded = 0
    failed: list[tuple[str, str]] = []
    with httpx.Client(timeout=60) as client:
        try:
            ensure_public_bucket(client, base, svc)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print()
        for png in pngs:
            try:
                upload_png(client, base, svc, png.name, png.read_bytes())
                uploaded += 1
            except Exception as e:  # noqa: BLE001
                failed.append((png.name, str(e)))

    print("\n=== Bucket-only upload complete ===")
    print(f"  uploaded PNGs: {uploaded} / {len(pngs)}")
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for name, reason in failed:
            print(f"    ! {name}: {reason}")
    print("\n  (No figures rows touched; image_url unset — set it in the passage-load "
          "pass via the normal `upload_figures.py` run.)\n")
    return 1 if failed else 0


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what WOULD be uploaded/updated; touch nothing.")
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR),
                    help=f"Local PNG directory (default: {DEFAULT_FIGURES_DIR}).")
    ap.add_argument("--bucket-only", action="store_true",
                    help="Upload PNGs to the Storage bucket WITHOUT touching the DB "
                         "(no figures rows read, no image_url set). For staging figures "
                         "before their rows/passages are loaded. Source of truth is the "
                         "local figures dir (every PNG in it).")
    args = ap.parse_args()

    figdir = Path(args.figures_dir)
    if not figdir.is_dir():
        print(f"ERROR: figures dir not found: {figdir}", file=sys.stderr)
        return 1

    # --- bucket-only: push PNGs to Storage, no DB reads/writes ---
    if args.bucket_only:
        return _bucket_only(figdir)

    supabase_url = env("SUPABASE_URL")
    svc = env("SUPABASE_SERVICE_ROLE_KEY")
    dsn = env("SUPABASE_DB_URL")

    # --- assemble the authoritative figure list ---
    if dsn:
        if psycopg is None:
            print('ERROR: psycopg not installed. pip install "psycopg[binary]"',
                  file=sys.stderr)
            return 1
        try:
            figures = fetch_db_figures(dsn)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not read figures from DB: {e}", file=sys.stderr)
            return 1
        source = f"DB ({len(figures)} rows)"
    elif args.dry_run:
        figures = scan_local_figures(figdir)
        source = f"local dir ({len(figures)} PNGs) - DB NOT connected, preview only"
    else:
        print("ERROR: SUPABASE_DB_URL is required for a real run (to know which "
              "figure rows exist and to UPDATE image_url). Set it, or use --dry-run.",
              file=sys.stderr)
        return 1

    if not supabase_url:
        print("ERROR: set SUPABASE_URL (Dashboard -> Project Settings -> API -> "
              "Project URL).", file=sys.stderr)
        return 1
    if not args.dry_run and not svc:
        print("ERROR: set SUPABASE_SERVICE_ROLE_KEY (Dashboard -> Project Settings "
              "-> API -> service_role secret).", file=sys.stderr)
        return 1

    print(f"Source of truth: {source}")
    print(f"Local figures:   {figdir}")
    print(f"Bucket:          {BUCKET}  (public)\n")

    # --- plan: match each figure row to a local PNG ---
    to_process: list[tuple[dict, str, Path, str]] = []   # (fig, obj_key, local, url)
    missing: list[tuple[str, str]] = []                  # (figure_id, reason)
    for fig in figures:
        obj_key = key_from_path(fig["image_path"], fig["figure_id"])
        local = figdir / obj_key
        if not local.is_file():
            missing.append((fig["figure_id"], f"PNG not on disk: {obj_key}"))
            continue
        to_process.append((fig, obj_key, local, public_url(supabase_url, obj_key)))

    # Info: PNGs on disk that don't correspond to any figure row (e.g. the orphan).
    wanted = {k for _, k, _, _ in to_process}
    orphan_pngs = [p.name for p in sorted(figdir.glob("*.png")) if p.name not in wanted]

    # --- DRY RUN ---
    if args.dry_run:
        already = sum(1 for fig, *_ in to_process if fig["image_url"])
        print(f"WOULD upload + set image_url for {len(to_process)} figure(s):")
        for fig, obj_key, _local, url in to_process:
            flag = "(image_url already set)" if fig["image_url"] else ""
            print(f"  {fig['figure_id']:30s} -> {url} {flag}")
        print(f"\n  of those, {already} already have a non-null image_url "
              f"(would be re-set to the same URL).")
        if missing:
            print(f"\nWOULD SKIP - figure rows with no PNG on disk ({len(missing)}):")
            for fid, reason in missing:
                print(f"  ! {fid}: {reason}")
        if orphan_pngs:
            print(f"\nPNGs on disk NOT matched to a figure row ({len(orphan_pngs)}) "
                  f"- left untouched:")
            for n in orphan_pngs:
                print(f"  - {n}")
        print("\nDRY RUN - nothing uploaded, nothing written.")
        return 0

    # --- REAL RUN ---
    base = f"{supabase_url}/storage/v1"
    uploaded = 0
    failed: list[tuple[str, str]] = []
    updates: list[tuple[str, str]] = []  # (image_url, figure_id)

    with httpx.Client(timeout=60) as client:
        try:
            ensure_public_bucket(client, base, svc)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}\n"
                  "If bucket admin is blocked, create it manually: Dashboard -> "
                  "Storage -> New bucket -> name 'figures', toggle 'Public bucket' ON, "
                  "then re-run.", file=sys.stderr)
            return 1
        print()

        for fig, obj_key, local, url in to_process:
            try:
                upload_png(client, base, svc, obj_key, local.read_bytes())
                uploaded += 1
                updates.append((url, fig["figure_id"]))
            except Exception as e:  # noqa: BLE001
                failed.append((fig["figure_id"], str(e)))

    # --- UPDATE figures.image_url (keyed on figure_id) ---
    updated = 0
    have_url = total = None
    if updates:
        try:
            with psycopg.connect(dsn) as conn:  # type: ignore[union-attr]
                with conn.cursor() as cur:
                    cur.executemany(
                        "update figures set image_url = %s where figure_id = %s",
                        updates)
                    updated = len(updates)
                    cur.execute("select count(*) filter (where image_url is not null), "
                                "count(*) from figures")
                    have_url, total = cur.fetchone()
                conn.commit()
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: uploads done but DB UPDATE failed: {e}", file=sys.stderr)
            return 1

    # --- summary ---
    print("\n=== Upload complete ===")
    print(f"  uploaded PNGs:        {uploaded}")
    print(f"  image_url rows set:   {updated}")
    if have_url is not None:
        print(f"  figures with URL now: {have_url} / {total}")
    if missing:
        print(f"\n  skipped (no PNG on disk) ({len(missing)}):")
        for fid, reason in missing:
            print(f"    ! {fid}: {reason}")
    if failed:
        print(f"\n  FAILED uploads ({len(failed)}):")
        for fid, reason in failed:
            print(f"    ! {fid}: {reason}")
    if orphan_pngs:
        print(f"\n  PNGs on disk not in DB (left untouched): {', '.join(orphan_pngs)}")
    print()
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
