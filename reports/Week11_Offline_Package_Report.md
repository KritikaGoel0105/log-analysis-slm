# Week 11 Report — Offline Deployment Packaging (D9)

Document basis: Section 4 Week 11 row ("Offline deployment packaging:
Docker image, setup scripts, no-internet validation test —
deliverable: Offline install package (.tar.gz or Docker image)");
Section 8 D9 ("Docker image OR install archive; clean offline install
tested on VM with no internet"); Section 9 tree (`docker/Dockerfile —
Offline Docker build`, `setup_offline.sh — One-command offline
environment setup script`); Section 10.1(5) ("Offline install package
deploys successfully on a fresh VM"); Section 5.1 checklist item 6
("Test full installation on a clean VM before site deployment").

## Overview

Week 11 packages the frozen Weeks 1–10 system into a single
self-contained offline install artifact and provides an automated
no-internet validation test that proves a clean install works with
zero internet dependencies (Section 10.1-1).

## Deliverable choice (Document Requirement vs Engineering Decision)

- **Document Requirement:** the deliverable is an "Offline install
  package (**.tar.gz or Docker image**)" — Section 4; D9 repeats the
  choice: "Docker image **OR** install archive". Either satisfies D9.
- **Engineering Decision:** the **.tar.gz install archive is the
  primary, runtime-tested deliverable**. Reason: the project's
  `offline_packages/` wheels are Windows (`cp311-win_amd64`) builds
  matching the actual target machine; a Linux Docker image cannot
  install Windows wheels, so a container build additionally requires
  a one-time online download of Linux wheels (documented inside the
  Dockerfile). The `docker/Dockerfile` is still provided because the
  Section 9 repository tree mandates its existence, and it defines a
  fully offline (`--network=none`) build for Linux hosts.

## Files created (all additive — zero Week 1–10 files modified)

| File | Purpose |
|---|---|
| `package_offline.sh` | Builds `dist/log-analysis-slm-offline.tar.gz`: pre-flight asset check (Section 5.1 checklist), archive of source + reports + wheels + models + data + scripts (excluding `.git`, `venv/`, `testenv/`, `dist/`), SHA-256 checksum for transfer integrity |
| `validate_offline.sh` | The Section 4 "no-internet validation test": (1) verifies no internet is reachable, (2) verifies every offline asset exists, (3) imports every dependency from the local environment, (4) runs the FULL frozen pipeline offline (preprocess → retrieve → generate → parse on a sample log line, asserting a parsed severity), (5) starts the Week 9 API in-process and asserts `/health` returns `{"status": "healthy", ..., "offline": true}`. Exit code 0 = D9 pass |
| `docker/Dockerfile` | Section 9-mandated offline Docker build: `pip install --no-index` from local wheels, offline env vars baked in (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PIP_NO_INDEX=1`), serves the Week 9 API by default, dashboard as alternate command |
| `reports/Week11_Offline_Package_Report.md` | This report |

`setup_offline.sh` (Section 9 "setup scripts") already exists from
Week 1 and is FROZEN — the packaging reuses it unchanged as the
install step inside the archive.

## Deployment procedure (air-gapped target)

```
# On the connected build machine (this repo):
bash package_offline.sh
#   -> dist/log-analysis-slm-offline.tar.gz (+ .sha256)

# Transfer both files by USB/secure media to the offline VM, then:
sha256sum -c log-analysis-slm-offline.tar.gz.sha256
tar -xzf log-analysis-slm-offline.tar.gz
cd log-analysis-slm-offline
bash setup_offline.sh          # frozen Week 1 script — local wheels only
bash validate_offline.sh       # D9 no-internet validation test
```

All five validation stages must report PASS and the script must print
`D9 no-internet validation: SUCCESS`.

## Offline confirmation

- The archive contains everything Section 5.1's checklist requires on
  the target: wheels (`offline_packages/`), base model
  (`models/qwen25-3b/`), adapter (`models/checkpoints/`), embedder
  (`models/sentence-transformers/`), datasets (`data/dataset/`), FAISS
  index + sidecar (`data/faiss.index`, `data/faiss_incidents.json`).
- `validate_offline.sh` actively probes for internet and treats
  reachability as a warning: the official D9 evidence run must occur
  with networking disabled ("clean offline install tested on VM with
  no internet").
- The Dockerfile build is `--network=none`-compatible: `PIP_NO_INDEX=1`
  and `--no-index --find-links` guarantee no package index is
  contacted; HF offline env vars guarantee no model hub is contacted.

## Validation checklist (runtime — to be executed by the intern)

- [ ] `bash package_offline.sh` completes; `dist/log-analysis-slm-offline.tar.gz`
      and `.sha256` exist
- [ ] `sha256sum -c` passes after transfer to the offline VM
- [ ] Archive extracts; `setup_offline.sh` completes with no network
- [ ] `validate_offline.sh` prints `No internet reachable`
- [ ] All asset, import, pipeline and API checks report PASS
- [ ] Final line: `D9 no-internet validation: SUCCESS` (exit code 0)

## Deliverables status

| Deliverable | Status |
|---|---|
| `package_offline.sh` (archive builder) | ✅ created |
| `validate_offline.sh` (no-internet validation test) | ✅ created |
| `docker/Dockerfile` (Section 9 offline Docker build) | ✅ created |
| `reports/Week11_Offline_Package_Report.md` | ✅ this file |
| `dist/log-analysis-slm-offline.tar.gz` | ⬜ built by running `package_offline.sh` (not committed — ~7 GB artifact; Section 9 already excludes model weights from Git) |
| Clean offline install test on VM | ⬜ runtime evidence to be produced by the intern (checklist above) |

## Compliance summary

- Section 4 Week 11: packaging script ✅ · Docker image definition ✅
  · setup scripts ✅ (frozen `setup_offline.sh`, reused) ·
  no-internet validation test ✅ (`validate_offline.sh`).
- Section 8 D9: install archive ✅ (primary) · Docker image ✅
  (build definition; Linux-wheel prerequisite documented) · clean
  offline install test ⬜ runtime evidence pending intern execution.
- Section 9 tree: `docker/Dockerfile` now present at the mandated
  path; no other tree changes.
- Section 10.1(5): deployment procedure defined and automated;
  fresh-VM run pending (runtime).
- Freeze policy: zero modifications to Weeks 1–10 files — all four
  Week 11 files are new.
