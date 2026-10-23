# Security verification status

## What was performed

Source-level engineering review of local request boundaries, data paths, media input validation, training approvals, checkpoint loading, subprocess invocation, and optional network access. Seventeen functional tests include security-related regression checks; see EVALUATION.md. SECURITY.md documents the intended trust boundaries and known residual risks.

Reviewed surfaces include `nimo/app.py` (Host/Origin/token checks, strict local asset policy, no generic execution tools, fixed online provider), `nimo/security.py` (confinement and advisory review), `nimo/store.py` (bounded decoding, approval/version records), `nimo/models.py` (`weights_only=True` loading), and `nimo/training.py` (fixed-shape training, immutable metadata snapshot, atomic save, acceptance and recovery). The application never exposes model-generated arguments to a shell or arbitrary network destination.

## Codex Security integration status

The requested Codex Security skill was read, but its required referenced scan-prologue resource could not be retrieved from the skill provider. The connected tools expose only Codex Security access-status functionality, not a runnable repository-scan or report-finalization tool. Therefore **no completed Codex Security scan, canonical scanner report, or scanner certification is claimed**. This document is an engineering verification note, not a substitute for the unavailable standardized scan.

## Known gaps requiring further work

- Native media decoding and training execute with the user's OS permissions; no OS sandbox or hard memory quota exists.
- No antivirus engine, complete secret detector, semantic injection proof, or comprehensive data-poisoning protection is implemented.
- At-rest storage/backups are unencrypted, and SQL deletion is not secure erasure or unlearning.
- Large aggregate dataset exports can consume memory; requests and individual images are bounded, but aggregate disk usage has no total quota.
- Local malware and any party with access to the browser profile or data directory can bypass the single-user assumptions.
- Near-duplicate hashing is approximate; clip-level video leakage and persistent evaluation assignments across growth need stronger controls.
- Dependency vulnerability auditing and platform-specific sandbox testing were not completed.

Deployment is restricted to a trusted, single-user, loopback environment. Independent security review is required before widening that scope.
