# Security policy

## Support status

Version 0.2 is experimental and has no production security support guarantee. Only the latest source receives fixes. Do not use it as a credential store, internet-facing server, or authority for high-impact decisions.

## Reporting a vulnerability

Use the repository's private security reporting facility if enabled. If no private channel is configured, open an issue containing only a request for a private contact; do not publish secrets, private datasets, or exploit details. No security contact address is invented here.

## Enforced application boundaries

- CLI binds to `127.0.0.1`; Host allowlist blocks DNS rebinding to arbitrary names.
- Browser mutations require matching Origin when supplied and the local session token. Cross-origin reads are restricted by browser same-origin rules; no CORS permission is granted.
- A strict local-only content policy, text-only DOM insertion, no framing, and no content sniffing reduce browser injection risk.
- The intent classifier can select only allowlisted authored responses and falls back on uncertain or unsupported labels. Model scores are not calibrated.
- Models cannot invoke shell commands, arbitrary HTTP fetches, uploads, filesystem tools, or generated programs. Training worker arguments are fixed by application code and never passed through a shell.
- Online search is off at startup; explicit queries use a fixed Wikipedia HTTPS endpoint, TLS verification, timeout, and disabled redirects. There is no web crawler or automatic training import.
- Images are decoded and re-encoded for preview, constrained by accepted formats, upload bytes, and pixel count. Original bytes are never served as executable content. Filenames are content-addressed.
- Annotation coordinates, labels, UUIDs, and data paths are validated. Generic archive import, pickle model import, and arbitrary local path import are absent from the HTTP API. An explicit local CLI installs only bounded, fixed-name model bundles with hashes and architecture limits. Checksums do not authenticate the publisher; trust the bundle source.
- Pending training examples require approval. Flagged examples require explicit review acknowledgment. Dataset writes are blocked during live runs. Atomic checkpoints and accepted-version rollback preserve recovery paths.
- Imported notes, images, OCR-like transcriptions, source metadata, and generated outputs remain untrusted. The pattern detector is advisory and is not a semantic security guarantee.

## Residual risks

This is not an OS sandbox. Training and optional media decoders execute as the current user and inherit the process environment and OS permissions. Do not grant this process production credentials. Native image/video/PyTorch dependencies may contain vulnerabilities; install maintained versions. Local malicious software, a compromised browser, and other users with data-directory access can read or modify data. Secrets and personal information detection is incomplete. A person can explicitly approve harmful or mislabeled examples; no scanner establishes truth or safety.

There is no at-rest encryption, authentication for multiple users, robust rate limiting, antivirus integration, hard process memory quota, outbound firewall, model-output safety classifier, or verified cross-platform sandbox. The fixed model shapes, bounded inputs, periodic resource checks, and one-worker policy reduce ordinary resource exhaustion but cannot guarantee the OS will never terminate a process. Exports are assembled in memory and are intended only for small prototype datasets.

Use a dedicated OS user, encrypted storage, no privileged credentials, and an outbound-denying firewall for stronger isolation. Offline mode disables the application's search feature; it does not reconfigure OS networking. Never expose the loopback service through a tunnel or reverse proxy. Run one server per data directory.

## Data lifecycle

Original image bytes, processed copies, annotations, feedback, knowledge, audit events, and checkpoints remain under the chosen data directory. Sample deletion removes sample rows and image copies. Knowledge deletion removes its row. SQLite may retain deleted content in free pages/WAL, and weights may retain information; deletion is not secure erasure or machine unlearning. Stop the application before backing up the full directory. Protect backups and exports as private data. Rotate the local token by stopping the app and deleting `session.token`.

## Security tests

The automated suite exercises missing tokens, hostile Origin/Host headers, path confinement, flagged-example approval, corrupt images, invalid boxes, offline core operation, and absence of automatic feedback-to-training promotion. These tests do not prove resistance to every prompt injection or malicious native-media input.
