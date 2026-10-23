# Security Policy

Nimo Lite is a local single-user experimental application. Security issues should be reported privately to the repository owner rather than published with working exploit details.

## Runtime boundary

- Server binds to `127.0.0.1`.
- Host/Origin validation blocks DNS-rebinding/cross-origin requests.
- State-changing requests require a local session token.
- Files are confined to the selected data root.
- Requests/imports are size bounded.
- The language model has no shell execution, Python evaluation, arbitrary URL-fetch, or OS command tool.

Generated text is text only. A command-looking model response is never automatically executed.

## Dataset/import safety

Native corpus imports accept a bounded allowlist of text/code formats. ZIP imports reject absolute paths, `..` traversal, excessive members, oversized individual members, and excessive expanded size. JSON/JSONL item counts and text sizes are bounded.

Source/license metadata is required for imported corpus data.

## Checkpoint safety

Model files load with `torch.load(..., weights_only=True)`. Task and architecture are validated before model construction. Modern language checkpoints validate vocabulary/model/context bounds and embed the tokenizer artifact/hash. Do not bypass validation for untrusted checkpoints.

`weights_only=True` reduces pickle execution risk but does not authenticate a publisher. Obtain model files from a source you trust.

## Tokenizer safety

Tokenizer JSON is bounded, parsed as data, validates fixed special-token order, byte-piece length/count, and checksum. It does not contain executable code.

## Memory

Automatic memory deliberately filters common secret/contact/credential patterns and sensitive personal categories. Memory remains inspectable and deletable. This filter is conservative and cannot guarantee that every sensitive phrase is detected; do not use automatic memory as a secret vault.

SQLite/data files are not encrypted at rest by Nimo. Protect the OS account and data directory.

## Optional web access

Web access is off by default. When enabled, the reader:
- accepts only HTTP/HTTPS
- limits ports
- resolves hosts and rejects loopback/private/link-local/reserved destinations
- revalidates redirects
- bounds response bytes
- strips active HTML content
- limits PDF extraction
- marks retrieved content as untrusted prompt data

Web pages cannot grant themselves tool permissions.

## Updates

Release checking/installing is opt-in. The updater only accepts the official `nadeemmhdm/nimo-lite` GitHub release endpoint/source archive, bounds download/extraction, and installs through the current Python environment. The one-command bootstrap similarly uses the official repository and never overwrites a dirty Git checkout.

## Native dependencies

PyTorch, Pillow, PDF parsing, and optional video libraries process complex files with the privileges of your OS user. Keep dependencies updated and process untrusted media cautiously.

## Unsupported deployments

Do not expose the loopback server through a tunnel/reverse proxy or use it as a multi-user service without adding real authentication, authorization, TLS, quotas, and isolation.
