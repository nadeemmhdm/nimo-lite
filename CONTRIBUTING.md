# Contributing

Nimo Lite welcomes small, reproducible changes with honest capability claims.

1. Create a branch from the current main branch.
2. Set up Python following the README and install `.[test]`.
3. Add focused tests for behavior, data boundaries, and recovery when relevant.
4. Run `python -m pytest -q` and document hardware-specific checks.
5. Open a pull request explaining the problem, change, validation, and limits.

Never commit personal images, local knowledge, credentials, trained weights, or external datasets without documented rights and consent. Preserve attribution and dependency licenses. Do not represent retrieval, sample entry, or label correction as a completed weight update. Do not include decorative dashboard controls or silently enable network access.

New model architectures must document initialization, preprocessing, loss, evaluation splits, parameter counts, and resource requirements. Keep evaluation separate from training and report failed experiments as clearly as successful ones. Tests using tiny synthetic data verify execution, not real-world intelligence.

Use respectful communication. Report security concerns according to SECURITY.md rather than publishing private information in an issue.
