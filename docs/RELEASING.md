# Release checklist

1. Run the full test suite and the reproducible synthetic smoke experiment.
2. Review SECURITY.md, known limitations, dependency advisories, and any open blockers.
3. Update version metadata, CHANGELOG.md, and measured evaluation records.
4. Verify Git tracks only intended source, documentation, tests, and original examples.
5. Build a source archive/wheel and inspect its file list. Exclude private data and weights.
6. Review the commit and tag it after publication authorization.
7. Publish release notes that distinguish tested behavior from planned capabilities.
8. Attach checksums to any distributed artifacts and document licenses for all included material.

No GitHub release is automatically created by this project. The initial changelog describes an experimental source candidate, not a published or certified release.
