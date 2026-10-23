# Troubleshooting

| Symptom | Explanation and action |
|---|---|
| No model available | Train and evaluate the matching task. An accepted version must exist before model chat or suggestions. Knowledge retrieval is available without weights. |
| Empty split / missing category in split | Add independent diverse samples, at least three near-duplicate groups per class. Identical or near-identical images do not create independent evidence. |
| Conflicting labels or answers | Review duplicate groups or repeated questions in Dataset Manager. Correct annotations and approve again. |
| Dataset locked | Pause retains the training snapshot. Stop safely before editing any sample. |
| Insufficient memory | Close other workloads. The original tiny shapes are already bounded; larger custom models require separate engineering. A 6 GB target is not a 6 GB benchmark result. |
| Failed or interrupted run | Read the job error and `data/models/<job-id>.log`. Resume its saved checkpoint with unchanged dataset and a sufficient total epoch target. If no checkpoint exists, start a fresh run. |
| Resume dataset mismatch | Annotation version, labels, or split changed. Restore an exact dataset backup or begin a new experiment. |
| Loss improves but version not active | Classification acceptance also requires validation accuracy at least 0.5 and at least the reference model's accuracy. Failed versions remain visible. |
| Gibberish text output | Expected from tiny experimental models and limited training. Store sourced notes for useful retrieval; do not interpret lower byte loss as reasoning ability. |
| Invalid token / cross-origin error | Open the local app URL and refresh. Do not call mutation endpoints from another site. Avoid multiple servers on the same data directory. |
| Host denied | Use localhost/127.0.0.1. Remote hosting is unsupported. |
| Search fails | Enable online search explicitly, verify connectivity, and try later. Wikipedia may rate-limit or reject the request. Offline core functions remain usable. |
| Video dependency missing | Install `.[video]`. Only frame preparation is implemented; there is no audio transcription. |
| Tests hang under an execution sandbox | TestClient needs local socket/event-loop access. Run the tests with local network permission in the sandbox. This does not mean offline app features need internet. |

For reproducible bug reports include Python/PyTorch versions, OS, RAM tier, task, error text, and a minimal synthetic example. Exclude private data, full environment dumps, session tokens, and credentials.
