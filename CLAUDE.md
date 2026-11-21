# Working rules for this repo

## Track every change in HISTORY.md

Work runs in a loop: **change → track → analyze → decide next**. A task is not finished until `HISTORY.md` reflects
it.

1. **Change:** any change to the repo, such as a model, training setting, data, labels, script, result or doc.
2. **Track:** add a Log entry to [HISTORY.md](HISTORY.md) the same day, using the template at its bottom. Say what
   changed (files, commits) and which result dirs it creates or invalidates (label/data versions, runs started,
   output dir, ETA).
3. **Analyze:** when results exist, add the numbers that matter (metric, split, seeds). A real investigation goes in
   [findings/](findings/README.md) as `YYYY-MM-DD-<slug>.md`, gets a row in `findings/README.md`, and is linked from
   the log entry.
4. **Decide next:** update "Current status" and the "Next" list in `HISTORY.md`. Order "Next" by priority, given the
   current results: remove what is done, add what the results call for, and say what each item is blocked on.

If a change's results are still pending, write the entry at step 2 and complete steps 3–4 when the results arrive.

## Comparability

- Results from different label or data versions are not comparable. State the version in HISTORY and findings
  (tracegen labels: v1 before 2026-11-22, v2 after; see `findings/2026-11-22-workspace-auprc-gap.md`).
- A change to labels, graph building or training code invalidates the result fingerprints. Check for running jobs
  (`results/*/progress.json`) before changing such code, and note the effect in HISTORY.
