# Findings

Investigations of results, one file per investigation, newest first. This is the "analyze" step of the work loop in
[HISTORY.md](../HISTORY.md): change → track → **analyze** → decide next.

| date | finding | status | summary |
|---|---|---|---|
| 2026-11-22 | [Workspace AUPRC gap](2026-11-22-workspace-auprc-gap.md) | decided, v2 retrain pending | Workspace's 0.10 vs banking's 0.73 comes mostly from the reject share (0.66% vs 12.7%) and from missing labels (AgentDojo workspace injection tasks 6–13 have no ground truth). Fixed by labels v2. |
| 2026-11-21 | [Seed variance and stable training](../results/tracegen_all_edge/ANALYSIS.md) (in its result dir) | done | Seed variance came from unstable training and a 13-reject test split. Use 5×5 CV; per-guard `GUARD_FIX`, with Edge GNN on stable settings (CV 0.294 → 0.377). |

Status: **open** (question not answered yet), **decided** (answered, with an action chosen and possibly still
running), **done** (action finished and its results recorded).

## Conventions

- File name: `YYYY-MM-DD-<slug>.md`. Start the file with the date, status and links to this index and to
  `HISTORY.md`.
- Sections: **Question**, **Reproduce** (command and the result dir it read), **Short answer**, **Evidence** (tables
  with metric, split and seeds), **Decision and open items**.
- Put the analysis script in `scripts/` and its raw output next to the result dir it read, and link both.
- Every finding gets a row here and a `HISTORY.md` log entry that links to it. When its status changes, update the
  row, the file header and `HISTORY.md` "Next".
- A finding records the label/data version it used. Numbers from different versions are not comparable.
