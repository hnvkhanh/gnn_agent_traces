# Graph structure

Source: `src/graph/build.py` (`trace_to_graph`), model in `src/model/gnn.py`.

Each trace (one agent run, see "Trace format" in the README) becomes **one `torch_geometric` `HeteroData` graph**.
The task is **node-level classification**: every `action` node (one tool call) gets a label `approve` or `reject`
(`src/sandbox/policy.py:LABELS`). There is no graph-level or edge-level target.

## Nodes

All node types share one feature layout, `F = 66` dims: a 64-d signed hashed bag-of-tokens of a text
(`hash_vec`: lowercase `[a-z0-9_]+` tokens, CRC32 bucket and sign, L2-normalised), followed by `step / 10` and a
constant `1.0`.

| type | count per trace | text that is hashed | step |
|---|---|---|---|
| `task` | 1 | `user_task` | 0 |
| `memory` | `len(memory)` (may be 0) | each memory entry | 0 |
| `tool` | `len(tools)`: every tool visible to the agent, used or not | `name + " " + description` | 0 |
| `action` | `len(events)`: one per tool call | `tool + " " + json.dumps(args, sort_keys=True)` | event `step` |
| `observation` | `len(events)`: the result of each call | event `observation` | event `step` |

Labels: `g["action"].y`, one per event, `LABELS.index(event["label"])`.

A trace with no events returns `None` and is skipped.

Stored fields the graph does **not** read: tool `kind` and in-scope flag, `allowed_tools`, event `obs_injected`,
event `synthetic` (only used for scoring, via `meta_of`), and `attack`. So the model is not told which tools the task
needs or which observations were injected.

## Edges

For events `i` (the current action) and `j < i` (earlier ones):

| edge type | src → dst | when |
|---|---|---|
| `(task, proposes, action)` | task 0 → action i | always |
| `(memory, recalled, action)` | memory m → action i | every memory entry, every action |
| `(tool, invoked_as, action)` | tool t → action i | t is the tool action i called (if it is in `tools`) |
| `(action, returns, observation)` | action i → observation i | always |
| `(observation, precedes, action)` | observation j → action i | every earlier observation (dense, not just j = i-1) |
| `(action, next, action)` | action j → action i | every earlier action (dense) |
| `(observation, mentions, action)` | observation j → action i | observation j contains action i's tool name (case-insensitive) |
| `(observation, copies, action)` | observation j → action i | observation j contains one of action i's argument strings of length ≥ 4 (case-insensitive; nested args are flattened) |

`mentions` and `copies` are the content-based edges that carry the "the agent is acting on what it just read" signal,
which is how a prompt injection shows up.

### Causality

Every edge into `action i` comes from a node that exists before action i is decided: the task, memory, tools, and
earlier observations/actions. Observation i is reachable only from action i, and it only feeds actions `k > i`. So the
prediction for action i does not depend on anything at or after its own result, even after 2 layers of message
passing. `tests/test_all.py::test_graph_shapes_and_causality` checks this by perturbing future nodes.

`task`, `memory` and `tool` nodes have no incoming edges, so they keep their projected features through all layers.

```
 task ──proposes──────────────┐
 memory* ──recalled───────────┤
 tool(t) ──invoked_as─────────┤
                              ▼
 action 0 ──returns──► obs 0 ──precedes / mentions / copies──► action 1 ──returns──► obs 1 ──► action 2 ...
     └────────────────────next─────────────────────────────────┘ (and 0 → 2, 1 → 2, ...)
```

## Ablations

`trace_to_graph(tr, ablate)` takes a set from `ABLATIONS` (used by the `ablate` experiment section):

| ablation | effect |
|---|---|
| `tool_feat` | tool node features hash the empty string (tool identity hidden) |
| `observation` | drops `returns`, `precedes`, `mentions`, `copies` |
| `memory` | drops `recalled` |
| `injected` | removes the trace's `injected` text from task, memory and observation texts (and from `mentions`/`copies` matching) |
| `copies` | drops `copies` |
| `mentions` | drops `mentions` |

Node sets are never removed, only edges or feature text, so shapes stay the same.

## File nodes (`src/graph/files.py`, tracegen traces)

`trace_to_graph_files(tr, ablate)` returns `trace_to_graph(tr, ablate)` plus one more node type. `build.py` itself is
not changed, so the graphs above, and the stored results, stay the same.

| type | count per trace | text that is hashed | step |
|---|---|---|---|
| `file` | one per distinct file the trace refers to | the file key (filename, or `id:<N>`) | 0 |

A trace refers to a file through: a filename in the user task; an action argument (`file_path`, `filename`, `file_id`,
`attachments` as ids or `{file_id}` dicts); or an observation (a name with a file extension, or an `id_` + `filename`
pair in a workspace file listing). When an action names a `file_id`, it resolves to a filename **only through
observations before that action**. Otherwise it gets its own `id:<N>` node. A later observation therefore cannot merge
or rename the node an earlier action uses.

| edge type | src → dst | when |
|---|---|---|
| `(file, targets, action)` | file f → action i | action i's arguments reference f |
| `(file, appears_in, observation)` | file f → observation j | observation j names f |

File nodes have **no incoming edges**, so their state is just their projected feature. Observation j only feeds actions
after j, so the causality argument above is unchanged. `tests/test_tracegen_gnn.py` checks this by perturbing future
events, including the action's own observation.

Ablations, in addition to the list above: `files` drops both file edge types and keeps the file nodes isolated.
`attacker_values` is a leakage control: it replaces every attacker-chosen argument string of the suite with one
placeholder token in the task, action and observation text, in benign and attacked traces alike.

The model is `FileHeteroGNN` (`src/model/gnn_files.py`). It has the same architecture as `HeteroGNN`, but takes the node
and edge type lists as parameters.

## Edge graph (`src/graph/edges.py`, tracegen guard `edge-gnn`)

This graph flips the representation: artifacts are nodes and actions are edges.

**Nodes:**
- `agent`: a generic assistant node.
- `task`: the user prompt.
- `memory`
- `tool`
- `file`: the same rules as the file nodes above.
- `entity`: argument values of 4–64 characters, such as an IBAN, an email, a channel, a URL or a hotel name. An entity
  is keyed by its lowercased value, so a recipient and the same string in an earlier tool result are one node.

**Edges.** A past action j is a group of edges, each carrying `feat(text, step j)` as an edge feature:
- `agent –calls→ tool`
- `tool –acts_on→ entity|file`: one per argument.
- `tool –returns→ agent`: the observation.
- `tool –returned→ entity|file`: the values the observation names.

Context edges are `task|memory –mentions→ entity|file`. Every edge has a `rev_` twin, so messages flow both ways.

**Causality.** It comes from snapshots, not from edge direction. The graph holds one block-diagonal snapshot per
action. Snapshot k is built only from the prompt, the tools, events < k and the arguments of action k. Action k is the
snapshot's `query`, and read-only `q_*` index edges point at its agent, task, tool and argument nodes.

**Model.** `EdgeGNN` (`src/model/edge_gnn.py`) runs 2 `HeteroConv` layers of `GINEConv`, which uses the edge features,
with residual connections. The readout is `[h_agent; h_task; h_tool; mean h_args; query features]`.

**Ablations:** `directed`, `entities`, `returned`, `task_mentions`, `observation` and `attacker_values`.

## Prompt embedding (`src/prompt_embed.py`, guards `gnn+files+prompt`, `edge-gnn+prompt`)

A frozen `sentence-transformers/all-MiniLM-L6-v2` embedding of the user prompt (384-d, mean-pooled, L2-normalised,
cached in `data/cache/`) is projected and used in two places:
- it is added to the task node's initial state;
- it is concatenated into the readout.

There are two controls:
- `prompt_zero`: an all-zero embedding.
- `prompt_shuffle`: the prompt of another user task in the same suite.

The report compares each prompt guard with its no-prompt model and with both controls. It uses a paired bootstrap
over traces.

## Model on top

- `HeteroGNN`: a per-type `Linear(F, 64)` + ReLU, then 2 `HeteroConv` layers (one `SAGEConv` per edge type, summed),
  each with ReLU and a residual, then an MLP head on `x["action"]` → 2 logits per action.
- `ActionMLP` (baseline): the same action features, no edges.
- `GraphTransformer` (`src/model/transformer.py`, tracegen guard `transformer`): it runs on the same files graph as
  `FileHeteroGNN`. Every node is a token (per-type `Linear(F, 64)` + ReLU + a type embedding). 2 pre-LN attention
  layers with 4 heads. A causal mask lets action i attend only to the context (task/memory/tool/file nodes), actions
  ≤ i and observations < i. Each edge type adds a learned per-head bias to the attention logit dst → src, so the
  model sees the same relations as the GNN and the edge ablations apply to it too. The MLP head sits on the action
  tokens.
- Training (`src/train.py:fit`): batches of 32 graphs, class-weighted cross-entropy over all action nodes, Adam.

## Split metadata

`meta_of(tr)` records per graph: benchmark, model, agent, task `group`, `attack_id` (`attack_key`: the InjecAgent
attacker tool or the AgentDojo injection task), attack style/mode/category, per-action `synthetic` flags and tool
names. Splits use `group` and `attack_id` so train and test never share a task group (task split) or an attack.
