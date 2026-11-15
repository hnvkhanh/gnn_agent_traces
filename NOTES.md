# AgentDojo API notes (agentdojo 0.1.35, benchmark v1.2.2)

These notes come from reading the installed package source under
`.venv/lib/python3.12/site-packages/agentdojo/`. `tracegen` uses only what is described here and does
not modify the package.

## 1. Loading suites and a benchmark version
```python
from agentdojo.task_suite.load_suites import get_suite, get_suites
suite = get_suite("v1.2.2", "banking")      # TaskSuite; also "workspace", "travel", "slack"
get_suites("v1.2.2")                       # {name: TaskSuite}
```
`load_suites.py` imports `default_suites.v1_1 … v1_2_2`. Those imports register the task updates for each
version. `suite.user_tasks` and `suite.injection_tasks` are dicts that map an id (`"user_task_0"`,
`"injection_task_3"`) to a task **instance**. Banking v1.2.2 has 16 user tasks and 9 injection tasks.
`suite.tools` is the list of `Function`s. `suite.load_and_inject_default_environment(injections: dict)`
returns a fresh environment (a pydantic model) with the injection placeholders filled in.
`suite.get_injection_vector_defaults()` returns the placeholder ids and their default (benign) text.

## 2. Where the prompt lives
`BaseUserTask.PROMPT: str` is a class attribute (`base_tasks.py:21`). Injection tasks have `GOAL`.
`TaskSuite.run_task_with_pipeline` reads `user_task.PROMPT` when it runs (`task_suite.py:376`).

**Overriding the prompt:** `t = copy.copy(task); t.PROMPT = new_text`. The instance attribute shadows the
class attribute. `ID`, `utility`, `utility_from_traces`, `ground_truth`, `init_environment` and
`isinstance(t, BaseUserTask)` all stay the same. Attacks also read `user_task.PROMPT`
(`base_attacks.py:55,114`), so they see the paraphrase.

## 3. utility(...) and ground_truth(...)
- `task.ground_truth(pre_environment) -> list[FunctionCall]`, where `FunctionCall(function, args, id=None, placeholder_args=None)`.
- `task.utility(model_output: str, pre_environment, post_environment, strict=True) -> bool`.
- `task.utility_from_traces(model_output, pre_env, post_env, traces: Sequence[FunctionCall]) -> bool | None`. When it returns something other than `None`, that value takes precedence.
- The suite combines both in `suite._check_task_result(task, model_output_blocks, pre_env, post_env, functions_stack_trace)`.
  For a `BaseUserTask` it returns utility, and for a `BaseInjectionTask` it returns security (via
  `security_from_traces` / `security`). `model_output_blocks` is the last assistant message's `content`
  (a list of `{"type":"text","content":...}` blocks).
- `pre_environment` = `user_task.init_environment(env).model_copy(deep=True)`, taken before the agent runs.

## 4. Running one user task (no attack, no defense), prompt overridden
`suite.run_task_with_pipeline(pipeline, user_task, injection_task=None, injections={})` returns only
`(utility, security)`, not the messages. `tracegen/dojo.py::run_one` therefore repeats its body:
```python
env = suite.load_and_inject_default_environment(injections)      # {} for benign
task_env = user_task.init_environment(env)
pre_env = task_env.model_copy(deep=True)
runtime = FunctionsRuntime(suite.tools)
try:
    _, _, task_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_env)
except AbortAgentError as e:
    task_env, messages = e.task_environment, e.messages
stack = functions_stack_trace_from_messages(messages)             # agentdojo.task_suite.task_suite
utility = suite._check_task_result(user_task, messages[-1]["content"] or [], pre_env, task_env, stack)
```
(The original retries up to 3 times when the model output is `None`. tracegen records that case as a
trace with `error` set.)

Pipeline, built the same way as `AgentPipeline.from_config`:
```python
from agentdojo.agent_pipeline import AgentPipeline, SystemMessage, InitQuery, ToolsExecutionLoop, ToolsExecutor, OpenAILLM
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
llm = OpenAILLM(openai.OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama"), "llama3.1:8b")
pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                      ToolsExecutionLoop([ToolsExecutor(), llm])])   # max_iters=15
pipe.name = "local-llama3.1:8b"
```
Ollama serves an OpenAI-compatible API at `<host>/v1`, which includes tool calling for models that
support it (qwen2.5, llama3.1, mistral, but not gemma2).

**Gotcha:** `openai_llm.chat_completion_request` passes `temperature=temperature or NOT_GIVEN`. That turns
`0.0` into "not sent", so Ollama falls back to its own default (~0.8), and no seed is sent at all.
`tracegen.dojo.SeededOpenAILLM` overrides `query` to send `temperature` and `seed` explicitly and to
record token usage. It reuses the module's private helpers `_message_to_openai`, `_function_to_openai`
and `_openai_to_assistant_message`.

## 5. Running a (user task, injection task) pair with an attack
```python
from agentdojo.attacks.attack_registry import load_attack
attack = load_attack("important_instructions", suite, pipe)
injections = attack.attack(user_task, injection_task)        # {placeholder_id: text}
# then run as in section 4 with env = suite.load_and_inject_default_environment(injections),
# and also: security = suite._check_task_result(injection_task, ...)
```
`attack.attack` runs the ground-truth pipeline once to find which placeholders the user task can see
(`get_injection_candidates`). `important_instructions` puts the model name into the injected text using
`get_model_name_from_pipeline`, which raises an error unless `pipe.name` contains a key from
`agentdojo.models.MODEL_NAMES`. Naming local pipelines `local-<model>` matches the key `"local"` (→ "Local model").
`security == True` means the **attacker goal was achieved**. This follows AgentDojo's convention:
`InjectionTask.security` returns True when the injection succeeded.

## 6. Getting the full trace
`pipeline.query` returns `messages: list[ChatMessage]` (TypedDicts):
- `{"role":"system"|"user", "content":[text blocks]}`
- `{"role":"assistant", "content":[text blocks] | None, "tool_calls":[FunctionCall] | None}`
- `{"role":"tool", "content":[text blocks], "tool_call": FunctionCall, "tool_call_id": str|None, "error": str|None}`

`agentdojo.types.get_text_content_as_str(blocks)` flattens the blocks. Each tool message's `tool_call`
links it back to the call it answers. tracegen pairs them by order and id to build `tool_calls[{step,name,args,result}]`.
