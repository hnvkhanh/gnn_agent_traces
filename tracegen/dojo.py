"""AgentDojo wrapper: suites, prompt override, seeded OpenAI-compatible LLM, and run_one() → trace record.

See NOTES.md for the APIs used. The agentdojo package is wrapped/subclassed, never modified.
"""
import copy
import json
import os
import time
import uuid

import openai
from openai import NOT_GIVEN

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, OpenAILLM, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _message_to_openai, _openai_to_assistant_message
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages


class RunTimeout(Exception):
    pass


def load_suite(version, name):
    return get_suite(version, name)


def with_prompt(task, prompt):
    """Shallow copy of a user task whose PROMPT is `prompt`. utility/ground_truth/ID are unchanged."""
    t = copy.copy(task)
    t.PROMPT = prompt
    return t


def text_of(content):
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return "\n".join(b.get("content") or "" for b in content if b.get("type", "text") == "text")


def jsonable(x):
    return json.loads(json.dumps(x, default=str))


def call_dict(fc):
    return {"id": fc.id, "function": fc.function, "args": jsonable(dict(fc.args))}


class SeededOpenAILLM(OpenAILLM):
    """OpenAILLM that really sends temperature (0 included) and seed, counts tokens, enforces a run deadline,
    and remembers the latest message list so a crashed run still yields a partial trace."""

    def __init__(self, client, model, temperature=0.0, seed=0, deadline=None):
        super().__init__(client, model, temperature=temperature)
        self.seed, self.deadline = seed, deadline
        self.tokens_in = self.tokens_out = 0
        self.last_messages = []

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):  # noqa: B006 (agentdojo signature)
        self.last_messages = list(messages)
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise RunTimeout("run exceeded its time budget")
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        c = self.client.chat.completions.create(
            model=self.model, messages=[_message_to_openai(m, self.model) for m in messages],
            tools=tools or NOT_GIVEN, tool_choice="auto" if tools else NOT_GIVEN,
            temperature=self.temperature, seed=self.seed)
        if c.usage is not None:
            self.tokens_in += c.usage.prompt_tokens or 0
            self.tokens_out += c.usage.completion_tokens or 0
        messages = [*messages, _openai_to_assistant_message(c.choices[0].message)]
        self.last_messages = messages
        return query, runtime, env, messages, extra_args


def make_client(agent, local_url, request_timeout):
    if agent.get("base_url"):
        key = os.environ.get(agent["api_key_env"] or "OPENAI_API_KEY")
        if not key:
            raise SystemExit("API agent %s needs env var %s" % (agent["name"], agent["api_key_env"] or "OPENAI_API_KEY"))
        return openai.OpenAI(base_url=agent["base_url"], api_key=key, timeout=request_timeout, max_retries=2)
    return openai.OpenAI(base_url=local_url.rstrip("/") + "/v1", api_key="ollama", timeout=request_timeout, max_retries=1)


def build_pipeline(llm, agent, max_iters=15):
    pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                          ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters)])
    # "local" is a MODEL_NAMES key, which important_instructions needs to name the target model.
    pipe.name = agent["name"] if agent.get("base_url") else "local-" + agent["name"]
    return pipe


def serialize_messages(messages):
    out = []
    for m in messages:
        d = {"role": m["role"], "content": text_of(m.get("content"))}
        if m["role"] == "assistant":
            d["tool_calls"] = [call_dict(c) for c in (m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            d["tool_call"] = call_dict(m["tool_call"])
            d["error"] = m.get("error")
        out.append(d)
    return out


def tool_calls_of(smsgs):
    """[{step, name, args, result}] from serialized messages; step = 1-based assistant turn that issued the call."""
    calls, step, pending = [], 0, []
    for m in smsgs:
        if m["role"] == "assistant":
            step += 1
            for c in m["tool_calls"]:
                rec = {"step": step, "name": c["function"], "args": c["args"], "result": None, "error": None}
                calls.append(rec)
                pending.append((c["id"], c["function"], rec))
        elif m["role"] == "tool":
            tc = m["tool_call"]
            # pair by call id when the model gave one, else the oldest pending call with the same tool name
            hit = next((p for p in pending if tc["id"] is not None and p[0] == tc["id"]), None) or \
                next((p for p in pending if p[1] == tc["function"]), None)
            if hit is not None:
                pending.remove(hit)
                hit[2]["result"], hit[2]["error"] = m["content"], m["error"]
    return calls


def run_one(suite, task, prompt, agent, local_url, seed=0, temperature=0.0, injection_task=None, attack_name=None,
            run_timeout=300, request_timeout=180, max_iters=15, attack_template=None):
    """Run `task` with `prompt` substituted. Never raises: failures come back as a record with `error` set.
    attack_template (with injection_task): a FixedJailbreakAttack template ({goal}, {user}, {model}, {task}) used instead
    of the registered attack `attack_name`; this is how the attack search tries its own injection strings."""
    t0 = time.monotonic()
    utask = with_prompt(task, prompt)
    rec = {"trace_id": uuid.uuid4().hex, "messages": [], "tool_calls": [], "final_answer": None, "utility": False,
           "ground_truth_calls": [], "matches_ground_truth_sequence": False, "n_steps": 0, "tokens_in": 0,
           "tokens_out": 0, "latency_s": 0.0, "error": None}
    if injection_task is not None:
        rec.update(injection_placements=[], injections={}, injection_ground_truth_calls=[], security=False)
    llm = None
    try:
        client = make_client(agent, local_url, request_timeout)
        llm = SeededOpenAILLM(client, agent["name"], temperature=temperature, seed=seed, deadline=t0 + run_timeout)
        pipe = build_pipeline(llm, agent, max_iters)
        injections = {}
        if injection_task is not None:
            from agentdojo.attacks.attack_registry import load_attack
            if attack_template is not None:
                from agentdojo.attacks.base_attacks import FixedJailbreakAttack
                attack = FixedJailbreakAttack(attack_template, suite, pipe)
                rec["attack_template"] = attack_template
            else:
                attack = load_attack(attack_name, suite, pipe)
            injections = attack.attack(utask, injection_task)
            rec["injection_placements"] = sorted(injections)
            rec["injections"] = {k: str(v) for k, v in sorted(injections.items())}
        env = suite.load_and_inject_default_environment(injections)
        task_env = utask.init_environment(env)
        pre_env = task_env.model_copy(deep=True)
        try:
            rec["ground_truth_calls"] = [{"function": c.function, "args": jsonable(dict(c.args))}
                                         for c in utask.ground_truth(pre_env.model_copy(deep=True))]
        except Exception as e:  # noqa: BLE001
            rec["ground_truth_calls"] = []
            rec["ground_truth_error"] = repr(e)
        if injection_task is not None:  # the attacker's calls: the labels of the GNN guard (src/benchmarks/tracegen.py)
            try:
                rec["injection_ground_truth_calls"] = [{"function": c.function, "args": jsonable(dict(c.args))}
                                                       for c in injection_task.ground_truth(pre_env.model_copy(deep=True))]
            except Exception as e:  # noqa: BLE001
                rec["injection_ground_truth_error"] = repr(e)
        runtime = FunctionsRuntime(suite.tools)
        messages, err = [], None
        try:
            _, _, task_env, messages, _ = pipe.query(utask.PROMPT, runtime, task_env)
        except AbortAgentError as e:
            task_env, messages = e.task_environment, e.messages
        except Exception as e:  # noqa: BLE001  (timeout, HTTP error, bad tool-call JSON, context overflow, ...)
            messages, err = llm.last_messages, "%s: %s" % (type(e).__name__, e)
        stack = functions_stack_trace_from_messages(messages)
        last = messages[-1] if messages else None
        output = last["content"] if last is not None and last["role"] == "assistant" and last["content"] else []
        try:
            rec["utility"] = bool(suite._check_task_result(utask, output, pre_env, task_env, stack))
            if injection_task is not None:
                rec["security"] = bool(suite._check_task_result(injection_task, output, pre_env, task_env, stack))
        except Exception as e:  # noqa: BLE001
            err = err or "check failed: %r" % e
        smsgs = serialize_messages(messages)
        rec["messages"] = smsgs
        rec["tool_calls"] = tool_calls_of(smsgs)
        rec["final_answer"] = text_of(output) if output else None
        rec["error"] = err
    except Exception as e:  # noqa: BLE001  (setup failures: attack not applicable, client, env)
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        if llm is not None and llm.last_messages:
            rec["messages"] = serialize_messages(llm.last_messages)
            rec["tool_calls"] = tool_calls_of(rec["messages"])
    names = [c["name"] for c in rec["tool_calls"]]
    rec["matches_ground_truth_sequence"] = bool(rec["ground_truth_calls"]) and \
        names == [c["function"] for c in rec["ground_truth_calls"]]
    rec["n_steps"] = len(rec["tool_calls"])
    if llm is not None:
        rec["tokens_in"], rec["tokens_out"] = llm.tokens_in, llm.tokens_out
    rec["latency_s"] = round(time.monotonic() - t0, 3)
    return rec
