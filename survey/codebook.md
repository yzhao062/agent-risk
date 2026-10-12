# Codebook: Agent Characteristics Relevant to Risk

Code each source or incident against every characteristic below. Use the definitions exactly; do not add or merge codes. If something fits none, list it under "Uncoded" with a proposed name.

| Code | Characteristic | Definition |
|---|---|---|
| C1 | Tool access with real-world side effects | The agent invokes tools, commands, or APIs that change state outside the model (files, databases, money, messages, infrastructure). |
| C2 | Weak or missing human approval | Consequential actions execute without a required human confirmation, or an existing approval step is bypassed or ineffective. |
| C3 | Irreversible actions | Actions whose effects cannot be undone or rolled back (deletion without backup, payments, sent messages, destroyed infrastructure). |
| C4 | Excessive privilege or persistent credentials | The agent holds permissions, tokens, or credentials beyond what the task needs, or ambient authority that persists across tasks. |
| C5 | Long-horizon autonomy and compounding failure | Many sequential steps without human checkpoints; errors accumulate or reliability falls with task length or trajectory depth. |
| C6 | Untrusted input and prompt injection | Instructions embedded in user input, retrieved content, tool outputs, tool metadata, or third-party data redirect the agent. |
| C7 | Persistent memory or state | Memory, context, or state that persists across steps, sessions, or users, including memory poisoning and context compaction effects. |
| C8 | Multi-agent interaction and delegation | Interaction among agents, sub-agent spawning, delegation chains, or failure propagation across agents. |
| C9 | Hidden or unfaithful reasoning and observability gaps | Reasoning or actions that monitors and logs do not capture or that the agent misreports (unfaithful chain of thought, fabricated status reports, missing audit logs). |
| C10 | Underspecified goals and specification gaming | Open-ended objectives with wide discretion; reward hacking, environment tampering, or pursuing a proxy against the principal's intent. |
| C11 | Evaluation awareness and strategic behavior | Detecting evaluation versus deployment, sandbagging, scheming, or deceptive behavior toward overseers. |
| C12 | Open versus closed access | Whether weights, seeds, logs, or internals are available to the evaluator or deployer; black-box or API-only access; removability of safeguards from open weights. |
| C13 | Model updates and version drift | Behavior changes across model versions or silent updates of a hosted model. |
| C14 | Serving nondeterminism | The same input yields different outputs at fixed settings (for example batch-dependent inference at temperature zero). |
| C15 | Environment, tool, and supply-chain change or compromise | Third-party tools, plugins, skills, MCP servers, packages, or APIs that change or are malicious. |
| C16 | Speed and scale beyond human oversight | Actions at a rate or volume that prevents timely human review. |
| C17 | Oversight degradation and automation bias | Human reviewers approve without scrutiny or lose the ability to supervise. |
| C18 | Interface modality | GUI or computer-use interaction (screens, mouse, keyboard) rather than structured APIs. |
| C19 | Multi-principal conflict | Divergent interests among user, deployer, and developer that the agent may serve. |
| C20 | Premature execution under incomplete information | Acting on missing or ambiguous parameters instead of asking for clarification. |

Coding values: T (a top-level category, section, or primary finding of the source), N (named or discussed, not top-level), 0 (absent). For incidents: Y (contributed to the incident, per the incident report), 0 (no evidence). Every T, N, or Y needs a pinpoint and a short quote.
