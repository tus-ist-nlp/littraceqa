# src/littraceqa/di_pipeline/llm/

The LLM client `ReadingAgent` uses, reduced to one shape:
`__call__(prompt: str) -> str`.

- `base.py` — the `LLMClient` Protocol (that one method, and nothing else)
- `azure_openai.py` — `AzureOpenAILLM`: production, a real Azure OpenAI client
- `fake.py` — `FakeLLM`: for tests; returns the responses it was given, in order

`pipeline.build_agent()` passes `AzureOpenAILLM(reasoning_effort="medium")` by
default; a test hands `ReadingAgent` a `FakeLLM` instead.

## Credentials

API keys and the like come from `.env` at the repository root — never from code or
configuration. `.env.example` lists what is required.

**Constructing the client without a key raises immediately.** The agent wraps every
LLM call in try/except and falls back, so an authentication failure during a run
would show up as quietly degraded retrieval rather than an error. Surfacing it
while the pipeline is being assembled is what makes that impossible.
