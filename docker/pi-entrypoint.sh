#!/usr/bin/env bash
#
# Give pi its provider config, then get out of the way.
#
# The custom `gx10` provider lives in ~/.pi/agent/models.json on the host, which
# is deliberately NOT mounted — it carries auth for other providers alongside it.
# So the container renders its own, from VEB_LLM_BASE_URL, at start.
#
# The built-in providers (anthropic, openai) need no file: pi reads
# ANTHROPIC_API_KEY / OPENAI_API_KEY straight from the environment, which is
# what the harness's --env-file supplies. See pi docs/providers.md.

set -euo pipefail

: "${VEB_LLM_BASE_URL:=http://gx10-cbc5:8081/v1}"
: "${VEB_LLM_MODEL_ID:=unsloth/Qwen3.8-27B-GGUF:Q8_0}"
: "${VEB_LLM_CONTEXT_WINDOW:=200000}"
: "${VEB_LLM_MAX_TOKENS:=8192}"

agent_dir="${HOME}/.pi/agent"
mkdir -p "${agent_dir}"

cat > "${agent_dir}/models.json" <<JSON
{
  "providers": {
    "gx10": {
      "baseUrl": "${VEB_LLM_BASE_URL}",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "${VEB_LLM_MODEL_ID}",
          "name": "veb generator model (gx10)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": ${VEB_LLM_CONTEXT_WINDOW},
          "maxTokens": ${VEB_LLM_MAX_TOKENS},
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
JSON

# Deliberately bare. A defaultThinkingLevel here would silently apply to any arm
# whose model config leaves `thinking` null, which is exactly the "the run
# depends on the operator's box" failure conf/model/*.yaml already warns about.
cat > "${agent_dir}/settings.json" <<'JSON'
{
  "theme": "dark"
}
JSON

exec "$@"
