# Pico Agent — Hermes-Inspired System Optimization Plan

## Overview
Reference Hermes Agent architecture to systematically upgrade pico-agent's core systems.

## Phase 1: Registry — AST Pre-Filter for Tool Discovery
**File:** `pico/tools/registry.py`
**Reference:** `~/.hermes/hermes-agent/tools/registry.py` lines 29-54

Hermes uses `ast.parse()` to check if a module contains `registry.register()` calls BEFORE importing it. This prevents cascading import failures (e.g. paramiko not installed → all remote tools fail silently).

Changes:
- Add `_is_registry_register_call(node)` AST helper
- Add `_module_registers_tools(module_path)` that parses source and checks for top-level register calls
- Modify `discover()` to AST-pre-filter modules before importing
- Skip modules that don't contain register calls
- Keep sub-package discovery as-is (they always have __init__.py with register_tools)

## Phase 2: Session Store — WAL, Schema Versioning, FTS5, Cost Tracking
**File:** `pico/session.py`
**Reference:** `~/.hermes/hermes-agent/hermes_state.py`

Hermes has a production-grade session store with:
- WAL mode for concurrent readers
- Schema versioning (SCHEMA_VERSION = 11, with migration)
- FTS5 full-text search across all messages
- Per-session cost tracking (input/output/cache/reasoning tokens, estimated cost)
- Session source tagging (cli, telegram, etc.)
- Compression-triggered session splitting via parent_session_id

Changes:
- Enable WAL mode: `PRAGMA journal_mode=WAL`
- Add schema_version table + migration support
- Add FTS5 virtual table for message content search
- Add token/cost columns to sessions table (input_tokens, output_tokens, cache_read, cache_write, estimated_cost_usd)
- Add `search_messages(query)` method using FTS5
- Add `record_usage(session_id, usage_dict)` method
- Add `prune_sessions(older_than_days)` method
- Add `get_stats()` method for overview statistics

## Phase 3: Agent Loop Hardening
**File:** `pico/agent.py`
**Reference:** `~/.hermes/hermes-agent/run_agent.py`

Hermes agent loop features pico-agent lacks:
- Message role alternation validation (never two assistant or two user in a row)
- Tool execution timeout per call
- Usage/cost tracking per turn
- Interrupt handling (SIGINT → graceful stop)
- Grace call (one extra turn when budget exhausted)

Changes:
- Add `_validate_message_roles()` to ensure alternation
- Add tool execution timeout (configurable, default 300s)
- Record per-turn usage in session store
- Add SIGINT handler for graceful interruption
- Fix: accumulate usage across turns for total session cost

## Phase 4: LLM Provider Upgrades
**File:** `pico/llm.py`
**Reference:** Hermes run_agent.py lazy SDK import, credential pooling

Hermes lazy-loads the OpenAI SDK (~240ms import) via a proxy object. Pico creates a new client on every `chat()` call.

Changes:
- Cache the SDK client instance per provider (don't recreate each call)
- Add `fallback_model` support: if primary model returns 429/503, try fallback
- Add `cost_estimate()` helper that calculates cost from usage + model pricing
- Add streaming support skeleton (for future CLI progress indicators)

## Phase 5: Compression Upgrades
**File:** `pico/compression.py`
**Reference:** Hermes agent/compression.py + trajectory_compressor.py

Hermes splits compressed sessions (parent_session_id chain) so history is preserved.

Changes:
- When compression triggers, save original messages to a new "archived" session
- Link archived session via parent_session_id
- Add token counting that accounts for tool schemas in the context
- Better summarization prompt that preserves tool call results and key decisions

## Phase 6: CLI Upgrades
**File:** `pico/cli.py`
**Reference:** `~/.hermes/hermes-agent/cli.py` + `hermes_cli/commands.py`

Hermes has a rich command registry, /compact, /clear, session management.

Changes:
- Add `/compact` command (force context compression)
- Add `/clear` command (clear screen + new session)
- Add `/sessions` command (list recent sessions)
- Add `/usage` command (show token usage for current session)
- Add session pruning: `pico-agent prune --older-than 30`
- Better welcome banner with version, model, tool count

## Phase 7: Test + Review + Push
- Run full pytest suite
- Fix any regressions
- Code review via code-reviewer
- Commit and push
