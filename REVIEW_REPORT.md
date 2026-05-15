# Pico Agent — Code Review Report

**Reviewer:** REVIEWER Agent (automated)  
**Date:** 2026-05-14  
**Version Reviewed:** 0.1.0  
**Files Analyzed:** 24 Python modules, 12 test files, README, pyproject.toml, config.example.yaml

---

## Executive Summary

**Overall Quality Rating: 7.5 / 10**

Pico Agent is a well-structured AI agent framework with clean architecture, good separation of concerns, and comprehensive tooling. The codebase demonstrates solid software engineering practices: dependency injection in the core agent, ABC-based provider abstraction, protocol-based compression, and a registry pattern for tools.

**Top 3 Priorities:**
1. 🔴 **Security: Archive extraction without path validation** (zip slip vulnerability in `dataset_tools.py`)
2. 🟠 **Security: No filesystem sandboxing** — the agent can read/write any file on the system
3. 🟡 **Architecture: `shell=True` usage without allowlisting** — 12 instances across the codebase

---

## Architecture Findings

### [OK] Separation of Concerns — Generally Well Done
- `agent.py` (218 lines): Clean single-responsibility — conversation loop only.
- `config.py` (230 lines): Config loading, validation, env overrides. Clean dataclass design.
- `llm.py` (343 lines): ABC + two provider implementations. Clean factory pattern.
- `session.py` (235 lines): SQLite storage with null-object `MemorylessSession`.
- `compression.py` (136 lines): Single-responsibility with Protocol-based loose coupling.
- `delegation.py` (83 lines): Minimal and focused.
- `tools/registry.py` (249 lines): Registration, schema export, dispatch — well-structured.

### [LOW] cli.py Is Doing Too Much
- **Location:** `pico/cli.py` (616 lines)
- **Description:** CLI parsing, REPL loop, 6 shortcut commands, 6 rich helper functions, and slash command handling are all in one file.
- **Recommendation:** Extract rich helpers to `pico/display.py` and shortcut commands to `pico/shortcuts.py`. Keep cli.py as thin routing only.

### [MEDIUM] Global State in Tool Modules
- **Location:** `pico/tools/terminal.py:19-20` — `_bg_processes: dict`, `_bg_counter: int`
- **Location:** `pico/tools/remote/ssh_client.py:26` — `_ssh_clients: dict`
- **Location:** `pico/config.py:218` — `_current_config: Config | None`
- **Description:** Module-level mutable state makes testing harder and can cause state leakage between tests.
- **Recommendation:** Encapsulate background process tracking in a class. For config caching, ensure tests can reset the cache.

### [LOW] Duplicate Function Definition
- **Location:** `pico/tools/remote/__init__.py:27-28` and `pico/tools/remote/__init__.py:42-43`
- **Description:** `_paramiko_available()` is defined twice in the same file. The second definition silently shadows the first.
- **Recommendation:** Remove the duplicate (lines 42-43).

### [MEDIUM] Private Attribute Access Across Module Boundaries
- **Location:** `pico/delegation.py:57` — `tool_registry._tools.values()`
- **Location:** `pico/tools/__init__.py:29` — `len(registry._tools)`
- **Description:** Accessing `_tools` (private by naming convention) from outside the class. This couples callers to internal implementation.
- **Recommendation:** Add a public `ToolRegistry.get_tool_count()` and `ToolRegistry.get_all_tools()` method.

### [LOW] Config Mutation from Outside
- **Location:** `pico/cli.py:488` — `config.data["_history_path"] = history_path`
- **Description:** CLI mutates the Config data dict directly with an internal key (prefixed with `_`). This breaks encapsulation.
- **Recommendation:** Add a `history_path` property to Config or pass it separately.

### [LOW] Inconsistent Session Title Lengths
- **Location:** `pico/agent.py:71` — truncates at 50 chars for `create_session`
- **Location:** `pico/agent.py:146` — truncates at 60 chars for `update_session_title`
- **Recommendation:** Use a consistent constant, e.g., `MAX_TITLE_LEN = 60`.

### [OK] Dependency Injection Pattern
- `AIAgent.__init__` properly accepts `config`, `tool_registry`, `session`, `memory` via constructor.
- `ContextCompressor` takes a `_LLMCallable` Protocol, not a concrete class.
- `MemorylessSession` and `NoMemory` serve as null objects for delegation.
- **This is good design.**

### [OK] Error Handling Patterns
- All tools return standardized JSON error responses via `_error()`.
- `ToolRegistry.dispatch()` wraps all handler exceptions (line 185-190).
- CLI provides user-facing error messages with exit codes.
- Agent loop handles `KeyboardInterrupt` gracefully.

---

## Security Findings

### [CRITICAL] Zip Slip / Path Traversal in Archive Extraction
- **Location:** `pico/tools/dataset_tools.py:328` — `zf.extractall(output_path)`
- **Location:** `pico/tools/dataset_tools.py:333` — `tf.extractall(output_path)`
- **Description:** `extractall()` without filtering is vulnerable to zip slip attacks. A malicious archive with entries like `../../etc/cron.d/backdoor` can write files outside the target directory.
- **Impact:** Arbitrary file write if a malicious dataset archive is downloaded.
- **Recommendation:** Use Python 3.12+ `filter='data'` parameter, or validate all member paths:
  ```python
  import tarfile
  def _safe_extract(tar, path):
      for member in tar.getmembers():
          member_path = os.path.join(path, member.name)
          abs_path = os.path.realpath(member_path)
          if not abs_path.startswith(os.path.realpath(path)):
              raise Exception(f"Path traversal in {member.name}")
      tar.extractall(path)
  ```

### [HIGH] No Filesystem Sandboxing
- **Location:** `pico/tools/file_tools.py:37,67,95` — `Path(path).expanduser().resolve()`
- **Description:** `read_file`, `write_file`, and `search_files` can access any path on the filesystem. While paths are resolved (preventing simple `../` tricks), there's no restriction to a safe directory.
- **Impact:** An LLM prompt injection could cause the agent to:
  - Read `/etc/passwd`, `~/.ssh/id_rsa`, `~/.aws/credentials`
  - Write to `~/.bashrc`, `/etc/cron.d/`, any system file
- **Recommendation:** Add a configurable `allowed_paths` / `sandbox_dir` setting. Reject operations outside the sandbox by default. At minimum, block paths like `/etc/`, `~/.ssh/`, `~/.gnupg/`.

### [HIGH] Pervasive `shell=True` Usage
- **Locations (12 instances):**
  - `pico/tools/terminal.py:44,64`
  - `pico/tools/code_tools.py:290,406`
  - `pico/tools/detection/dataset.py:120`
  - `pico/tools/detection/evaluator.py:32`
  - `pico/tools/detection/exporter.py:29`
  - `pico/tools/detection/trainer.py:37`
  - `pico/tools/detection/config_gen.py:20`
  - `pico/tools/detection/inference.py:33`
  - `pico/tools/detection/annotation.py:32`
- **Description:** The LLM can craft arbitrary shell commands. While the `terminal` tool is *designed* for shell execution, the detection tools pass f-string commands through `shell=True`.
- **Impact:** Command injection if LLM-controlled values are interpolated into shell strings.
- **Recommendation:**
  1. For the terminal tool: add an allowlist/blocklist of dangerous commands (e.g., `rm -rf /`, `mkfs`, `dd`).
  2. For detection tools: use list-based `subprocess.run()` (no `shell=True`).
  3. Add audit logging for all executed commands.

### [MEDIUM] SSH Host Key Verification Disabled
- **Location:** `pico/tools/remote/ssh_client.py:98` — `paramiko.AutoAddPolicy()`
- **Description:** Accepts any SSH host key without verification. Vulnerable to man-in-the-middle attacks.
- **Recommendation:** Use `paramiko.RejectPolicy()` or `paramiko.WarningPolicy()`. Auto-add on first connect and verify on subsequent connections. Store known hosts in `~/.pico-agent/known_hosts`.

### [MEDIUM] SSH Passwords in Plaintext Config
- **Location:** `pico/tools/remote/ssh_client.py:59` — `srv_cfg.get("password")`
- **Location:** `config.example.yaml:18-22` — commented example shows `key_file` but not password
- **Description:** SSH passwords can be stored in plaintext in `~/.pico-agent/config.yaml`. The config file has no restricted permissions.
- **Recommendation:**
  1. Support `${SSH_PASSWORD}` env var references (already supported by `_resolve_env_vars`).
  2. Set config file permissions to 600 on creation.
  3. Document that passwords should use env var references.

### [LOW] SSRF via web_extract
- **Location:** `pico/tools/web_tools.py:91-143`
- **Description:** `web_extract()` follows redirects and can fetch arbitrary URLs, including internal network addresses (e.g., `http://169.254.169.254/` for cloud metadata).
- **Recommendation:** Add URL validation to block private IP ranges and localhost.

### [LOW] Exception Details Leaked to Users
- **Location:** Multiple — e.g., `pico/compression.py:136`, `pico/tools/registry.py:190`
- **Description:** Raw exception messages (including internal paths, stack details) are included in error responses returned to the LLM and potentially displayed to users.
- **Recommendation:** Log full exceptions internally but return sanitized error messages in tool responses.

### [OK] SQL Injection — Properly Mitigated
- **Location:** `pico/session.py` — All queries use `?` parameterized placeholders
- All 8 SQL queries use parameterized queries. No string formatting or f-strings in SQL. ✅

### [OK] YAML Deserialization — Safe
- **Location:** `pico/config.py:190` — `yaml.safe_load(f)`
- **Location:** `pico/tools/workflow_tools.py:49,405` — `yaml.safe_load(f)`
- All YAML loading uses `safe_load`. ✅

### [OK] API Key Not Logged
- API keys are never logged at any log level. Only model names and token counts are logged. ✅

---

## UX Findings

### [OK] CLI Help System — Well Designed
- **Location:** `pico/cli.py:529-582` — `_print_help()`
- Professional box-drawing header, organized sections (FLAGS, QUICK START, SHORTCUT COMMANDS, REPL SHORTCUTS, CONFIGURATION).
- Each subcommand has its own argparse parser with descriptive `--help`.

### [OK] Rich Output with Fallback
- **Location:** `pico/cli.py:42-98`
- All display functions try rich rendering and fall back to plain text. Good for CI/terminal compatibility.

### [LOW] No `--version` Flag
- **Description:** `pico-agent --version` is not supported. Users expect this from professional CLI tools.
- **Recommendation:** Add `--version` flag that reads from `pyproject.toml` version.

### [LOW] Manual Arg Parsing for Global Flags
- **Location:** `pico/cli.py:434-463`
- **Description:** Global flags are parsed with a manual `while` loop instead of argparse. The comment says "avoids argparse subparser issues" but this could miss edge cases and doesn't auto-generate help.
- **Recommendation:** Consider using argparse with `parse_known_args()` for global flags, then subparsers for commands.

### [LOW] REPL Lacks Session Context Display
- **Location:** `pico/cli.py:361`
- **Description:** The REPL prompt is just `>>> ` with no indication of the current session or model.
- **Recommendation:** Show abbreviated session ID or model name, e.g., `[gpt-4o] >>> `.

### [MEDIUM] Error Messages — Generally Good but Inconsistent
- CLI shortcut errors go to `stderr` with `print(f"Error: {err}", file=sys.stderr)` — good.
- Tool errors return JSON strings — the LLM interprets these, but CLI shortcuts parse JSON for display.
- **Inconsistency:** `_shortcut_search` and `_shortcut_download` route through `_single_shot()` (LLM), while `train`, `eval`, `clone` use direct tool dispatch. Users may get different error quality.

### [OK] Path Validation Before Execution
- **Location:** `pico/cli.py:175-192` — `_validate_dir()`, `_validate_file()`
- CLI shortcuts validate paths before dispatching to tools. Good UX.

---

## Documentation Findings

### [MEDIUM] README Config Example Mismatch
- **Location:** `README.md:168-179`
- **Description:** The config example shows:
  ```yaml
  model: gpt-4o
  api_base: https://api.openai.com/v1
  ```
  But the actual `config.py` expects:
  ```yaml
  model:
    provider: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
  ```
- **Recommendation:** Update README to match actual config schema, or point to `config.example.yaml`.

### [LOW] No CONTRIBUTING.md or Development Guide
- **Description:** No contributing guidelines, code style guide, or development setup instructions.
- **Recommendation:** Add a CONTRIBUTING.md with setup instructions, testing commands (`pytest`), and code style expectations (ruff is in dev deps).

### [LOW] No CHANGELOG
- **Description:** No changelog tracking version history.
- **Recommendation:** Add CHANGELOG.md or use conventional commits with auto-generation.

### [OK] README Structure
- Good architecture diagram, tool category table, dataset format table, download sources table.
- Installation options with optional dependency groups well documented.

### [LOW] Environment Variable Inconsistency
- **Location:** `README.md:151` — `PICO_API_BASE`
- **Location:** `pico/config.py:42` — `PICO_BASE_URL`
- **Description:** README uses `PICO_API_BASE` but config.py maps `PICO_BASE_URL`. The env var name in the README won't work.
- **Recommendation:** Use consistent naming. The correct one is `PICO_BASE_URL` per config.py.

---

## Prioritized Action Items

### P0 — Critical (Fix Before Release)

| # | Issue | Location | Effort |
|---|-------|----------|--------|
| 1 | Fix zip slip in `extractall()` — validate archive members | `dataset_tools.py:328,333` | 1h |
| 2 | Add filesystem sandbox (configurable `allowed_paths`) | `file_tools.py`, `config.py` | 4h |

### P1 — High (Fix Soon)

| # | Issue | Location | Effort |
|---|-------|----------|--------|
| 3 | Replace `shell=True` with list args in detection tools | 10 locations in `detection/` | 3h |
| 4 | Add command blocklist/allowlist for terminal tool | `terminal.py` | 2h |
| 5 | Fix SSH `AutoAddPolicy()` — use host key verification | `ssh_client.py:98` | 2h |
| 6 | Sanitize exception messages in tool responses | Multiple | 2h |

### P2 — Medium (Should Fix)

| # | Issue | Location | Effort |
|---|-------|----------|--------|
| 7 | Fix README config example mismatch | `README.md:168-179` | 0.5h |
| 8 | Fix `PICO_API_BASE` vs `PICO_BASE_URL` inconsistency | `README.md:151` | 0.5h |
| 9 | Remove duplicate `_paramiko_available()` | `remote/__init__.py:42-43` | 5min |
| 10 | Add public API to `ToolRegistry` instead of accessing `_tools` | `delegation.py:57`, `__init__.py:29` | 1h |
| 11 | Set config file permissions to 600 | `config.py:209` | 0.5h |
| 12 | Add SSRF protection to `web_extract` | `web_tools.py` | 1h |
| 13 | Add audit logging for all shell commands | `terminal.py`, `code_tools.py` | 1h |

### P3 — Low (Nice to Have)

| # | Issue | Location | Effort |
|---|-------|----------|--------|
| 14 | Add `--version` flag | `cli.py` | 0.5h |
| 15 | Extract rich helpers to `pico/display.py` | `cli.py` | 2h |
| 16 | Add REPL session/model context in prompt | `cli.py:361` | 0.5h |
| 17 | Consistent session title length constant | `agent.py:71,146` | 10min |
| 18 | Stop mutating `config.data` from CLI | `cli.py:488` | 0.5h |
| 19 | Add CONTRIBUTING.md | Project root | 1h |
| 20 | Add shell completion support | `cli.py` | 3h |

---

## Test Coverage Assessment

**Test files found:** 12 (conftest + 11 test modules)

| Module | Test File | Coverage Assessment |
|--------|-----------|-------------------|
| agent.py | test_agent.py | Good — simple response, tool call loop, max turns |
| cli.py | test_cli.py | Exists (not reviewed in detail) |
| config.py | test_config.py | Exists |
| session.py | test_session.py | Good — create, messages, list |
| compression.py | test_compression.py | Exists |
| llm.py | test_llm.py | Exists |
| memory.py | test_memory.py | Exists |
| registry.py | test_registry.py | Exists |
| file_tools.py | test_file_tools.py | Exists |
| terminal.py | test_terminal.py | Exists |
| web_tools.py | test_web_tools.py | Exists |

**Gaps:**
- No tests for security edge cases (path traversal, zip slip, command injection)
- No integration tests
- No tests for delegation.py
- No tests for remote/ tools (likely needs paramiko mocks)
- No tests for detection/ tools (likely needs ultralytics mocks)

---

## Summary

Pico Agent is a **solid v0.1.0** with clean architecture and good engineering practices. The most pressing concerns are security-related: archive extraction without validation, lack of filesystem sandboxing, and widespread `shell=True` usage. These should be addressed before any production deployment or public release.

The codebase is well-organized, has good test infrastructure, and demonstrates thoughtful design decisions (DI, ABC, Protocol, null-object pattern). With the security fixes applied, this would be a strong 8.5/10 codebase.
