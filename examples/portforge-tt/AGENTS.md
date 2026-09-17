# PortForge-TT agent context

Follow the root AGENTS.md. Read PROJECT.md when starting here; read SETUP.md
when running commands. HISTORY.md is an archive for targeted lookup, not
required startup context. The proposals in assets/ define the research scope.

Keep this directory as a coordination hub: agent instructions, short current
state, shared setup, and historical notes only. Each runnable experiment/loop
must live in its own sibling examples/portforge-tt-<experiment>/ directory,
with its code, tests, runbook, and ignored runs/ artifacts. Do not put executable
loops, tests, generated files, or run logs in this hub.

Keep PROJECT.md short: current state, decisions, and next tasks. Put detailed
investigations in HISTORY.md. Link to experiment directories instead of copying
their implementation or logs here. Preserve source snapshots and result records
when moving experiments. Existing experiments: ../portforge-tt-smoke/,
../portforge-tt-rmsnorm/, ../portforge-tt-reference/, and ../portforge-tt-nllb/.

All four Blackhole cards are authorized by the user, including parallel use
(September 14, 2026). Select each worker's card by full PCI BDF before Python
starts; allow one evaluation per card at a time. Ray labels do not enforce
isolation. Do not reset cards, flash firmware, or change shared host services.

Agents may propose candidates and development tests. The trusted evaluator
owns the oracle, acceptance thresholds, timing protocol, and held-out cases.
Never weaken those to accept a candidate. Keep credentials out of source and
artifacts. Unrestricted generated-code execution needs isolation first.
