# Autonomous porting harness

The controller, evaluation policy and reference implementation are trusted code.
Never copy an earlier PortForge implementation into an agent workspace. Seed only
`agent/`, the public task contract, and the selected model's identity. Preserve
failed trials as well as successful ones. Do not change acceptance thresholds in
response to candidate failures. Changes to this harness require its unit tests
and an isolation preflight before another fresh campaign.

The agent's actual instructions live in `agent/AGENTS.md`. Those instructions do
not authorize it to edit this harness or the evaluation policy.
