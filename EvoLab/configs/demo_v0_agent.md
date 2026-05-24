# Demo V0 Meta-Agent

Dispatch the task through the configured subagents:

1. Run `solver` first. It should read `inputs/catalyst_evidence.txt` and make the initial recommendation.
2. Run `reviewer` second. It should check the solver recommendation against the same evidence.
3. Finish the task after the reviewer completes.

Return exactly one `DispatchDecision` JSON object per meta-agent step.
