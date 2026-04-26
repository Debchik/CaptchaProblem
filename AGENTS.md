# AGENTS.md

## Objective
Win this ML competition by improving the target metric through honest validation, disciplined experiments, and reproducible modeling.

## Ground truth
- Main notebook: `notebooks/competition_main.ipynb`
- Experiment log: `reports/experiment_log.md`
- Source code: `src/`
- Outputs: `outputs/`

## Hard rules
- Every experiment starts with a hypothesis.
- Do not mix many major changes in one experiment.
- Log every meaningful result.
- Do not trust leaderboard-only improvements.
- Check leakage before celebrating any unusually high score.
- Keep notebook cells and markdown clean.
- Add short conclusions after each major analysis block.

## Validation
- Use the project's fixed validation scheme unless a validation audit justifies a change.
- Any validation change must be explained and compared.

## Feature engineering
- Keep only empirically useful features.
- Flag leakage-prone features explicitly.

## Modeling
- Start simple, then increase complexity.
- Compare models fairly under the same CV.

## Subagents
Use subagents only for clearly separable tasks.
Each subagent must return:
- hypothesis
- method
- code changes
- measured result
- conclusion
- recommendation