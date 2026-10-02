# Usage guide

[Back to the project overview](../README.md)

Run the commands below from the repository root after following the [installation and quickstart](../README.md#quickstart).

## Configuration

Everything is configured through environment variables (a `.env` file in the repository root is read
automatically; variables already exported in your shell take precedence):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | API key used by the agent (Claude Code) and by the judge |
| `ANTHROPIC_BASE_URL` | Anthropic API | Only needed for a proxy or gateway |
| `LTC_MODEL` | `claude-sonnet-4-6` | Model for all agent phases (`--model` overrides it) |
| `LTC_MAX_TOKENS` | `32768` | Max output tokens per LLM call |
| `LTC_THINKING` | `disabled` | Extended thinking in agent phases (`disabled` as in the paper, or `adaptive`) |
| `LTC_CONCURRENCY` | `4` | Test tasks solved in parallel (`--concurrency` overrides it) |
| `LTC_JUDGE_MODEL` | `LTC_MODEL` | Pairwise judge, stage 1 (analysis) |
| `LTC_JUDGE_EXTRACTOR_MODEL` | `LTC_JUDGE_MODEL` | Pairwise judge, stage 2 (JSON extraction) |
| `LTC_SANDBOX_ROOT` | `<tmp>/ltc_sandbox` | Where per-phase working directories are created |
| `LTC_SANDBOX_KEEP` | unset | Set to `1` to keep working directories for debugging |
| `LTC_CLAUDE_CONFIG_DIR` | `<LTC_SANDBOX_ROOT>/claude_config` | Private Claude Code config dir used by the agent |

Agent sessions are isolated from your personal Claude Code setup: they use a private config directory and load
no user/project settings, `CLAUDE.md` files, memory, plugins, skills or MCP servers.

## Running on the released benchmarks

The released JSONL files inline every diff but not the repository snapshots (several GB). Build them with
`scripts/s1_prepare_benchmark.py`, which clones the upstream repositories and runs `git archive` at the right
revision (the parent `<sha>~1` for learning commits and Learning-to-Commit test tasks, the instance base
commit for SWE-bench Pro tasks):

```bash
# quick check: one repository, one test task, one learning commit (~2 minutes, ~0.6 GB on disk)
python scripts/s1_prepare_benchmark.py --repos DeepSpeed --limit 1 --max-learning-commits 1

# full Learning to Commit benchmark (all 5 repositories)
python scripts/s1_prepare_benchmark.py

# SWE-bench Pro subset
python scripts/s1_prepare_benchmark.py --benchmark data/swebench_pro_tasks.jsonl
```

Outputs go to `data/prepared/<benchmark name>/` (`repos/`, `snapshots/`, `diffs/`, `benchmark_tasks.jsonl`;
all git-ignored). Then run the pipeline on one repository at a time, e.g.:

```bash
python scripts/s1_prepare_benchmark.py --repos DeepSpeed --out-dir data/prepared/deepspeed
python workflow/run_sequential.py \
    --input-file data/prepared/deepspeed/benchmark_tasks.jsonl \
    --signature l2c-deepspeed --run-baseline
```

All test tasks of a repository share the same 10 learning commits, so learning runs once per repository and
the resulting skills are reused for all its test tasks. Each SWE-bench Pro task has its own 3 learning
commits. Real repositories are large: expect tens of agent steps per phase and budget accordingly (use
`--max-items` / `--max-learning-commits` for small trial runs). Everything is cached on disk; rerunning the
same command resumes where it stopped.

Useful options of `workflow/run_sequential.py`:

| Option | Meaning |
|---|---|
| `--run-baseline` | Also solve each task without skills |
| `--skill-dir DIR` | Skip learning and solve with an existing skill directory |
| `--learning-mode observe_only` | Ablation: no blind attempt, the agent studies the oracle diff directly |
| `--baseline-variant forced` | Forced-exploration baseline: exploration-heavy prompt and at least `--forced-min-steps` (default 40) tool steps, written to `02_resolution/baseline_forced/` |
| `--max-items N`, `--max-learning-commits N` | Partial runs |

## Evaluation

```bash
python evaluation/programmatic_metrics.py --exp-dir output/<signature>/<model> --tasks-file <benchmark_tasks.jsonl>
python evaluation/pairwise_judge.py       --exp-dir output/<signature>/<model> --tasks-file <benchmark_tasks.jsonl>
```

- **File IoU**: Jaccard overlap between the files the agent changed and the files the oracle commit changed.
- **Trajectory steps**: number of LLM responses that issued tool calls while solving.
- **Line deviation**: `(agent_lines − oracle_lines) / max(1, oracle_lines)` over added + removed lines,
  reported as a signed mean and as a mean absolute value.
- **Pairwise judge**: the two patches are shown as anonymous developers A and B, with the task and the oracle
  diff as reference, and compared on scope alignment, logic similarity, redundancy/hallucination and code
  style; four rounds per task in ABBA order. Each judgement is a free-form analysis followed by a separate JSON
  extraction call (`system_prompts/judge_pairwise.txt`).

Use `--variants with_skill,baseline_forced` (metrics) or `--baseline-variant baseline_forced` (judge) to
evaluate the forced-exploration baseline.

## Output layout

```
output/<signature>/<model>/
├── 01_onboarding/curriculum_<key>/
│   ├── iter_00_<sha>/
│   │   ├── attempt_trajectory.{json,md}   blind attempt
│   │   ├── attempt_changes.diff           the attempt's patch
│   │   ├── reflect_trajectory.{json,md}   attempt + reflection (same session)
│   │   ├── skills/                        skills after this commit
│   │   └── summary.json
│   ├── iter_01_<sha>/ ...
│   └── final_skills/
├── 02_resolution/
│   ├── with_skill/<task_id>/{trajectory.json, trajectory.md, changes.diff, summary.json}
│   ├── baseline/<task_id>/...
│   └── task_records/<task_id>.json, index.jsonl
└── 03_evaluation/
    ├── programmatic.json
    └── pairwise/{judgments.jsonl, summary.json}
```

`trajectory.json` stores the SDK message stream in an OpenAI-style message list (`system`, `user`,
`assistant` with `tool_calls`, `tool`), plus the session id, number of turns, usage and cost.

## Building a benchmark from your own repository

`data_construction/` contains the five-stage pipeline used to build the benchmark (all LLM calls use the same
`ANTHROPIC_API_KEY`):

```bash
cd data_construction
python s1_scan_commits.py --repo-dir /path/to/repo --max-commits 500      # prefilter + LLM suitability scan
python s2_cluster.py --input runs/<repo>/s1_scanned.jsonl                 # derive 4-8 development categories
python s3_tag.py --input runs/<repo>/s1_scanned.jsonl \
    --categories runs/<repo>/s2_categories.json --repo-dir /path/to/repo  # tag commits
python s4_sample.py --input runs/<repo>/s3_tagged.jsonl --n-learn 10 --n-test 10
python s5_build.py --input runs/<repo>/benchmark_train10_test10.jsonl --repo-dir /path/to/repo
```

`s5_build.py` writes a `benchmark_tasks.jsonl` (with `diffs/` and `snapshots/`) that `workflow/run_sequential.py`
accepts directly. The oldest 80% of the scanned commits form the learning pool and the newest 20% the test pool,
so all learning commits predate all test commits.

## Execution flow

```
benchmark_tasks.jsonl
        │
        ▼
Step 1  onboarding (per curriculum of learning commits, sequential)
        for each learning commit c_t:
            workspace = snapshot of c_t's parent, skills = skills after c_{t-1}
            attempt:  "Implement this change: <task description>"         (blind)
            reflect:  same session + oracle diff → update SKILL.md & co.   (contrastive)
        → 01_onboarding/curriculum_<key>/final_skills/
        │
        ▼
Step 2  resolution (per test task, in parallel)
            with_skill: skills copied next to the workspace, solve the task
            baseline:   same task, no skills          (--run-baseline)
        → 02_resolution/{with_skill,baseline}/<task_id>/changes.diff
        │
        ▼
Step 3  post-run evaluation
            evaluation/programmatic_metrics.py   File IoU, steps, line deviation
            evaluation/pairwise_judge.py         A/B judge with the oracle as reference
```

## Repository layout

```
.
├── CITATION.cff                            preferred paper citation
├── docs/                                   usage guide and paper figure
├── data/
│   ├── learning_to_commit_benchmark.jsonl   50 test tasks (5 repos × 10), diffs inlined
│   └── swebench_pro_tasks.jsonl             50 SWE-bench Pro tasks with 3 learning commits each
├── scripts/
│   └── s1_prepare_benchmark.py              clone repos, build snapshots/diffs, write benchmark_tasks.jsonl
├── examples/toy/
│   └── build_toy_benchmark.py               tiny synthetic repo: 2 learning commits + 1 test commit
├── workflow/
│   ├── run_sequential.py                    main entry point (Step 1 + Step 2)
│   ├── learning.py                          sequential learning with resume
│   ├── phases.py                            attempt/reflect and solve phases
│   ├── pipeline.py                          per-task orchestration
│   ├── output_layout.py, config.py
├── modules/
│   ├── agent.py                             Claude Agent SDK session + trajectory recording
│   ├── sandbox.py                           per-phase working directory (snapshot + git base commit)
│   ├── llm.py                               plain Messages API calls (judge, data construction)
│   ├── prompt_builder.py, helpers.py, trajectory_to_md.py, config.py
├── evaluation/
│   ├── programmatic_metrics.py
│   └── pairwise_judge.py
├── system_prompts/                          prompts for every phase (+ data_construction/)
└── data_construction/                       optional S1–S5 pipeline to build a benchmark from your own repo
```
