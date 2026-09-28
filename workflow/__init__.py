"""Experiment pipeline: sequential learning (Step 1), solving (Step 2); evaluation lives in evaluation/.

    run_sequential.py  CLI entry point
    pipeline.py        per-task orchestration (with_skill vs. baseline solves)
    learning.py        sequential learning over a curriculum of historical commits, with resume
    phases.py          single phases: learning iteration (attempt -> reflect) and solve
    output_layout.py   output directory layout
    config.py          constants and prompt paths
"""
