# -*- coding: utf-8 -*-
"""`fit_hl2915.py` for the HD-1910: run a BAM module with the HD-1910 actuator registered.

Same bench, same BAM tooling -- only the actuator name changes, and with it the datasheet-derived
starting point and bounds in `hd1910.py`.

    uv run python fit_hd1910.py bam.process --raw raw/hd-m105 --logdir processed --dt 0.005
    uv run python fit_hd1910.py bam.fit --logdir processed --actuator hd1910 --model m4 \
        --method cmaes --trials 2000 --workers 16 --output params/hd1910/m4.json
    uv run python fit_hd1910.py bam.plot --logdir processed --sim --params params/hd1910/m4.json

Bench notes: 7.4 V limited supply, never 3S; over-current and over-voltage
protection are OFF by default on this servo, so the supply's current limit is the only hard
protection.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd1910  # noqa: E402,F401  registers "hd1910"

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
