# -*- coding: utf-8 -*-
"""BAM actuator model for the Feetech **HD-1910-C001** as this robot drives it (2S, HLS map).

Drop-in sibling of `hl2915.py`: same voltage-controlled law, different servo.

    duty = clip(kp * (q_target - q) * error_gain * error_gain_ratio, current window, +-max_pwm)
    tau  = kt * V / R - kt^2 * dq / R        (+ the BAM model's friction terms)

Numbers below are **from the supplier datasheet** (feetech.cn/510257, edition A/0, 2026-09-07),
so they are a starting point and a set of bounds for a bench fit -- not a fit. Run
`training/bam/fit_hl2915.py` with this module registered (the file is servo-agnostic; only the
`--actuator` name changes) against pendulum logs from an HD-1910, and it writes the JSON.

| datasheet | 4.8 V | 6.0 V | 7.4 V |
| --- | --- | --- | --- |
| no-load speed (s/60 deg) | 0.137 | 0.109 | 0.088 |
| stall torque (kgf.cm) | 9 | 12 | 15 |
| stall current (A) | 1.2 | 1.6 | 2.0 |
| rated torque / current | 2.2 / 0.5 | 3.0 / 0.69 | 3.7 / 0.9 |

Gear 1:320, backlash <= 0.5 deg, 12-bit magnetic encoder, coreless motor, 21 +- 2 g bare
(22.8 g with horn). Working range 4-8.4 V -- the 2S pack, and 8.4 V is the top of the range with
no margin, which is why nothing here models a 3S.

Register it before loading models or fitting::

    import hd1910  # noqa: F401  (adds "hd1910" to bam.actuators.actuators)
"""
from __future__ import annotations

import numpy as np

from bam.actuator import VoltageControlledActuator
from bam.actuators import actuators
from bam.parameter import Parameter
from bam.testbench import Pendulum

# Fitted error gain is error_gain * error_gain_ratio; see hl2915.py's docstring for why the
# bench value (Kp 16 holds ~0.1 N.m at ~5 deg of error) transfers.
ERROR_GAIN = 0.05
# The register-44 cap robotd actually writes. **Read off the robot on 2026-11-15**: its
# `[bus] current_limit_ma = 1300` with `current_ma_per_count = 0.65` (2000 counts). Keep this
# equal to that value or the fit/inference is for a different servo setting. HD-1910's rated
# current at 7.4 V is 0.9 A and its stall is 2.0 A, so 1.3 A is the robot's choice, not the spec.
DEFAULT_MAX_CURRENT = 1.3


class HD1910Actuator(VoltageControlledActuator):
    def __init__(self, testbench_class=Pendulum):
        super().__init__(
            testbench_class,
            # 2S nominal. robotd feeds 6.3-8.4 V; the training cfg randomizes vin_range over it.
            vin=7.4,
            kp=16,
            error_gain=ERROR_GAIN,
            max_pwm=1.0,
            max_current=DEFAULT_MAX_CURRENT,
        )

    def load_log(self, log: dict):
        super().load_log(log)
        if "current_limit_ma" in log:
            self.max_current = log["current_limit_ma"] / 1000.0

    def initialize(self):
        # Datasheet @ 7.4 V: stall 15 kgf.cm = 1.47 N.m at 2.0 A, KT quoted as 7.5 kgf.cm/A
        # = 0.74 N.m/A. So R ~ 7.4 / 2.0 = 3.7 ohm. Bounds are wide: the fit decides.
        self.model.kt = Parameter(0.74, 0.2, 2.5)
        self.model.R = Parameter(3.7, 1.0, 15.0)
        self.model.armature = Parameter(0.002, 0.0001, 0.05)
        self.model.error_gain_ratio = Parameter(1.8, 0.1, 10.0)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def compute_control(self, q_target, q, dq, dt):
        duty = (q_target - q) * self.kp * self.error_gain * self.model.error_gain_ratio.value
        if self.max_current is not None:
            back_emf = self.model.kt.value * dq
            span = self.model.R.value * self.max_current / self.vin
            centre = back_emf / self.vin
            duty = self.backend.clamp(duty, centre - span, centre + span)
        duty = self.backend.clamp(duty, -self.max_pwm, self.max_pwm)
        return self.vin * duty

    def to_mujoco(self):
        saved = self.error_gain
        self.error_gain = saved * self.model.error_gain_ratio.value
        try:
            super().to_mujoco()
        finally:
            self.error_gain = saved


actuators["hd1910"] = lambda: HD1910Actuator(Pendulum)
