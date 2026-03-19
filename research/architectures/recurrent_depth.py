from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentDepthConfig:
    enabled: bool
    prelude_layers: int
    core_layers: int
    steps: int
    backprop_steps: int
    coda_layers: int
    eval_steps: int
    state_init: str
    input_injection: str

    def validate(self, *, num_layers: int) -> None:
        if self.prelude_layers < 0:
            raise ValueError(f"RECURRENT_PRELUDE_LAYERS must be >= 0, got {self.prelude_layers}")
        if self.core_layers < 1:
            raise ValueError(f"RECURRENT_CORE_LAYERS must be >= 1, got {self.core_layers}")
        if self.steps < 1:
            raise ValueError(f"RECURRENT_STEPS must be >= 1, got {self.steps}")
        if self.backprop_steps < 1:
            raise ValueError(f"RECURRENT_BACKPROP_STEPS must be >= 1, got {self.backprop_steps}")
        if self.backprop_steps > self.steps:
            raise ValueError(
                "RECURRENT_BACKPROP_STEPS must be <= RECURRENT_STEPS; "
                f"got RECURRENT_BACKPROP_STEPS={self.backprop_steps}, RECURRENT_STEPS={self.steps}"
            )
        if self.coda_layers < 0:
            raise ValueError(f"RECURRENT_CODA_LAYERS must be >= 0, got {self.coda_layers}")
        if self.eval_steps < 1:
            raise ValueError(f"RECURRENT_EVAL_STEPS must be >= 1, got {self.eval_steps}")
        effective_depth = self.prelude_layers + self.core_layers * self.steps + self.coda_layers
        if effective_depth != num_layers:
            raise ValueError(
                "RECURRENT_PRELUDE_LAYERS + RECURRENT_CORE_LAYERS * RECURRENT_STEPS + "
                "RECURRENT_CODA_LAYERS must equal NUM_LAYERS; "
                f"got {self.prelude_layers} + {self.core_layers} * {self.steps} + {self.coda_layers} = "
                f"{effective_depth}, NUM_LAYERS={num_layers}"
            )
        if self.state_init not in {"like_init", "zero", "normal"}:
            raise ValueError(
                "RECURRENT_STATE_INIT must be one of like_init, zero, normal; "
                f"got {self.state_init!r}"
            )
        if self.input_injection != "linear_concat":
            raise ValueError(
                "RECURRENT_INPUT_INJECTION must be 'linear_concat' in recurrent_depth_v1; "
                f"got {self.input_injection!r}"
            )
