Success is non-terminal.
Episodes have a fixed 14-step horizon.
Reward is collected at every step.
Evaluation records whether a goal was reached at least once.


## May be possible ablation

terminated = success
truncated = horizon reached without termination
PPO must stop bootstrapping after terminated
successful lanes must be masked or reset correctly
reward after success must not be accumulated

then compare

Current continuing environment
versus
terminal-on-success environment