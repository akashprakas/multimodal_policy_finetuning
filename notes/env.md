# Environment Design & Dynamics

## Episode Structure
- **Non-terminal success:** Episodes have a fixed 14-step horizon; reaching a goal does not terminate the episode early.
- **Dense step rewards:** Reward is collected at every step based on the Gaussian density around goal centers.
- **Evaluation criterion:** An episode is marked successful if the agent visits a goal region ($d < d_{\text{success}}$) at least once during the rollout.

## Design Consideration: Terminal vs. Continuing Rollouts
In the current continuing formulation, the agent continues collecting rewards after reaching a goal basin, encouraging it to remain within the goal region. A potential ablation is a terminal-on-success formulation:
- `terminated`: Reached goal threshold ($d < 0.30$).
- `truncated`: Reached horizon without goal contact.
- Value bootstrapping stops upon termination, masking out post-success reward accumulation.

Comparing the continuing formulation against a terminal-on-success formulation evaluates whether continuing rewards induce policy stability or mode-holding behaviors.