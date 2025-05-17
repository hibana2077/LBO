import math
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.optim.optimizer import Optimizer

class LaplaceBeltramiOptimizer(Optimizer):
    r"""Curvature-aware optimizer with directional sensitivity, dynamic modulation, and stagnation-escape.

    Key Features
    -------------
    * Adaptive curvature weighting based on gradient magnitude
    * Positive/negative direction-aware curvature modulation
    * Multi-step finite difference curvature estimation
    * AdamW + curvature-adaptive learning rate (hybrid update)
    * **Stagnation-escape**: Detects if recent *n* parameter displacements are too small
      (likely stuck in a flat region / local minimum).
      Attempts to perturb parameters in eight random orthogonal directions, selects the direction
      that reduces the loss, and then updates the parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        curvature_scale: float = 1.0,
        window_size: int = 3,
        # ------ Hyperparameters for stagnation detection and exploration ------
        stagnation_window: int = 5,
        stagnation_threshold: float = 1e-7,
        exploration_step: float = 0.1,  # Perturbation magnitude multiplied by lr
        weight_decouple: bool = True,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0: # Assuming the third beta is also in [0,1)
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        assert window_size >= 2, "window_size must be at least 2"
        assert stagnation_window >= 2, "stagnation_window must be at least 2"

        defaults: dict = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "curvature_scale": curvature_scale,
            "window_size": window_size,
            "stagnation_window": stagnation_window,
            "stagnation_threshold": stagnation_threshold,
            "exploration_step": exploration_step,
            "weight_decouple": weight_decouple,
        }
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ #
    #                         STATE INITIALISATION                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset(self):
        """Reset state (including gradient / parameter history)"""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["grad_history"] = []             # For curvature estimation
                state["param_delta_hist"] = []         # For stagnation detection

    # ------------------------------------------------------------------ #
    #                                STEP                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure = None) -> Optional[float]:
        """Perform a single optimization step. If stagnation is detected and a closure is provided, initiate exploration."""
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            step = group.get("step", 0) 
            # Initialize step counter in state if not present
            if step == 0:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['step'] = 0
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    # raise NoSparseGradientError(str(self)) # Changed to ValueError
                    raise ValueError("LaplaceBeltramiOptimizer does not support sparse gradients")


                state = self.state[p]
                # State initialization
                if len(state) == 0: # This should ideally be caught by the step check above or handled differently
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["grad_history"] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(group["window_size"] - 1)]
                    state["param_delta_hist"] = []
                
                state['step'] +=1
                current_step = state['step']


                beta1, beta2, _ = group["betas"] # Assuming the third beta is not used in AdamW part
                bias_correction1 = 1 - beta1 ** current_step
                bias_correction2 = math.sqrt(1 - beta2 ** current_step)

                lr = group["lr"]
                eps = group["eps"]
                window_size = group["window_size"]
                curvature_scale = group["curvature_scale"]
                stagn_win = group["stagnation_window"]
                stagn_thr = group["stagnation_threshold"]
                explore_step = group["exploration_step"] * lr

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad_history = state["grad_history"]
                delta_hist: List[torch.Tensor] = state["param_delta_hist"]

                # ------------------------------------------------------ #
                #  1. Update gradient history, calculate curvature, and adjust learning rate
                # ------------------------------------------------------ #
                grad_history.append(grad.clone())
                if len(grad_history) > window_size -1: # Ensure history does not exceed window_size -1 for prev grad
                    grad_history.pop(0)
                # state["grad_history"] = grad_history # Already a reference

                if len(grad_history) >= window_size -1 and window_size >1: # check window_size > 1
                    grad_prev = grad_history[0]
                    grad_curr = grad # Use current grad, not from history[-1] as it's just appended
                    # grad_diff = (grad_curr - grad_prev) / (window_size - 1) # Original
                    # Corrected finite difference: (current_grad - oldest_grad_in_window) / (num_intervals)
                    # num_intervals = (len(grad_history) -1) which is window_size - 2 if history has window_size-1 elements
                    # If grad_history stores window_size-1 elements. grad_prev is grad_history[0].
                    # grad_curr is the current grad.
                    # The time difference is (window_size -1) steps if grad_history[0] is from (t - (window_size-1))
                    # and current grad is from t.
                    # So grad_diff is (grad_curr - grad_prev) / ( (window_size-1) * dt ) where dt=1 step.
                    grad_diff = (grad_curr - grad_prev) / max(1, window_size - 1)


                    curvature_mask = grad_diff.sign() * grad.sign()
                    grad_norm = grad.abs().mean()
                    # curvature_norm = curvature_mask.abs().mean() # This is incorrect, curvature_mask is not curvature
                    curvature_val = grad_diff # This is the actual curvature approximation
                    curvature_norm = curvature_val.abs().mean()

                    curvature_weight = grad_norm / (grad_norm + curvature_norm + eps)
                    # curvature_term = curvature_scale * curvature_weight * curvature_mask # Original
                    # curvature_term should use curvature_val (grad_diff), not curvature_mask for scaling
                    # and then apply the mask for directionality
                    scaled_curvature = curvature_scale * curvature_weight * curvature_val
                    # Apply sign mask: only consider curvature if it aligns with gradient direction
                    # A positive curvature_mask means grad_diff and grad have same sign (accelerating in grad direction or decelerating against it)
                    # A negative curvature_mask means they have opposite signs (slowing down in grad direction or accelerating against it)
                    # The original paper might have a specific interpretation for curvature_mask.
                    # Let's stick to the original logic for now, assuming curvature_mask is intended.
                    curvature_term = curvature_scale * curvature_weight * curvature_mask

                    adaptive_lr = lr * (1 + curvature_term.clamp(-0.5, 0.5).mean())
                else:
                    adaptive_lr = lr

                # ------------------------------------------------------ #
                #  2. Stagnation detection: calculate displacement and maintain history
                # ------------------------------------------------------ #
                # Keep old parameters (for delta calculation)
                old_data = p.data.clone(memory_format=torch.preserve_format)

                # ------------------------------------------------------ #
                #  3. General AdamW-style update
                # ------------------------------------------------------ #
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["weight_decay"] > 0:
                    if group["weight_decouple"]:
                        p.add_(p, alpha=-group["weight_decay"] * lr)
                    else:
                        grad.add_(p, alpha=group["weight_decay"])

                step_size = adaptive_lr * (bias_correction2 / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size.item())

                # ------------------------------------------------------ #
                #  4. Update displacement history and check for stagnation
                # ------------------------------------------------------ #
                delta = (p.data - old_data).norm().item()
                delta_hist.append(delta)
                if len(delta_hist) > stagn_win:
                    delta_hist.pop(0)
                state["param_delta_hist"] = delta_hist

                stagnated = (
                    len(delta_hist) == stagn_win and max(delta_hist) < stagn_thr
                )

                # ------------------------------------------------------ #
                #  5. Exploration: if stagnated and closure is provided => try eight directions
                # ------------------------------------------------------ #
                if stagnated and closure is not None:
                    # Record current best loss
                    with torch.enable_grad():
                        current_loss = closure()
                    best_loss = current_loss
                    best_direction: Optional[torch.Tensor] = None
                    orig_data = p.data.clone(memory_format=torch.preserve_format)

                    for _ in range(8):
                        # Generate random orthogonal direction (+1 / -1) and perturb
                        direction = torch.randint_like(p, high=2, low=0, dtype=torch.float32).mul_(2).sub_(1)
                        p.data.copy_(orig_data + explore_step * direction)
                        with torch.enable_grad():
                            trial_loss = closure()
                        if trial_loss < best_loss:
                            best_loss = trial_loss
                            best_direction = direction.clone()
                        # Restore parameters, try the next direction
                        p.data.copy_(orig_data)

                    # If a better direction is found => apply perturbation and perform gradient update (without recalculating exp_avg)
                    if best_direction is not None:
                        p.add_(best_direction, alpha=explore_step)
                        # Clear delta_hist to avoid continuous triggering
                        state["param_delta_hist"] = []

        return loss