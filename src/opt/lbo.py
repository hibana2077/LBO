import math
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

class LaplaceBeltramiOptimizer(BaseOptimizer):
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
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999, 0.9),
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
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, "eps")
        self.validate_non_negative(curvature_scale, "curvature_scale")
        assert window_size >= 2, "window_size must be at least 2"
        assert stagnation_window >= 2, "stagnation_window must be at least 2"

        defaults: DEFAULTS = {
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

    def __str__(self) -> str:
        return "ImprovedLaplaceBeltrami"

    # ------------------------------------------------------------------ #
    #                         STATE INITIALISATION                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset(self):
        """Reset state (including gradient / parameter history)"""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["grad_history"] = []             # For curvature estimation
                state["param_delta_hist"] = []         # For stagnation detection

    # ------------------------------------------------------------------ #
    #                                STEP                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        """Perform a single optimization step. If stagnation is detected and a closure is provided, initiate exploration."""
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            step = group.get("step", 0) + 1
            group["step"] = step

            beta1, beta2, _ = group["betas"]
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = math.sqrt(1 - beta2 ** step)

            lr = group["lr"]
            eps = group["eps"]
            window_size = group["window_size"]
            curvature_scale = group["curvature_scale"]
            stagn_win = group["stagnation_window"]
            stagn_thr = group["stagnation_threshold"]
            explore_step = group["exploration_step"] * lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                # Initialize state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["grad_history"] = [torch.zeros_like(p) for _ in range(window_size - 1)]
                    state["param_delta_hist"] = []

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad_history = state["grad_history"]
                delta_hist: List[torch.Tensor] = state["param_delta_hist"]

                # ------------------------------------------------------ #
                #  1. Update gradient history, calculate curvature, and adjust learning rate
                # ------------------------------------------------------ #
                grad_history.append(grad.clone())
                if len(grad_history) > window_size - 1:
                    grad_history.pop(0)
                state["grad_history"] = grad_history

                if len(grad_history) >= window_size - 1:
                    grad_prev = grad_history[0]
                    grad_curr = grad_history[-1]
                    grad_diff = (grad_curr - grad_prev) / (window_size - 1)

                    curvature_mask = grad_diff.sign() * grad.sign()
                    grad_norm = grad.abs().mean()
                    curvature_norm = curvature_mask.abs().mean()
                    curvature_weight = grad_norm / (grad_norm + curvature_norm + eps)
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