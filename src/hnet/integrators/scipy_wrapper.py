"""Scipy solve_ivp wrapper — default inference integrator."""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from hnet._protocols import HamiltonianModelProtocol


class ScipyIntegrator:
    """Wrap a learned model as a scipy solve_ivp vector field.

    Uses RK45 by default. Tight tolerances (rtol=1e-6, atol=1e-9) give
    accurate trajectories that reveal the true long-time energy conservation
    behavior of the learned Hamiltonian (rather than integrator error).

    Parameters
    ----------
    method : str
        scipy ODE solver method. Default "RK45".
    rtol : float
    atol : float
    device : str or torch.device
    """

    def __init__(
        self,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        device: str | torch.device = "cpu",
    ) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.device = torch.device(device) if isinstance(device, str) else device

    def integrate(
        self,
        model: HamiltonianModelProtocol,
        z0: np.ndarray,
        t_span: tuple[float, float],
        n_steps: int = 1000,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate model from z0 over t_span.

        Parameters
        ----------
        model : any object with vector_field(z: Tensor) -> Tensor
        z0 : np.ndarray, shape (state_dim,)
        t_span : (t0, tf)
        n_steps : number of output time points

        Returns
        -------
        t_eval : np.ndarray, shape (n_steps,)
        Z      : np.ndarray, shape (n_steps, state_dim)
        """
        z0 = np.asarray(z0, dtype=float)
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        def vf(t: float, z: np.ndarray) -> np.ndarray:
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.enable_grad():
                dz = model.vector_field(z_t, create_graph=False)
            return dz.detach().cpu().numpy().flatten()

        sol = solve_ivp(
            vf,
            t_span,
            z0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"ScipyIntegrator failed: {sol.message}")
        return sol.t, sol.y.T
