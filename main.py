#!/usr/bin/env python3
"""


Goal of this project:
  • Start with the cart at position y = -2 m
  • Move it near y = 0 m (the target)
  • Calm down the swing of the hanging load by the end (~6 s into the move)
  • Do all that without using crazy force spikes

What this script does:
  • builds a linear cart+pendulum (crane + load) model
    - assumes small swing angles (linearized model)
  • converts it to discrete time
  • solves one Model Predictive Control (MPC) optimization
    - includes a limit on the cart force
    - includes a smoothness penalty so the motion is not jerky
    - solved once as an "open-loop" plan (we compute the full move from start to finish)
      NOTE: we do NOT update the controller step-by-step with feedback here
  • plots position / velocity / swing angle / control force
  • creates an animation GIF of the motion
  • prints simple performance numbers (settling time, peak force, etc.)

Dependencies:
  pip install numpy scipy cvxpy osqp matplotlib pillow
"""

from typing import Optional
import os
import numpy as np
import cvxpy as cp

import matplotlib
try:
    matplotlib.use("TkAgg")  # helps on Windows; safe to ignore if it warns
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from scipy.linalg import expm


# ---------------------------------------------------------------------
# Model + discretization
# ---------------------------------------------------------------------
def crane_pendulum_matrices(m1: float = 10.0, m2: float = 2.0):
    """
    Continuous-time linearized model (assumes small swing angles).

    State vector:
        x = [ y , v , theta , q ]^T
            y     = cart horizontal position [m]
            v     = cart horizontal velocity [m/s]
            theta = load swing angle [rad] (0 = hanging straight down)
            q     = theta rate [rad/s] (angular speed of the swing)

    Input:
        u = horizontal force applied to the cart [N]

    Dynamics:
        y_dot     = v
        v_dot     = eps * theta + u        (cart accelerates from its force u,
                                            plus coupling from the swinging mass)
        theta_dot = q
        q_dot     = -theta - u             (swing reacts to cart motion)

    where eps = m2 / (m1 + m2)

    NOTE:
      • This is a linearized model of a crane + hanging mass.
      • Valid for relatively small swing angles (theta small).
    """
    eps = m2 / (m1 + m2)

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],   # y_dot     = v
        [0.0, 0.0, eps,  0.0],  # v_dot     = eps*theta + u
        [0.0, 0.0, 0.0,  1.0],  # theta_dot = q
        [0.0, 0.0,-1.0,  0.0],  # q_dot     = -theta - u
    ])

    B = np.array([
        [0.0],   # y_dot doesn't depend directly on u
        [1.0],   # v_dot gets +u
        [0.0],   # theta_dot doesn't depend directly on u
        [-1.0],  # q_dot gets -u
    ])

    return A, B


def c2d_exact(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Exact zero-order-hold discretization:
        x[k+1] = Ad * x[k] + Bd * u[k]

    This gives the *discrete-time* model used by the MPC.
    """
    n, m = A.shape[0], B.shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B
    Md = expm(aug * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

# ---------------------------------------------------------------------
# MPC solve (single finite horizon, open-loop)
# ---------------------------------------------------------------------
def solve_mpc(
    m1=10.0,
    m2=2.0,
    dt=0.1,
    T=7.0,
    t_final=6.2,
    y0=-2.0,   # initial cart position [m]
    v0=0.0,    # initial cart velocity [m/s]
    th0=0.0,   # initial swing angle theta [rad]
    q0=0.0,    # initial angular rate theta_dot [rad/s]
    u_max=5.0, # max allowed force [N]
):
    """
    We solve one quadratic program for the full move.
    This is an "open-loop" MPC plan:
      • We compute the entire force profile once from start to finish.
      • We do NOT re-solve at every time step with new measurements.

    Decision variables:
      X[:, k]  = predicted state at step k
                 [y, v, theta, q]
      U[k]     = predicted cart force at step k [N]

    Subject to:
      X[:,k+1] = Ad * X[:,k] + Bd * U[k]   (system dynamics)
      |U[k]| <= u_max                      (force limit)

    Cost we minimize:
      1) keep position close to target and keep swing small
      2) end in a calm state near t_final
      3) keep force small
      4) keep force changes smooth (no jerks / no sudden hits)

    We do this with cvxpy + OSQP (a QP solver).
    """

    # Discretize continuous model
    A, B = crane_pendulum_matrices(m1, m2)
    Ad, Bd = c2d_exact(A, B, dt)

    # Horizon length in steps
    N = int(round(T / dt))
    n = Ad.shape[0]

    # We'll use Bd as a flat vector (shape (4,))
    b = Bd.flatten()

    # Build a time-dependent weight w[k] that becomes strong near t_final
    # This makes the controller care more about being calm and settled
    # around t_final.
    kf = int(round(t_final / dt))  # index of "we really want to be settled"
    w = np.zeros(N + 1)
    k0 = max(0, kf - int(0.7 / dt))  # start ramp about last 0.7 seconds
    for k in range(k0, N + 1):
        denom = max(1, (kf - k0))
        w[k] = min(1.0, (k - k0) / denom)
    w[kf:] = 1.0  # after t_final, weight stays high

    # Cost tuning matrices / scalars
    Q   = np.diag([25.0, 5.0, 12.0, 2.0])          # penalize states during the move
    Qf  = np.diag([600.0, 150.0, 400.0, 80.0])     # penalize final state
    ru  = 1e-3                                     # penalize large force
    rdu = 1e-2                                     # penalize fast changes in force

    # Decision variables in cvxpy
    X = cp.Variable((n, N + 1))   # [y, v, theta, q] predictions
    U = cp.Variable(N)            # force predictions [N]

    # Initial condition
    x_init = np.array([y0, v0, th0, q0])
    cons = [X[:, 0] == x_init]

    # Dynamics and input constraints for all steps
    for k in range(N):
        cons += [X[:, k+1] == Ad @ X[:, k] + b * U[k]]
        cons += [cp.abs(U[k]) <= u_max]

    # Build the objective J
    #  - running state cost
    stage_cost = sum(w[k] * cp.quad_form(X[:, k], Q) for k in range(N))

    #  - terminal cost at the end of horizon
    term_cost  = cp.quad_form(X[:, N], Qf)

    #  - force "energy" cost
    energy_cost = ru * cp.sum_squares(U)

    #  - smoothness cost on ∆u (this tries to avoid jerky commands)
    dU = U[1:] - U[:-1]
    smooth_cost = rdu * cp.sum_squares(dU)

    J = stage_cost + term_cost + energy_cost + smooth_cost

    # Solve QP
    prob = cp.Problem(cp.Minimize(J), cons)
    prob.solve(
        solver=cp.OSQP,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=20000,
        warm_start=True,
    )

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("MPC optimization failed: " + prob.status)

    # Time vector [s]
    t = np.arange(N + 1) * dt
    return t, X.value, U.value


# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
def plot_response(t, X, U, title="MPC response (m2 = 2 kg)"):
    """
    Plot the main signals:
      • cart force u(t)
      • cart speed v(t)
      • cart position y(t)
      • swing angle θ(t) and angular rate θ̇(t)
    """
    y, v, th, q = X

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    axs[0].plot(t[:-1], U, lw=2)
    axs[0].set_ylabel("Cart force u [N]")
    axs[0].grid(True)

    axs[1].plot(t, v, lw=2)
    axs[1].set_ylabel("Cart speed v [m/s]")
    axs[1].grid(True)

    axs[2].plot(t, y, lw=2)
    axs[2].set_ylabel("Cart position y [m]")
    axs[2].grid(True)

    axs[3].plot(t, th, lw=2, label="swing angle θ [rad]")
    axs[3].plot(t, q,  lw=2, ls="--", label="angular rate θ̇ [rad/s]")
    axs[3].set_ylabel("Swing states")
    axs[3].set_xlabel("Time [s]")
    axs[3].grid(True)
    axs[3].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------
def animate_motion(
    t,
    X,
    m1=10.0,
    m2=2.0,
    L=0.8,
    save_path: Optional[str] = "mpc_anim.gif",
    speed: float = 1.0,
    show_window: bool = False,
):
    """
    Simple cart + pendulum animation.
    Saves a GIF by default.

    This animation is what I show in the presentation.
    """
    y, v, theta, q = X
    y = np.asarray(y).ravel()
    theta = np.asarray(theta).ravel()

    # Cart and bob geometry
    x_cart = y
    z_cart = np.zeros_like(y)
    x_bob = x_cart + L * np.sin(theta)
    z_bob = z_cart - L * np.cos(theta)

    # Plot limits
    x_min = float(np.min(y) - L - 0.5)
    x_max = float(np.max(y) + L + 0.5)
    z_min, z_max = -L - 0.5, 0.6

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Cart + Pendulum MPC (m1={m1} kg, m2={m2} kg)")

    # Rail
    ax.plot([x_min, x_max], [0, 0], "k-", lw=1, alpha=0.6)

    # Cart (rectangle)
    cart_w, cart_h = 0.25 * L, 0.15 * L
    cart_patch = patches.Rectangle(
        (x_cart[0] - cart_w/2, 0.0),
        cart_w,
        cart_h,
        linewidth=1.5,
        edgecolor="k",
        facecolor="tab:blue",
        zorder=3,
    )
    ax.add_patch(cart_patch)

    # Pendulum 
    (rod_line,) = ax.plot(
        [x_cart[0], x_bob[0]],
        [0.0,       z_bob[0]],
        color="tab:orange",
        lw=2,
    )
    bob_patch = patches.Circle(
        (x_bob[0], z_bob[0]),
        0.06 * L,
        ec="k",
        fc="tab:orange",
        zorder=4,
    )
    ax.add_patch(bob_patch)

    # Time label in corner
    time_text = ax.text(0.02, 0.92, f"t = {t[0]:.2f} s", transform=ax.transAxes)

    def init():
        cart_patch.set_xy((x_cart[0] - cart_w/2, 0.0))
        rod_line.set_data([x_cart[0], x_bob[0]], [0.0, z_bob[0]])
        bob_patch.center = (x_bob[0], z_bob[0])
        time_text.set_text(f"t = {t[0]:.2f} s")
        return rod_line, bob_patch, cart_patch, time_text

    def update(i):
        cart_patch.set_xy((x_cart[i] - cart_w/2, 0.0))
        rod_line.set_data([x_cart[i], x_bob[i]], [0.0, z_bob[i]])
        bob_patch.center = (x_bob[i], z_bob[i])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return rod_line, bob_patch, cart_patch, time_text

    # Animation timing
    frame_dt = t[1] - t[0] if len(t) > 1 else 0.05
    interval_ms = 1000 * frame_dt / max(speed, 1e-6)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=False,
    )

    # Save GIF / MP4
    if save_path:
        try:
            if save_path.lower().endswith(".mp4"):
                from matplotlib.animation import FFMpegWriter
                anim.save(
                    save_path,
                    writer=FFMpegWriter(fps=int(1.0 / frame_dt * speed)),
                )
            else:
                from matplotlib.animation import PillowWriter
                anim.save(
                    save_path,
                    writer=PillowWriter(fps=int(1.0 / frame_dt * speed)),
                )
            print("Animation saved at:", os.path.abspath(save_path))
        except Exception as e:
            print("Animation save failed:", e)

    if show_window and not save_path:
        plt.show()

    return anim


# ---------------------------------------------------------------------
# Simple performance metrics
# ---------------------------------------------------------------------
def settling_time(t, sig, tol=0.02, t_start=0.0):
    """
    First time >= t_start after which |sig| always stays within tol.
    If it never settles, returns NaN.
    """
    idx0 = np.searchsorted(t, t_start)
    for k in range(idx0, len(t)):
        if np.all(np.abs(sig[k:]) <= tol):
            return t[k]
    return np.nan


def summarize_run(t, X, U, t_final=6.2, name="MPC"):
    """
    Print the key results that I also mention in the presentation:
      • how fast the cart settles at the target
      • how fast the swing dies out
      • how large the peak force was
      • how smooth the force was overall
    """
    y, v, th, q = X

    st_y  = settling_time(t, y,  tol=0.02, t_start=t_final)
    st_th = settling_time(t, th, tol=0.02, t_start=t_final)
    st_q  = settling_time(t, q,  tol=0.02, t_start=t_final)

    peak_u = float(np.max(np.abs(U)))           # max |force|
    tv_u   = float(np.sum(np.abs(np.diff(U))))  # total variation of u (smoothness)


     # print the model matrices used in the controller:
    A, B = crane_pendulum_matrices(m1=10.0, m2=10.0)
    Ad, Bd = c2d_exact(A, B, dt=0.1)

    print("\nContinuous-time model (linearized):")
    print("A =")
    print(A)
    print("B =")
    print(B)

    print("\nDiscrete-time model (used by MPC):")
    print("Ad =")
    print(Ad)
    print("Bd =")
    print(Bd)


    print(f"{name}:")
    print(f"  settle y     ~ {st_y:.2f} s")
    print(f"  settle theta ~ {st_th:.2f} s")
    print(f"  settle q     ~ {st_q:.2f} s")
    print(f"  max |u|      = {peak_u:.2f} N")
    print(f"  total Δu     = {tv_u:.2f} (lower = smoother)")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Run MPC once for m2 = 10 kg, starting at y = -2 m.
    # NOTE:
    #   This is a single open-loop plan for this scenario.
    t, X, U = solve_mpc(
        m1=10.0,
        m2=10.0,
        dt=0.1,
        T=7.0,
        t_final=6.2,
        y0=-2.0,   # [m]
        v0=0.0,    # [m/s]
        th0=0.0,   # [rad]
        q0=0.0,    # [rad/s]
        u_max=5.0, # [N]
    )

    # Make and save animation (shown in presentation)
    animate_motion(
        t,
        X,
        m1=10.0,
        m2=10.0,
        L=0.8,
        save_path="mpc_anim.gif",
        speed=1.0,
        show_window=False,
    )

    # Plot main signals (shown in presentation)
    plot_response(t, X, U, title="MPC response (m2 = 10 kg)")

    # Print summary numbers (what I report on the Results slide)
    summarize_run(t, X, U, t_final=6.2, name="MPC run")



# #--------Case with m2 = 2 kg (for comparison)--------
#     # Run MPC once for m2 = 2 kg, starting at y = -2 m.
#     t2, X2, U2 = solve_mpc(
#         m1=10.0,
#         m2=2.0,
#         dt=0.1,
#         T=7.0,
#         t_final=6.2,
#         y0=-2.0,   # [m]
#         v0=0.0,    # [m/s]
#         th0=0.0,   # [rad]
#         q0=0.0,    # [rad/s]
#         u_max=5.0, # [N]
#     )

#     # Plot main signals for m2 = 2 kg
#     plot_response(t2, X2, U2, title="MPC response (m2 = 2 kg)")

#     # Print summary numbers for m2 = 2 kg
#     summarize_run(t2, X2, U2, t_final=6.2, name="MPC run (m2 = 2 kg)")