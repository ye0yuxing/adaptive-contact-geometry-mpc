# Model and numerical conventions

The state `q` contains the peg position (3), scalar-first quaternion (4), and
three fingertip positions (9). Generalized velocity has 15 coordinates: world
translation, body angular velocity, and fingertip translation. `B(q)` maps
velocity to configuration derivative. The action `u` is a nine-dimensional
desired fingertip displacement, with each component bounded by ±5 mm.

## Geometry

For a box, form its six signed support-plane residuals `s_i(p)`. The field is
`log(sum(exp(rho*s_i)))/rho`. A negative log-sum-exp combines the two peg parts
and separately aggregates 26 surface samples per part against each environment
body. The table aggregation includes the fixed `log(9)/rho` correction for
the nine coplanar lower-face samples. This is a local calibration, not an exact
distance guarantee for arbitrary orientation.

There are 11 potential pairs and four friction-pyramid rows per pair. Let
`phi(q)` be the repeated **full gap**, `Jn = d(phi)/dq @ B(q)`, and `Jt` the
four signed tangential rows. Define `J = Jn + mu*Jt`, with `mu = 0.6`.
Environment tangents use fixed world coordinate axes. Fingertip tangents use
the field normal and a world-z reference; near alignment with that reference,
the tangent construction can degenerate. The task is predominantly planar.

Geometry is computed inside the symbolic graph; gradients include its state
dependence in the adaptive arm. Both arms refresh their measured-state
geometry on every real simulation step. “Frozen” only refers to the prediction
horizon within one MPC solve. With horizon 1, the arms must coincide.

## Explicit CF step

The response follows Jin's dual-cone approximation. In the displacement
notation used here:

```text
Q = diag(50,50,50, 0.06,0.06,0.06, 100,...,100)
b = [0,0,-0.098,0,0,0, 100*u]
d_free = inverse(Q) * b
lambda = SoftPlus_100(-phi - (1 + 0.1/h_model) * J*d_free)
d = d_free + inverse(Q) * transpose(J)*lambda
q_next = q + B(q)*d
```

`Q` is a quasi-dynamic control metric, not the physical MuJoCo mass matrix.
`K=I`, `h_model=0.1`, and damping `D=0.1I` is **inside** SoftPlus. Stable
log-sum-exp and SoftPlus expressions avoid exponential overflow.

The response is an approximation, not an exact multi-contact solve. In
particular, adapting geometry does not make the approximate contact response
an exact physical model. The quaternion update is first-order and does not
renormalize predicted states. MuJoCo maintains its own physical quaternion.
The polynomial rotation map is evaluated at non-unit predicted quaternions too.

## Optimization and execution

The objective is a running grasp/proximity term plus `50*||u||^2` and a terminal
`6000*||p-p_target||^2 + 350*(1-qw^2)` term. The complete running term appears
in `task.py`; target orientation is the identity quaternion. Fingertip bounds
are x ∈ [−0.18, 0.19], y ∈ [−0.10, 0.10], z ∈ [0.006, 0.035] metres.

Direct multiple shooting uses one action and next-state block per stage.
CasADi MX expressions are expanded; IPOPT uses limited-memory Hessians and a
300-iteration cap. The first guess uses the measured state at every stage.
Subsequent solves reuse the previous primal/dual solution without shifting.
Finite failed iterates are executed and identified in the logs.

MuJoCo executes a fingertip PD controller (gains 100 and 2) with gravity
compensation for 10 × 2 ms. Motors are not force-clamped. The geometry functions
used by MPC do not replace MuJoCo collision geometry. The latter determines
execution contacts and the final success hold. Planning time is measured from
`action()` entry through the solver return, excluding post-solve diagnostic
geometry queries, simulation, initial graph construction, and rendering.

For mathematical sources and attribution, see [provenance](provenance.md).
