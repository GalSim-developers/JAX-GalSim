import warnings

import jax
import jax.scipy.optimize as jspop
import numpy as np
import optax
import scipy.optimize as spop
import tqdm


def min_optax(
    fun,
    x0,
    args=None,
    maxiter=100_000,
    learning_rate=1e-1,
    method="adan",
    optimizer=None,
    opt_state=None,
    update_prog_iter=100,
):
    args = args or tuple()
    _vag_fun = jax.jit(jax.value_and_grad(fun))

    if optimizer is None:
        optimizer = getattr(optax, method)(learning_rate)
        opt_state = optimizer.init(x0)

        @jax.jit
        def _update_func(coeffs, opt_state):
            loss, grads = _vag_fun(coeffs, *args)
            updates, opt_state = optimizer.update(grads, opt_state, params=coeffs)
            coeffs = optax.apply_updates(coeffs, updates)
            return coeffs, opt_state, loss

    loss, _ = _vag_fun(x0, *args)

    prev_loss = loss
    coeffs = x0

    loss = fun(coeffs, *args)
    initial_desc = f"{method}: {loss:12.8e} ({np.nan:+9.2e} delta)"

    with tqdm.trange(maxiter, desc=initial_desc) as pbar:
        for i in pbar:
            coeffs, opt_state, loss = _update_func(coeffs, opt_state)

            if i % update_prog_iter == 0 or i == 0:
                if prev_loss is not None:
                    dloss = loss - prev_loss
                else:
                    dloss = np.nan

                pbar.set_description(f"{method}: {loss:12.8e} ({dloss:+9.2e} delta)")

                prev_loss = loss

    return coeffs, (optimizer, opt_state)


def min_bfgs(
    fun,
    x0,
    args=None,
    maxiter=100,
    use_scipy=False,
    maxiter_per_bfgs_call=None,
    tol=None,
    method="BFGS",
):
    args = args or tuple()
    jac = jax.jit(jax.grad(fun))
    hess = jax.jit(jax.hessian(fun))

    coeffs = x0
    prev_loss = None
    if tol is None:
        tol = 1e-32 if use_scipy else 1e-16
    if maxiter_per_bfgs_call is None:
        maxiter_per_bfgs_call = 100_000 if use_scipy else 1000

    loss = fun(coeffs, *args)
    initial_desc = f"{method}: {loss:12.8e} ({np.nan:+9.2e} delta, status {-1}, nit {0:6d})"

    with tqdm.trange(maxiter, desc=initial_desc) as pbar:
        for _ in pbar:
            if use_scipy:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
                    warnings.filterwarnings("ignore", message=".*does not use Hessian information.*")
                    warnings.filterwarnings("ignore", message=".*does not use gradient information.*")
                    res = spop.minimize(
                        fun,
                        coeffs,
                        method=method,
                        args=args,
                        jac=jac,
                        hess=hess,
                        tol=tol,
                        options={
                            "maxiter": maxiter_per_bfgs_call,
                            "gtol": tol,
                            "xatol": tol,
                            "fatol": tol,
                            "adaptive": True,
                        },
                    )
            else:
                res = jspop.minimize(
                    fun,
                    coeffs,
                    method="BFGS",
                    args=args,
                    tol=tol,
                    options={"maxiter": maxiter_per_bfgs_call, "gtol": tol, "line_search_maxiter": 40},
                )

            if np.all(coeffs == res.x):
                coeffs = coeffs * (1.0 + (np.random.uniform(size=coeffs.shape[0]) - 0.5) * 1e-10)
            else:
                coeffs = res.x

            if prev_loss is not None:
                dloss = res.fun - prev_loss
            else:
                dloss = np.nan

            prev_loss = res.fun

            pbar.set_description(
                f"{method}: {res.fun:12.8e} ({dloss:+9.2e} delta, status {res.status}, nit {res.nit:6d})"
            )

    return res.x
