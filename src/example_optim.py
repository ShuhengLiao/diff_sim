import jax
import jax.numpy as np
import time
import scipy.optimize as opt
from src.utils import read_path, obj_to_vtu
from src.arguments import args
from src.allen_cahn import polycrystal_fd, phase_field, odeint, rk4, explicit_euler, write_sols
from src.example import initialization
from src.checkpoint import chunk
jax.config.update("jax_enable_x64", True)


def set_params():
    '''
    If a certain parameter is not set, a default value will be used (see src/arguments.py for details).
    '''
    args.case = 'optimize'
    args.num_grains = 10000
    args.domain_length = 0.5
    args.domain_width = 0.2
    args.domain_height = 0.1
    args.r_beam = 0.03
    args.power = 100
    args.write_sol_interval = 500


def run():
    set_params()
    # TODO: bad symbol ys
    ts, xs, ys, ps = read_path(f'data/txt/fd_example_1.txt')

    dt = ts[1] - ts[0]
    polycrystal, mesh = polycrystal_fd(args.case)
    y0 = initialization(polycrystal)
   
    state_rhs, get_T = phase_field(polycrystal)

    # Remark: JAX is type sensitive, if you specify [20., 4], it fails.
    ode_params_0 = [22.9, 4.1]
    # ode_params_0 = [24.9, 5.1]
 
    ode_params_gt = [25., 5.2]
    target_yf, _ = odeint(polycrystal, mesh, get_T, explicit_euler, state_rhs, y0, ts, ode_params_gt)

    def obj_func(yf, target_yf):
        # Some arbitrary objective function
        return np.sum((yf - target_yf)**2)
 
    obj_func_partial = lambda yf: obj_func(yf, target_yf)
 
    # Early discretization seems to be the best option
    # If we further use checkpoint method, we can compute derivative for a long time chain
    def get_ode_fn():
        @jax.jit
        def ode_fn(y_prev, params_prev):
            ode_params, dt, t_prev = params_prev
            y_crt = y_prev + dt * state_rhs(y_prev, t_prev, ode_params)
            t_crt = t_prev + dt
            params_crt = (ode_params, dt, t_crt)
            return (y_crt, params_crt)
        return ode_fn

    ode_fn = get_ode_fn()
    y_combo_ini = (y0, (ode_params_0, dt, 0.))
    # print(f"start of checkpoint")
    chunksize = len(ts[1:])
    num_total_steps = len(ts[1:])
    # chunk(get_ode_fn, obj_func_partial, y_combo_ini, chunksize)


    # parameter calibration
    obj = lambda ode_params: obj_func_partial(odeint(polycrystal, mesh, get_T, explicit_euler, state_rhs, y0, ts, ode_params)[0])
    jac = lambda ode_params: chunk(ode_fn, obj_func_partial, (y0, (ode_params, dt, 0.)), chunksize, num_total_steps)[1][0]
    
    start_time = time.time()
    results = opt.minimize(obj, ode_params_0, method='CG', jac=jac)
    time_elapsed = time.time()-start_time
    print(f"Time elapsed {time_elapsed} for optimization") 
    print(results)



if __name__ == "__main__":
    # neper_domain()
    # write_vtu_files()
    run()