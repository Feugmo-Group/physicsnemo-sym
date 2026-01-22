from registry import *
from sympy import Symbol, Function, Number, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE


class PoissonNernstPlanck(PDE):
    """
    Nondimensional 1D Poisson-Nernst-Planck (PNP) system for two ionic species
    Reference:
    Subramaniam, A., Chen, J., Jang, T., Geise, N. R., Kasse, R. M.,
    Toney, M. F., & Subramanian, V. R. (2019). Analysis and Simulation
    of One-Dimensional Transport Models for Lithium Symmetric Cells.
    Journal of The Electrochemical Society, 166(15), A3806.
    doi:10.1149/2.0261915jes

    Parameters
    ==========
    eps: float
        Dimensionless Poisson parameter. Defined as
        sqrt(R * T * eps_s * eps_0 / (z_p ** 2 * F ** 2 * c_0 * L ** 2))
        Default is 1.
    xi: float
        Dimensionless ratio of diffusion coefficients: D_p / D_n. Default is 1.

    Example
    ========
    >>> pnp = PoissonNernstPlanck(eps=0.1, xi=0.1)
    >>> pnp.pprint()
    poisson: -cn + cp + 0.01*phi__x__x
    continuity_p: -cp*phi__x__x - cp__x*phi__x - cp__x__x + cp__y
    continuity_n: 0.1*cn*phi__x__x + 0.1*cn__x*phi__x - 0.1*cn__x__x + cn__y
    """

    name = "PoissonNernstPlanck"

    def __init__(self, eps=1.0, xi=1.0):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make cp, cn, and phi functions
        cp = Function("cp")(*input_variables)
        cn = Function("cn")(*input_variables)
        phi = Function("phi")(*input_variables)

        # nondimensional constants
        eps = Number(eps)
        xi = Number(xi)

        # set equations
        self.equations = {}
        self.equations["poisson"] = (
            eps ** 2 * phi.diff(x, 2) + (cp - cn)
        )
        self.equations['continuity_p'] = (
            cp.diff(y, 1) - (
                cp.diff(x, 2) + cp * phi.diff(x, 2)
                + cp.diff(x, 1) * phi.diff(x, 1)
            )
        )
        self.equations['continuity_n'] = (
            cn.diff(y, 1) - xi * (
                cn.diff(x, 2) - cn * phi.diff(x, 2)
                - cn.diff(x, 1) *  phi.diff(x, 1)
            )
        )


class BoundaryConditions(PDE):
    """
    Boundary conditions for lithium symmetric cell 1D PNP system
    Reference:
    Subramaniam, A., Chen, J., Jang, T., Geise, N. R., Kasse, R. M.,
    Toney, M. F., & Subramanian, V. R. (2019). Analysis and Simulation
    of One-Dimensional Transport Models for Lithium Symmetric Cells.
    Journal of The Electrochemical Society, 166(15), A3806.
    doi:10.1149/2.0261915jes

    Parameters
    ==========
    delta: float
        Dimensionless cation flux parameter. Defined as
        I_app * L / (z_p * F * c_0 * D_p)
        Default is 1.

    Example
    ========
    >>> bc = BoundaryConditions(delta=0.1)
    >>> bc.pprint()
    neumann_phi_left: phi__x
    flux_cp_left: -cp*phi__x - cp__x - 0.1
    flux_cn_left: cn*phi__x - cn__x
    dirichlet_phi_right: phi
    flux_cp_right: -cp*phi__x - cp__x - 0.1
    flux_cn_right: cn*phi__x - cn__x
    """

    name = "BoundaryConditions"

    def __init__(self, delta=1.0):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make cp, cn, and phi functions
        cp = Function("cp")(*input_variables)
        cn = Function("cn")(*input_variables)
        phi = Function("phi")(*input_variables)

        # nondimensional constants
        delta = Number(delta)

        # set equations
        self.equations = {}

        # left boundary (x=0)
        self.equations["neumann_phi_left"] = (
            phi.diff(x, 1)
        )
        self.equations["flux_cp_left"] = (
            -cp.diff(x, 1) - cp * phi.diff(x, 1) - delta
        )
        self.equations["flux_cn_left"] = (
            -cn.diff(x, 1) + cn * phi.diff(x, 1)
        )

        # right boundary (x=1)
        self.equations["dirichlet_phi_right"] = (
            phi
        )
        self.equations["flux_cp_right"] = (
            -cp.diff(x, 1) - cp * phi.diff(x, 1) - delta
        )
        self.equations["flux_cn_right"] = (
            -cn.diff(x, 1) + cn * phi.diff(x, 1)
        )


@physicsnemo.sym.main(config_path="conf", config_name="config_kan")
def run(cfg: PhysicsNeMoConfig) -> None:
    # instantiate simulation parameters
    p = Parameters()

    # make a list of nodes for the graph to unroll on
    pnp = PoissonNernstPlanck(eps=p.eps, xi=p.xi)
    bc = BoundaryConditions(delta=p.delta)
    net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        cfg=cfg.arch[next(iter(cfg.arch))],
    )
    nodes = pnp.make_nodes() + bc.make_nodes() + [net.make_node(name='net')]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    y_f = p.t_f / p.t_c  # final dimensionless time

    if cfg.custom.grid_sampling:
        rec = GridRectangle((0.0, 0.0), (1.0, y_f))
    else:
        rec = Rectangle((0.0, 0.0), (1.0, y_f))

    # make pnp domain
    pnp_domain = Domain()

    # initial condition
    initial = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"cp": 1.0, "cn": 1.0, "phi": 0.0},
        batch_size=cfg.batch_size.Initial,
        criteria=Eq(y, 0.0),
        quasirandom=True,
    )
    pnp_domain.add_constraint(initial, "initial")

    # left boundary
    left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"flux_cn_left": 0.0, "flux_cp_left": 0.0, "neumann_phi_left": 0.0},
        batch_size=cfg.batch_size.Left,
        criteria=Eq(x, 0.0),
        quasirandom=True,
    )
    pnp_domain.add_constraint(left, "left")

    # right boundary
    right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"flux_cn_right": 0.0, "flux_cp_right": 0.0, "dirichlet_phi_right": 0.0},
        batch_size=cfg.batch_size.Right,
        criteria=Eq(x, 1.0),
        quasirandom=True,
    )
    pnp_domain.add_constraint(right, "right")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"poisson": 0.0, "continuity_p": 0.0, "continuity_n": 0.0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "poisson": Symbol("sdf"),
            "continuity_p": Symbol("sdf"),
            "continuity_n": Symbol("sdf"),
        },
        quasirandom=True,
    )
    pnp_domain.add_constraint(interior, "interior")

    # make solver
    slv = Solver(cfg, pnp_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    register_custom_arch_configs()
    register_custom_loss_configs()

    run()