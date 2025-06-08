import typing
from dataclasses import field

import elodin as el
import jax
from jax import numpy as jnp
from jax import random
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
MAX_TICKS = 2400
BALL_RADIUS = 0.25
ELEVATION = 35.0
AZIMUTH = 30.0

# Ref: https://www.thrustcurve.org/motors/AeroTech/M685W/
THRUST_CURVE = {
    "time": (
        0.083,
        0.13,
        0.249,
        0.308,
        0.403,
        0.675,
        1.018,
        1.456,
        1.977,
        2.995,
        3.99,
        4.985,
        5.494,
        5.991,
        7.258,
        7.862,
        8.015,
        8.998,
        9.993,
        10.514,
        11.496,
        11.994,
    ),
    "force": (
        1333.469,
        1368.376,
        1361.395,
        1380.012,
        1359.068,
        1184.53,
        1072.826,
        996.029,
        958.794,
        914.578,
        856.399,
        781.929,
        730.732,
        679.534,
        542.231,
        463.107,
        456.125,
        330.458,
        207.118,
        137.303,
        34.908,
        0.0,
    ),
}


# DATA STRUCTURES

ThrustDirection = typing.Annotated[
    jax.Array,
    el.Component(
        "angles",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "elevation, azimuth"},
    ),
]

Thrust = typing.Annotated[
    jax.Array,
    el.Component(
        "thrust",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x, y, z"},
    ),
]


@el.dataclass
class ThrustData(el.Archetype):
    direction: ThrustDirection = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    thrust: Thrust = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@el.dataclass
class WindData(el.Archetype):
    seed: el.Seed = field(default_factory=lambda: jnp.int64(0))
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


# SAMPLING FUNCTIONS


@el.map
def sample_angles(_t: ThrustDirection) -> ThrustDirection:
    return jnp.array([ELEVATION, AZIMUTH])


@el.map
def sample_wind(s: el.Seed, _w: Wind) -> Wind:
    return random.normal(random.key(s), shape=(3,))


# BASIC PHYSICS


@el.map
def add_ground_plane(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[2], v.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(linear=v.linear() * jnp.array([0.0, 0.0, 0.0])),
        lambda _: v,
        operand=None,
    )


@el.map
def apply_gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


# THRUST


@el.system
def compute_thrust(
    tick: el.Query[el.SimulationTick], dt: el.Query[el.SimulationTimeStep], q: el.Query[ThrustDirection]
) -> el.Query[Thrust]:
    def compute_direction(td: ThrustDirection):
        theta = jnp.radians(td[0])
        phi = jnp.radians(td[1])
        return jnp.array(
            [
                jnp.cos(theta) * jnp.cos(phi),  # x
                jnp.cos(theta) * jnp.sin(phi),  # y
                jnp.sin(theta),  # z
            ]
        )

    def compute_vector(td: ThrustDirection) -> Thrust:
        t = tick[0] * dt[0]
        mag = jnp.interp(t, jnp.array(THRUST_CURVE["time"]), jnp.array(THRUST_CURVE["force"]))
        vec = compute_direction(td)
        return mag * vec

    return q.map(Thrust, compute_vector)


@el.map
def apply_thrust(f: el.Force, thrust: Thrust) -> el.Force:
    return f + el.SpatialForce(linear=thrust)


# WIND DRAG


def calculate_drag(Cd, r, V, A):
    return 0.5 * (Cd * r * V**2 * A)


@el.map
def apply_drag(w: Wind, v: el.WorldVel, f: el.Force) -> el.Force:
    fluid_movement_vector = w
    fluid_movement_vector -= v.linear()

    ball_drag_coefficient = 0.5
    fluid_density = 1.225
    fluid_velocity = la.norm(fluid_movement_vector)
    ball_surface_area = 2 * 3.1415 * BALL_RADIUS**2

    drag_force = calculate_drag(ball_drag_coefficient, fluid_density, fluid_velocity, ball_surface_area)

    fluid_vector_direction = fluid_movement_vector / fluid_velocity
    return el.SpatialForce(linear=f.force() + drag_force * fluid_vector_direction)


# WORLD


def world(seed: int = 0) -> el.World:
    world = el.World()
    geometry = world.insert_asset(el.Mesh.sphere(BALL_RADIUS))
    color = world.insert_asset(el.Material.color(12.7, 9.2, 0.5))
    warhead = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.SpatialInertia(mass=jnp.array([50.0])),
            ),
            el.Shape(geometry, color),
            WindData(seed=jnp.int64(seed)),
            ThrustData(direction=jnp.array([ELEVATION, AZIMUTH])),
        ],
        name="Warhead",
    )
    world.spawn(
        el.Panel.viewport(
            track_rotation=False,
            active=True,
            pos=[-40.0, -40.0, 20.0],
            looking_at=[0.0, 0.0, 5.0],
            show_grid=True,
            hdr=True,
        ),
        name="Viewport",
    )
    world.spawn(el.Line3d(warhead, "world_pos", index=[4, 5, 6], line_width=2.0), name="Warhead Trajectory")
    return world


def system() -> el.System:
    effectors = apply_thrust | apply_gravity | apply_drag
    sys = add_ground_plane | compute_thrust | sample_wind | el.six_dof(sys=effectors)
    return sys


if __name__ == "__main__":
    world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=MAX_TICKS)
