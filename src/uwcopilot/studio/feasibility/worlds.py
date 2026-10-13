"""SYNTHETIC validation worlds: a plan, a site demand range and a loan per world (Studio S3 validation).

A world is what a founder would bring to the simulator. validation.py then draws a hidden "true" future for the
world and checks the simulator's bands and probabilities against it. Nothing here describes a real business: every
constant below is a generator parameter chosen by the project author. They are not benchmarks.

Why the site demand is "anchored": with the profiles' ASSUMPTION ranges, a demand drawn at random would make most
worlds hopeless (or trivially safe), and a probability that is always about 0 is easy to score well on. So each
world first draws its plan and loan and a target DSCR for its one-forecast case (TARGET_SINGLE_POINT_DSCR), then
finds by bisection the most likely site demand that gives that DSCR when every input is at its mode. When even the
capacity mode cannot reach the target (a plan that loses money at full capacity under the profile's modes), the
demand is set to the capacity mode and the world is marked `anchored=False`; such worlds are kept and counted.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from uwcopilot.studio.data.fabric_base import FactorValue
from uwcopilot.studio.feasibility.assumptions import FeasibilityAssumptions
from uwcopilot.studio.feasibility.inputs import Loan, Plan, SiteDemand
from uwcopilot.studio.feasibility.metrics import annual_total
from uwcopilot.studio.feasibility.model import single_point_run, single_point_values
from uwcopilot.studio.feasibility.setup import SimulationSetup, build_setup
from uwcopilot.studio.profiles.models import Profile

# Generator parameters (SYNTHETIC; design choices, not data).
TARGET_SINGLE_POINT_DSCR = (0.6, 2.4)  # the one-forecast DSCR each world is anchored to
DEMAND_LOW_FACTOR = (0.5, 0.85)  # low end of the site's range, as a share of its mode
DEMAND_HIGH_FACTOR = (1.15, 1.6)  # high end of the site's range, as a multiple of its mode
CAPEX_FACTOR = (0.6, 1.3)  # planned capex, as a multiple of the profile's capex modes
LOAN_SHARE_OF_CAPEX = (0.5, 0.85)
WORKING_CAPITAL_SHARE_OF_CAPEX = (0.05, 0.3)
LOAN_TERMS_MONTHS: tuple[int, ...] = (60, 84, 120)
VARIABLE_RATE_SHARE = 0.5
PRICE_STATED_SHARE = 0.5  # share of worlds where the founder types a price
PRICE_FACTOR = (0.85, 1.2)  # stated price, as a multiple of the profile's price mode
RENT_QUOTE_SHARE = 0.5  # share of worlds with a rent quote
RENT_FACTOR = (0.6, 1.3)  # quote, as a multiple of the metro rent value
OPENING_YEAR = 2027
BISECTION_STEPS = 40
ANCHOR_YEAR = 2
WORLD_SOURCE = "SYNTHETIC validation world (src/uwcopilot/studio/feasibility/worlds.py)"
METRO_ID = "synthetic_metro"


@dataclass(frozen=True)
class World:
    """One SYNTHETIC founder scenario."""

    world_id: int
    profile_id: str
    site: SiteDemand
    plan: Plan
    loan: Loan
    target_single_point_dscr: float
    anchored: bool


def uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    """One uniform draw between the bounds."""
    return float(rng.uniform(bounds[0], bounds[1]))


def single_point_dscr(setup: SimulationSetup, demand: float) -> float:
    """Year-2 DSCR (unrounded) of the one-forecast case with the site's most likely demand set to `demand`."""
    values = single_point_values(setup)
    values["mature_demand"] = np.array([demand])
    paths = single_point_run(setup, values)
    return float(annual_total(paths.noi, ANCHOR_YEAR)[0] / annual_total(paths.debt_service, ANCHOR_YEAR)[0])


def anchor_demand(setup: SimulationSetup, target: float, ceiling: float) -> tuple[float, bool]:
    """(demand, reached): the demand in (0, ceiling] whose one-forecast DSCR equals target, by bisection.

    The one-forecast DSCR rises with demand (each unit adds price x (1 - cost ratios) > 0), so bisection works.
    When the ceiling cannot reach the target, return (ceiling, False).
    """
    if single_point_dscr(setup, ceiling) < target:
        return ceiling, False
    low, high = 0.0, ceiling
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0
        if single_point_dscr(setup, middle) < target:
            low = middle
        else:
            high = middle
    return high, True


def draw_plan(profile: Profile, metro_inputs: dict[str, FactorValue], rng: np.random.Generator) -> tuple[Plan, Loan]:
    """The world's plan and loan (draws always in the same order, so a seed reproduces them)."""
    capex = sum(item.cost.point() for item in profile.capex) * uniform(rng, CAPEX_FACTOR)
    loan_amount = capex * uniform(rng, LOAN_SHARE_OF_CAPEX)
    working_capital = capex * uniform(rng, WORKING_CAPITAL_SHARE_OF_CAPEX)
    term = int(rng.choice(LOAN_TERMS_MONTHS))
    variable = bool(rng.uniform() < VARIABLE_RATE_SHARE)
    price_draw, price_stated = uniform(rng, PRICE_FACTOR), bool(rng.uniform() < PRICE_STATED_SHARE)
    rent_draw, rent_quoted = uniform(rng, RENT_FACTOR), bool(rng.uniform() < RENT_QUOTE_SHARE)
    month = int(rng.integers(1, 13))
    metro_rent = metro_inputs["commercial_rent"].value or profile.rent.monthly_rent.point()
    plan = Plan(
        opening_date=date(OPENING_YEAR, month, 1),
        owner_equity=round(capex + working_capital - loan_amount, 2),
        capex=round(capex, 2),
        price=round(profile.price.point() * price_draw, 2) if price_stated else None,
        monthly_rent=round(metro_rent * rent_draw, 2) if rent_quoted else None,
    )
    loan = Loan(amount=round(loan_amount, 2), term_months=term, rate_type="variable" if variable else "fixed")
    return plan, loan


def generate_world(
    profile: Profile,
    metro_inputs: dict[str, FactorValue],
    assumptions: FeasibilityAssumptions,
    world_id: int,
    rng: np.random.Generator,
    months: int = 36,
) -> World:
    """Draw one world and anchor its site demand (see the module docstring)."""
    plan, loan = draw_plan(profile, metro_inputs, rng)
    target = uniform(rng, TARGET_SINGLE_POINT_DSCR)
    low_factor, high_factor = uniform(rng, DEMAND_LOW_FACTOR), uniform(rng, DEMAND_HIGH_FACTOR)
    competition = float(rng.uniform())
    ceiling = profile.capacity.max_demand.point()
    placeholder = SiteDemand(low=ceiling, mode=ceiling, high=ceiling, unit=profile.demand_unit, status="SYNTHETIC")
    setup = build_setup(profile, METRO_ID, "SYNTHETIC", metro_inputs, placeholder, plan, loan, assumptions, months)
    demand, anchored = anchor_demand(setup, target, ceiling)
    site = SiteDemand(
        low=demand * low_factor,
        mode=demand,
        high=demand * high_factor,
        unit=profile.demand_unit,
        status="SYNTHETIC",
        source=f"{WORLD_SOURCE}, world {world_id}",
        competition_percentile=competition,
    )
    return World(world_id, profile.id, site, plan, loan, target, anchored)
