"""The demand estimate for the simulator (S3): low / mode / high in the profile's demand unit, every input listed.

Each demand formula (profiles.factors.DEMAND_FORMULAS) is written here as a sum of terms. A term multiplies one
measured data value (or nothing) by profile rates and a constant:

    restaurant_cafe   covers/day        = residents x visits_per_resident_per_month / 30 x capture_share
                                        + daytime workers x visits_per_worker_per_month / 30 x capture_share
    car_wash          cars/day          = AADT x capture_rate
    motel             room-nights/day   = rooms x occupancy
    short_term_rental booked nights/mo  = nights_available_per_month x occupancy
    laundromat        machine turns/day = renter households x share_of_renters_using
                                          x turns_per_user_household_per_week / 7 x capture_share
    retail_storefront transactions/day  = AADT x drive_by_conversion
                                        + residents x visits_per_resident_per_month / 30 x capture_share

Method for the range (stated in every result): low uses every rate at its low end, mode at its mode, high at its
high end. The rates are the profile's ASSUMPTION ranges, so the range is wide on purpose; it is a bound for the
simulator to sample within, not a confidence interval. The measured data value is used as it is (survey margins of
error are not modeled). A term whose data is UNKNOWN is left out and the estimate is marked incomplete; it is never
filled in. Short-term rental comparables are listed as context: listing counts and prices do not give occupancy, so
occupancy stays the profile's ASSUMPTION.
"""

from dataclasses import dataclass

from uwcopilot.studio.profiles.factors import DEMAND_FORMULAS, LOCATION_FACTORS
from uwcopilot.studio.profiles.models import Profile
from uwcopilot.studio.siteiq.factors import RawFactor
from uwcopilot.studio.siteiq.models import COVERED_STATUSES, DemandEstimate, DemandInput

DAYS_PER_MONTH = 30
DAYS_PER_WEEK = 7
RANGE_METHOD = (
    "low = every profile rate at its low end, mode = at its mode, high = at its high end (rates are ASSUMPTION "
    "ranges); measured data values are used as they are"
)


@dataclass(frozen=True)
class Term:
    """One additive term of a demand formula: data factor (or None) x rates x constant."""

    data_factor: str | None
    rates: tuple[str, ...]
    constant: float = 1.0


FORMULA_TERMS: dict[str, tuple[Term, ...]] = {
    "population_visit_rate": (
        Term("resident_population", ("visits_per_resident_per_month", "capture_share"), 1 / DAYS_PER_MONTH),
        Term("daytime_workers", ("visits_per_worker_per_month", "capture_share"), 1 / DAYS_PER_MONTH),
    ),
    "aadt_capture": (Term("traffic_aadt", ("capture_rate",)),),
    "rooms_occupancy": (Term(None, ("rooms", "occupancy")),),
    "nights_comparables": (Term(None, ("nights_available_per_month", "occupancy")),),
    "renter_usage": (
        Term(
            "renter_households",
            ("share_of_renters_using", "turns_per_user_household_per_week", "capture_share"),
            1 / DAYS_PER_WEEK,
        ),
    ),
    "traffic_conversion": (
        Term("traffic_aadt", ("drive_by_conversion",)),
        Term("resident_population", ("visits_per_resident_per_month", "capture_share"), 1 / DAYS_PER_MONTH),
    ),
}
# Data factors a formula lists as context only (shown with the inputs, not multiplied in).
CONTEXT_FACTORS: dict[str, tuple[str, ...]] = {"nights_comparables": ("str_comparables",)}


def term_value(term: Term, data_value: float, rates: dict[str, float]) -> float:
    """data x product of the rates x constant."""
    value = data_value * term.constant
    for name in term.rates:
        value *= rates[name]
    return value


def rate_inputs(profile: Profile) -> list[DemandInput]:
    """The profile's demand rates as inputs (each an ASSUMPTION or SOURCED range with its note)."""
    inputs = []
    for name, quantity in profile.demand.inputs.items():
        low, high = quantity.bounds()
        status = "ASSUMPTION" if quantity.status == "ASSUMPTION" else "DATA"
        inputs.append(
            DemandInput(
                name=name,
                role="rate",
                value=quantity.point(),
                low=low,
                high=high,
                unit=quantity.unit,
                status=status,
                reason=quantity.note or (quantity.source.title if quantity.source else "profile value"),
                profile_field=f"demand.inputs.{name}",
            )
        )
    return inputs


def data_input(raw: RawFactor, role: str = "data") -> DemandInput:
    """A measured factor as a demand input."""
    return DemandInput(
        name=raw.factor,
        role=role,
        value=raw.value,
        unit=LOCATION_FACTORS[raw.factor].unit,
        status=raw.status,
        reason=f"{raw.reason} [{raw.area}]",
        sources=list(raw.sources),
    )


def not_measured(factor: str, area: str) -> RawFactor:
    """An UNKNOWN placeholder for a data factor the caller did not measure."""
    return RawFactor(factor, None, "UNKNOWN", area, "not measured for this estimate")


def capacity_note(profile: Profile, mode: float | None) -> str | None:
    """A note when the mode estimate is above the high end of the profile's capacity range."""
    capacity = profile.capacity.max_demand
    _, high = capacity.bounds()
    if mode is None or mode <= high:
        return None
    return (
        f"The mode estimate ({mode:,.1f}) is above the high end of the capacity range ({high:,.0f} {capacity.unit}); "
        "capacity, not demand, would limit sales"
    )


def estimate_demand(profile: Profile, data: dict[str, RawFactor], area: str) -> DemandEstimate:
    """The demand estimate from the profile's formula and the measured data (see the module docstring)."""
    formula = DEMAND_FORMULAS[profile.demand.formula]
    terms = FORMULA_TERMS[formula.name]
    rates = profile.demand.inputs
    levels = {
        "low": {name: quantity.bounds()[0] for name, quantity in rates.items()},
        "mode": {name: quantity.point() for name, quantity in rates.items()},
        "high": {name: quantity.bounds()[1] for name, quantity in rates.items()},
    }
    totals = {"low": 0.0, "mode": 0.0, "high": 0.0}
    inputs: list[DemandInput] = []
    missing: list[str] = []
    used_terms = 0
    uses_data = False
    for term in terms:
        data_value = 1.0
        if term.data_factor is not None:
            raw = data.get(term.data_factor) or not_measured(term.data_factor, area)
            inputs.append(data_input(raw))
            if raw.value is None or raw.status not in COVERED_STATUSES:
                missing.append(f"{term.data_factor}: {raw.reason}")
                continue
            data_value = raw.value
            uses_data = True
        used_terms += 1
        for level, values in levels.items():
            totals[level] += term_value(term, data_value, values)
    for factor in CONTEXT_FACTORS.get(formula.name, ()):
        if factor in data:
            inputs.append(data_input(data[factor], role="context"))
    inputs.extend(rate_inputs(profile))
    known = used_terms > 0
    basis = "UNKNOWN" if not known else ("data x ASSUMPTION rates" if uses_data else "ASSUMPTION rates only")
    return DemandEstimate(
        formula=formula.name,
        formula_description=formula.description,
        unit=profile.demand_unit,
        low=round(totals["low"], 3) if known else None,
        mode=round(totals["mode"], 3) if known else None,
        high=round(totals["high"], 3) if known else None,
        basis=basis,
        complete=not missing,
        missing=missing,
        method=RANGE_METHOD,
        area=area,
        inputs=inputs,
        capacity_note=capacity_note(profile, round(totals["mode"], 3) if known else None),
    )
