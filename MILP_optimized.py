import argparse
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pulp


# -----------------------------------------------------------------------------
# Input handling and validation
# -----------------------------------------------------------------------------


def csv_path(filename: Path) -> Path:
    """Find a CSV in the current folder, script folder, or system temp folder."""
    filename = Path(filename)

    if filename.is_absolute():
        candidates = [filename]
    else:
        candidates = [
            Path.cwd() / filename,
            Path(__file__).resolve().parent / filename,
            Path(tempfile.gettempdir()) / filename,
        ]

    for candidate in candidates:
        if candidate.is_file():
            print(f"Found CSV: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def require_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def validate_and_prepare_data(demand, sites, distances):
    demand_required = {"demand_id", "ev_sessions_per_day"}
    site_required = {
        "candidate_id",
        "site_name",
        "fixed_build_cost_lakh",
        "cost_per_charger_lakh",
        "annual_fixed_operating_cost_lakh",
        "annual_operating_cost_per_charger_lakh",
        "capacity_per_charger_sessions_per_day",
        "max_chargers",
    }
    distance_required = {"demand_id", "candidate_id", "distance_km"}

    require_columns(demand, demand_required, "demand CSV")
    require_columns(sites, site_required, "sites CSV")
    require_columns(distances, distance_required, "distances CSV")

    # IDs are read as strings so values such as D001 or site 001 remain intact.
    for df, column, label in (
        (demand, "demand_id", "demand_id"),
        (sites, "candidate_id", "candidate_id"),
        (distances, "demand_id", "distance demand_id"),
        (distances, "candidate_id", "distance candidate_id"),
    ):
        if df[column].isna().any():
            raise ValueError(f"Column {label} contains an empty ID.")
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"Column {label} contains a blank ID.")

    if demand["demand_id"].duplicated().any():
        duplicates = demand.loc[demand["demand_id"].duplicated(), "demand_id"].tolist()
        raise ValueError(f"Duplicate demand_id values found: {duplicates[:10]}")
    if sites["candidate_id"].duplicated().any():
        duplicates = sites.loc[sites["candidate_id"].duplicated(), "candidate_id"].tolist()
        raise ValueError(f"Duplicate candidate_id values found: {duplicates[:10]}")
    if distances.duplicated(["demand_id", "candidate_id"]).any():
        duplicates = distances.loc[
            distances.duplicated(["demand_id", "candidate_id"], keep=False),
            ["demand_id", "candidate_id"],
        ]
        raise ValueError(
            "The distance CSV has duplicate demand-zone/site pairs. "
            f"Examples: {duplicates.head(10).to_dict('records')}"
        )

    numeric_columns = {
        "demand": ["ev_sessions_per_day"],
        "sites": [
            "fixed_build_cost_lakh",
            "cost_per_charger_lakh",
            "annual_fixed_operating_cost_lakh",
            "annual_operating_cost_per_charger_lakh",
            "capacity_per_charger_sessions_per_day",
            "max_chargers",
        ],
        "distances": ["distance_km"],
    }
    for df, columns, label in (
        (demand, numeric_columns["demand"], "demand CSV"),
        (sites, numeric_columns["sites"], "sites CSV"),
        (distances, numeric_columns["distances"], "distances CSV"),
    ):
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="raise")
            if df[column].isna().any() or not df[column].map(math.isfinite).all():
                raise ValueError(f"{label}: column {column} contains missing/non-finite values.")

    if (demand["ev_sessions_per_day"] < 0).any():
        raise ValueError("Demand values cannot be negative.")

    cost_columns = [
        "fixed_build_cost_lakh",
        "cost_per_charger_lakh",
        "annual_fixed_operating_cost_lakh",
        "annual_operating_cost_per_charger_lakh",
    ]
    for column in cost_columns:
        if (sites[column] < 0).any():
            raise ValueError(f"Site cost column {column} cannot contain negative values.")

    if (sites["capacity_per_charger_sessions_per_day"] <= 0).any():
        raise ValueError("capacity_per_charger_sessions_per_day must be positive.")
    if (sites["max_chargers"] < 1).any():
        raise ValueError("Every candidate site must allow at least one charger.")
    if not (sites["max_chargers"] % 1 == 0).all():
        raise ValueError("max_chargers must contain whole numbers.")
    sites["max_chargers"] = sites["max_chargers"].astype(int)

    if (distances["distance_km"] < 0).any():
        raise ValueError("Distances cannot be negative.")

    # Drop distance records that reference IDs absent from the corresponding input.
    valid_demand = set(demand["demand_id"])
    valid_sites = set(sites["candidate_id"])
    distances = distances[
        distances["demand_id"].isin(valid_demand)
        & distances["candidate_id"].isin(valid_sites)
    ].copy()

    return demand, sites, distances


# -----------------------------------------------------------------------------
# MILP model
# -----------------------------------------------------------------------------


def build_model(
    demand,
    sites,
    distances,
    capex_budget_lakh,
    radius_km,
    min_stations,
    max_stations,
    integer_allocation=True,
):
    """Build the EV charging allocation model.

    CAPEX is constrained separately. Annual OPEX and travel are returned as
    separate expressions and are never added together in one weighted objective.
    """
    demand_ids = demand["demand_id"].tolist()
    site_ids = sites["candidate_id"].tolist()

    if min_stations < 0 or max_stations < 0 or min_stations > max_stations:
        raise ValueError("Station limits must satisfy 0 <= min-stations <= max-stations.")
    model = pulp.LpProblem("EV_station_allocation", pulp.LpMinimize)

    opened = {
        j: pulp.LpVariable(f"open_{j}", cat=pulp.LpBinary)
        for j in site_ids
    }
    chargers = {
        j: pulp.LpVariable(f"chargers_{j}", lowBound=0, cat=pulp.LpInteger)
        for j in site_ids
    }

    eligible = {
        (str(row.demand_id), str(row.candidate_id)): float(row.distance_km)
        for row in distances.itertuples(index=False)
        if float(row.distance_km) <= radius_km
    }

    allocation_category = pulp.LpInteger if integer_allocation else pulp.LpContinuous
    x = {}
    for i in demand_ids:
        for j in site_ids:
            if (i, j) in eligible:
                x[i, j] = pulp.LpVariable(
                    f"x_{i}_{j}", lowBound=0, cat=allocation_category
                )

    demand_lookup = demand.set_index("demand_id")
    site_lookup = sites.set_index("candidate_id")

    # Separate cost coefficients. All cost values are in lakh rupees.
    capex_fixed = {
        j: float(site_lookup.loc[j, "fixed_build_cost_lakh"])
        for j in site_ids
    }
    capex_per_charger = {
        j: float(site_lookup.loc[j, "cost_per_charger_lakh"])
        for j in site_ids
    }
    opex_fixed_annual = {
        j: float(site_lookup.loc[j, "annual_fixed_operating_cost_lakh"])
        for j in site_ids
    }
    opex_per_charger_annual = {
        j: float(site_lookup.loc[j, "annual_operating_cost_per_charger_lakh"])
        for j in site_ids
    }

    capex_expr = pulp.lpSum(
        capex_fixed[j] * opened[j] + capex_per_charger[j] * chargers[j]
        for j in site_ids
    )
    annual_opex_expr = pulp.lpSum(
        opex_fixed_annual[j] * opened[j]
        + opex_per_charger_annual[j] * chargers[j]
        for j in site_ids
    )
    weighted_distance_expr = pulp.lpSum(
        eligible[i, j] * x[i, j] for i, j in x
    )

    # Every demand zone must have at least one candidate station within radius.
    for i in demand_ids:
        eligible_at_zone = [x[i, j] for j in site_ids if (i, j) in x]
        if not eligible_at_zone:
            return (
                None, None, None, None, None, None, None,
                f"No candidate site is within {radius_km:g} km of demand zone {i}.",
            )
        model += (
            pulp.lpSum(eligible_at_zone)
            == float(demand_lookup.loc[i, "ev_sessions_per_day"]),
            f"demand_coverage_{i}",
        )

    # Charger capacity and station-opening logic.
    for j in site_ids:
        capacity_per_charger = float(
            site_lookup.loc[j, "capacity_per_charger_sessions_per_day"]
        )
        max_chargers_here = int(site_lookup.loc[j, "max_chargers"])

        model += (
            pulp.lpSum(x[i, j] for i in demand_ids if (i, j) in x)
            <= capacity_per_charger * chargers[j],
            f"capacity_{j}",
        )
        model += chargers[j] <= max_chargers_here * opened[j], f"linking_{j}"
        model += chargers[j] >= opened[j], f"minimum_one_charger_{j}"

    model += pulp.lpSum(opened[j] for j in site_ids) >= min_stations, "minimum_stations"
    model += pulp.lpSum(opened[j] for j in site_ids) <= max_stations, "maximum_stations"

    # IMPORTANT: the budget constrains CAPEX only, not annual OPEX.
    if capex_budget_lakh is not None:
        model += capex_expr <= capex_budget_lakh, "capital_expenditure_budget"

    # Initial objective is a feasibility placeholder. The caller sets the real objective.
    model += 0

    expressions = {
        "capex": capex_expr,
        "annual_opex": annual_opex_expr,
        "weighted_distance": weighted_distance_expr,
    }
    coefficients = {
        "capex_fixed": capex_fixed,
        "capex_per_charger": capex_per_charger,
        "opex_fixed_annual": opex_fixed_annual,
        "opex_per_charger_annual": opex_per_charger_annual,
    }
    return model, opened, chargers, x, eligible, expressions, coefficients, None


# -----------------------------------------------------------------------------
# Solve modes: lexicographic objectives (no unit-mixing weights)
# -----------------------------------------------------------------------------


def solve_lexicographic(
    demand,
    sites,
    distances,
    args,
    primary_objective,
    capex_budget_lakh,
):
    """Optimize one metric first, then optimize the other without worsening it.

    Stage 2 is built as a fresh PuLP model instead of re-solving the stage-1
    model. This avoids carrying solver state between CBC runs on Windows.
    """
    if primary_objective == "opex":
        primary_key, secondary_key = "annual_opex", "weighted_distance"
        primary_label = "Annual OPEX"
        secondary_label = "Demand-weighted travel distance"
        tag = "opex_first"
    elif primary_objective == "distance":
        primary_key, secondary_key = "weighted_distance", "annual_opex"
        primary_label = "Demand-weighted travel distance"
        secondary_label = "Annual OPEX"
        tag = "distance_first"
    else:
        raise ValueError(f"Unsupported objective: {primary_objective}")

    def build_for_stage():
        built_model = build_model(
            demand=demand,
            sites=sites,
            distances=distances,
            capex_budget_lakh=capex_budget_lakh,
            radius_km=args.radius,
            min_stations=args.min_stations,
            max_stations=args.max_stations,
            integer_allocation=not args.continuous_allocation,
        )
        if built_model[-1] is not None:
            return None, built_model[-1]
        return built_model, None

    # Stage 1: minimize the priority metric.
    stage1, error = build_for_stage()
    if error:
        return {"status": "Input infeasible", "message": error}

    model1, _, _, _, _, expressions1, _, _ = stage1
    model1.setObjective(expressions1[primary_key])
    status = model1.solve(pulp.PULP_CBC_CMD(msg=args.verbose))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        return {
            "status": status_name,
            "message": (
                f"No optimal {tag} solution under the CAPEX budget. "
                "The model may be infeasible or the solver may have stopped early."
            ),
        }

    primary_optimum = float(pulp.value(expressions1[primary_key]))
    tolerance = max(1e-7, abs(primary_optimum) * 1e-9)

    # Stage 2: rebuild the model from scratch, preserve the primary optimum,
    # then minimize the secondary metric. This is lexicographic optimization.
    stage2, error = build_for_stage()
    if error:
        return {"status": "Input infeasible", "message": error}

    model, opened, chargers, x, eligible, expressions, coefficients, _ = stage2
    model += (
        expressions[primary_key] <= primary_optimum + tolerance,
        f"preserve_primary_{primary_key}",
    )
    model.setObjective(expressions[secondary_key])
    status = model.solve(pulp.PULP_CBC_CMD(msg=args.verbose))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        return {
            "status": status_name,
            "message": (
                f"Stage 1 found the optimum for {primary_label}, but the secondary "
                f"{secondary_label} stage returned {status_name}."
            ),
        }

    result = {
        "status": status_name,
        "tag": tag,
        "primary_label": primary_label,
        "primary_value": primary_optimum,
        "secondary_label": secondary_label,
        "secondary_value": float(pulp.value(expressions[secondary_key])),
        "model": model,
        "opened": opened,
        "chargers": chargers,
        "x": x,
        "eligible": eligible,
        "expressions": expressions,
        "coefficients": coefficients,
    }
    return result


def check_operational_feasibility(demand, sites, distances, args):
    """Find minimum CAPEX; an optimal result proves operational feasibility.

    A separate zero-objective feasibility solve is unnecessary. Minimizing
    CAPEX in one solve both proves feasibility and calculates the minimum
    capital required, while avoiding a second CBC invocation on the same model.
    """
    built = build_model(
        demand=demand,
        sites=sites,
        distances=distances,
        capex_budget_lakh=None,
        radius_km=args.radius,
        min_stations=args.min_stations,
        max_stations=args.max_stations,
        integer_allocation=not args.continuous_allocation,
    )
    if built[-1] is not None:
        return False, None, built[-1]

    model, _, _, _, _, expressions, _, _ = built
    model.setObjective(expressions["capex"])
    status = model.solve(pulp.PULP_CBC_CMD(msg=args.verbose))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        return (
            False,
            None,
            f"Operational constraints are {status_name} even without a CAPEX budget.",
        )

    min_capex = float(pulp.value(expressions["capex"]))
    return True, min_capex, None


# -----------------------------------------------------------------------------
# Result formatting and exports
# -----------------------------------------------------------------------------


LOCATION_HINTS = (
    "name", "latitude", "longitude", "lat", "lon", "address", "location",
    "zone", "ward", "area", "locality", "place", "region", "city", "pincode",
)


def location_metadata(row, prefix=""):
    """Keep useful location/name fields from an input row for map matching."""
    result = {}
    for column, value in row.items():
        lower = str(column).lower()
        if any(hint in lower for hint in LOCATION_HINTS):
            already_prefixed = prefix and str(column).lower().startswith(prefix.lower())
            key = str(column) if already_prefixed or not prefix else f"{prefix}{column}"
            result[key] = value
    return result


def output_path_with_tag(path_value, tag):
    path = Path(path_value)
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}_{tag}{suffix}")


def make_solution_tables(result, demand, sites, total_demand):
    opened = result["opened"]
    chargers = result["chargers"]
    x = result["x"]
    eligible = result["eligible"]
    coefficients = result["coefficients"]

    demand_lookup = demand.set_index("demand_id")
    site_lookup = sites.set_index("candidate_id")

    assignments_by_site = {j: 0.0 for j in sites["candidate_id"]}
    allocation_rows = []
    for (i, j), variable in x.items():
        value = variable.value()
        if value is None or value <= 1e-7:
            continue

        sessions = float(value)
        assignments_by_site[j] += sessions
        demand_row = demand_lookup.loc[i]
        site_row = site_lookup.loc[j]

        row = {
            "demand_id": i,
            "candidate_id": j,
            "site_name": site_row["site_name"],
            "demand_sessions_per_day": float(demand_row["ev_sessions_per_day"]),
            "allocated_sessions_per_day": sessions,
            "distance_km": eligible[i, j],
            "weighted_distance_km_sessions_per_day": eligible[i, j] * sessions,
        }
        row.update(location_metadata(demand_row, prefix="demand_"))
        row.update(location_metadata(site_row, prefix="site_"))
        allocation_rows.append(row)

    station_rows = []
    for j in sites["candidate_id"]:
        is_open = opened[j].value() is not None and opened[j].value() > 0.5
        if not is_open:
            continue

        charger_count = int(round(chargers[j].value() or 0))
        site_row = site_lookup.loc[j]
        assigned = assignments_by_site[j]
        capacity = float(site_row["capacity_per_charger_sessions_per_day"]) * charger_count
        fixed_capex = coefficients["capex_fixed"][j]
        variable_capex = coefficients["capex_per_charger"][j] * charger_count
        fixed_opex = coefficients["opex_fixed_annual"][j]
        variable_opex = coefficients["opex_per_charger_annual"][j] * charger_count

        row = {
            "candidate_id": j,
            "site_name": site_row["site_name"],
            "open": 1,
            "chargers": charger_count,
            "assigned_sessions_per_day": assigned,
            "installed_capacity_sessions_per_day": capacity,
            "capacity_utilization_percent": (100.0 * assigned / capacity) if capacity else 0.0,
            "fixed_build_cost_lakh": fixed_capex,
            "charger_purchase_cost_lakh": variable_capex,
            "total_capex_lakh": fixed_capex + variable_capex,
            "annual_fixed_opex_lakh_per_year": fixed_opex,
            "annual_charger_opex_lakh_per_year": variable_opex,
            "total_annual_opex_lakh_per_year": fixed_opex + variable_opex,
        }
        row.update(location_metadata(site_row))
        station_rows.append(row)

    allocation_columns = [
        "demand_id", "candidate_id", "site_name", "demand_sessions_per_day",
        "allocated_sessions_per_day", "distance_km",
        "weighted_distance_km_sessions_per_day",
    ]
    station_columns = [
        "candidate_id", "site_name", "open", "chargers", "assigned_sessions_per_day",
        "installed_capacity_sessions_per_day", "capacity_utilization_percent",
        "fixed_build_cost_lakh", "charger_purchase_cost_lakh", "total_capex_lakh",
        "annual_fixed_opex_lakh_per_year", "annual_charger_opex_lakh_per_year",
        "total_annual_opex_lakh_per_year",
    ]

    allocation_df = pd.DataFrame(allocation_rows)
    station_df = pd.DataFrame(station_rows)
    if allocation_df.empty:
        allocation_df = pd.DataFrame(columns=allocation_columns)
    if station_df.empty:
        station_df = pd.DataFrame(columns=station_columns)

    # Put the important mapping columns first; retain any extra location metadata after them.
    allocation_df = allocation_df[
        [c for c in allocation_columns if c in allocation_df.columns]
        + [c for c in allocation_df.columns if c not in allocation_columns]
    ]
    station_df = station_df[
        [c for c in station_columns if c in station_df.columns]
        + [c for c in station_df.columns if c not in station_columns]
    ]

    capex = float(pulp.value(result["expressions"]["capex"]))
    annual_opex = float(pulp.value(result["expressions"]["annual_opex"]))
    weighted_distance = float(pulp.value(result["expressions"]["weighted_distance"]))
    average_distance = weighted_distance / total_demand if total_demand > 0 else 0.0

    metrics = {
        "capex_lakh": capex,
        "annual_opex_lakh": annual_opex,
        "weighted_distance": weighted_distance,
        "average_distance_km": average_distance,
        "total_demand": total_demand,
    }
    return station_df, allocation_df, metrics


def print_solution(result, station_df, allocation_df, metrics, args, capex_budget_lakh):
    title = "OPEX-FIRST SOLUTION" if result["tag"] == "opex_first" else "DISTANCE-FIRST SOLUTION"
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(f"Solver status: {result['status']}")
    print(f"Total demand: {metrics['total_demand']:.2f} sessions/day")

    if capex_budget_lakh is None:
        print("CAPEX budget: Disabled")
    else:
        print(
            f"CAPEX budget: ₹{capex_budget_lakh:.2f} lakh "
            f"(₹{capex_budget_lakh / 100:.3f} crore)"
        )
    print(
        f"CAPEX used: ₹{metrics['capex_lakh']:.2f} lakh "
        f"(₹{metrics['capex_lakh'] / 100:.3f} crore)"
    )
    print(
        f"Annual OPEX: ₹{metrics['annual_opex_lakh']:.2f} lakh/year "
        f"(₹{metrics['annual_opex_lakh'] / 100:.3f} crore/year)"
    )
    print(
        "Demand-weighted travel distance: "
        f"{metrics['weighted_distance']:.2f} km·sessions/day"
    )
    print(f"Average allocated distance: {metrics['average_distance_km']:.3f} km/session")
    print(f"Primary objective: {result['primary_label']} = {result['primary_value']:.6f}")
    print(f"Secondary objective: {result['secondary_label']} = {result['secondary_value']:.6f}")

    print("\nSELECTED STATIONS AND CHARGERS")
    if station_df.empty:
        print("No stations selected.")
    else:
        display_columns = [
            c for c in [
                "candidate_id", "site_name", "chargers", "assigned_sessions_per_day",
                "installed_capacity_sessions_per_day", "capacity_utilization_percent",
                "total_capex_lakh", "total_annual_opex_lakh_per_year",
                "latitude", "longitude", "lat", "lon", "address", "location",
            ] if c in station_df.columns
        ]
        print(station_df[display_columns].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nSELECTED DEMAND-ZONE -> STATION ALLOCATIONS")
    if allocation_df.empty:
        print("No demand allocations found.")
    else:
        display_columns = [
            c for c in [
                "demand_id", "candidate_id", "site_name", "allocated_sessions_per_day",
                "distance_km", "demand_latitude", "demand_longitude",
                "demand_lat", "demand_lon", "site_latitude", "site_longitude",
                "site_lat", "site_lon",
            ] if c in allocation_df.columns
        ]
        # Show other useful fields too (for example demand_name or zone_name).
        for column in allocation_df.columns:
            if column not in display_columns and any(
                hint in column.lower()
                for hint in ("name", "address", "location", "zone", "ward", "area", "place")
            ):
                display_columns.append(column)
        print(allocation_df[display_columns].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def solve(args):
    demand = pd.read_csv(csv_path(args.demand), dtype={"demand_id": "string"})
    sites = pd.read_csv(csv_path(args.sites), dtype={"candidate_id": "string"})
    distances = pd.read_csv(
        csv_path(args.distances),
        dtype={"demand_id": "string", "candidate_id": "string"},
    )
    demand, sites, distances = validate_and_prepare_data(demand, sites, distances)

    total_demand = float(demand["ev_sessions_per_day"].sum())
    capex_budget_lakh = None if args.budget <= 0 else args.budget

    print("\nChecking operational feasibility without applying the CAPEX budget...")
    feasible, minimum_capex, feasibility_message = check_operational_feasibility(
        demand, sites, distances, args
    )
    if not feasible:
        print(f"INFEASIBLE: {feasibility_message}", file=sys.stderr)
        return 2
    if feasibility_message:
        print(feasibility_message)
    if minimum_capex is not None:
        print(f"Minimum CAPEX required by the operational constraints: ₹{minimum_capex:.2f} lakh")
        if capex_budget_lakh is not None and capex_budget_lakh + 1e-6 < minimum_capex:
            print(
                f"WARNING: Your CAPEX budget of ₹{capex_budget_lakh:.2f} lakh is below "
                f"the minimum feasible CAPEX of ₹{minimum_capex:.2f} lakh."
            )

    objectives = ["opex", "distance"] if args.objective == "both" else [args.objective]
    successful_solutions = 0

    for objective in objectives:
        result = solve_lexicographic(
            demand=demand,
            sites=sites,
            distances=distances,
            args=args,
            primary_objective=objective,
            capex_budget_lakh=capex_budget_lakh,
        )

        if result["status"] != "Optimal":
            print("\n" + "=" * 88, file=sys.stderr)
            print(f"{objective.upper()}-FIRST RUN: {result['status']}", file=sys.stderr)
            print(result.get("message", "No further details available."), file=sys.stderr)
            if capex_budget_lakh is not None and minimum_capex is not None:
                print(
                    "The unbudgeted model is operationally feasible, but this CAPEX budget "
                    "does not permit a feasible solution. Try a budget at least as high as "
                    f"₹{minimum_capex:.2f} lakh.",
                    file=sys.stderr,
                )
            continue

        station_df, allocation_df, metrics = make_solution_tables(
            result, demand, sites, total_demand
        )
        print_solution(result, station_df, allocation_df, metrics, args, capex_budget_lakh)

        station_output = output_path_with_tag(args.output_stations, result["tag"])
        allocation_output = output_path_with_tag(args.output_allocations, result["tag"])
        station_output.parent.mkdir(parents=True, exist_ok=True)
        allocation_output.parent.mkdir(parents=True, exist_ok=True)
        station_df.to_csv(station_output, index=False)
        allocation_df.to_csv(allocation_output, index=False)

        print(f"\nStation details saved to: {station_output.resolve()}")
        print(f"Allocation mapping saved to: {allocation_output.resolve()}")
        successful_solutions += 1

    return 0 if successful_solutions else 3


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize EV charging station allocation with a CAPEX-only budget "
            "and separate OPEX/distance objectives."
        )
    )
    parser.add_argument("--demand", type=Path, default=Path("demand_points.csv"))
    parser.add_argument(
        "--sites", type=Path, default=Path("candidate_site_costs_capacity.csv")
    )
    parser.add_argument("--distances", type=Path, default=Path("distance_long.csv"))
    parser.add_argument("--radius", type=float, default=10.0, help="Maximum eligible distance in km")
    parser.add_argument(
        "--budget",
        type=float,
        default=215.0,
        help="CAPEX budget in lakh rupees; use 0 to disable the budget",
    )
    parser.add_argument("--min-stations", type=int, default=1)
    parser.add_argument("--max-stations", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=["both", "opex", "distance"],
        default="both",
        help="Run OPEX-first, distance-first, or both lexicographic optimizations",
    )
    parser.add_argument(
        "--continuous-allocation",
        action="store_true",
        help="Allow fractional expected sessions; default is integer session allocation",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-stations", type=Path,
        default=script_dir / "ev_station_solution.csv",
    )
    parser.add_argument(
        "--output-allocations", type=Path,
        default=script_dir / "ev_allocation_solution.csv",
    )
    parser.add_argument("--verbose", action="store_true", help="Show CBC solver output")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        raise SystemExit(solve(arguments))
    except pulp.PulpSolverError as exc:
        print("CBC SOLVER ERROR: PuLP could not complete a CBC solver run.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        print(
            "Run the same command again with --verbose to display CBC's log. "
            "If even a tiny test MILP fails, repair/reinstall PuLP or use a "
            "separately installed CBC executable.",
            file=sys.stderr,
        )
        raise SystemExit(20)
    except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(10)
