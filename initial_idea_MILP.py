import argparse
from email import parser
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pulp


def csv_path(filename: Path) -> Path:
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


def build_model(demand, sites, distances, budget, alpha, radius, min_stations, max_stations):
    demand_ids = demand["demand_id"].astype(str).tolist()
    site_ids = sites["candidate_id"].astype(str).tolist()

    model = pulp.LpProblem("EV_station_allocation", pulp.LpMinimize)
    opened = {
        site_id: pulp.LpVariable(f"open_{site_id}", cat=pulp.LpBinary)
        for site_id in site_ids
    }
    chargers = {
        site_id: pulp.LpVariable(
            f"chargers_{site_id}", lowBound=0, cat=pulp.LpInteger
        )
        for site_id in site_ids
    }
    eligible = {
        (row["demand_id"], row["candidate_id"]): float(row["distance_km"])
        for _, row in distances.iterrows()
        if float(row["distance_km"]) <= radius
    }

    x = {}
    for demand_id in demand_ids:
        for site_id in site_ids:
            if (demand_id, site_id) in eligible:
                x[demand_id, site_id] = pulp.LpVariable(
                    f"x_{demand_id}_{site_id}",
                    lowBound=0,
                    cat=pulp.LpInteger,
                )

    demand_lookup = demand.set_index("demand_id")
    site_lookup = sites.set_index("candidate_id")

    fixed_cost = {
        site_id: float(
            site_lookup.loc[site_id, "fixed_build_cost_lakh"]
            + site_lookup.loc[site_id, "annual_fixed_operating_cost_lakh"]
        )
        for site_id in site_ids
    }
    charger_cost = {
        site_id: float(
            site_lookup.loc[site_id, "cost_per_charger_lakh"]
            + site_lookup.loc[site_id, "annual_operating_cost_per_charger_lakh"]
        )
        for site_id in site_ids
    }

    model += pulp.lpSum(fixed_cost[j] * opened[j] + charger_cost[j] * chargers[j] for j in site_ids) + alpha * pulp.lpSum(
        eligible[i, j] * x[i, j] for i, j in x
    )

    for i in demand_ids:
        eligible_at_zone = [x[i, j] for j in site_ids if (i, j) in x]
        if not eligible_at_zone:
            return model, opened, chargers, x, None, fixed_cost, charger_cost, f"No site within {radius} km of demand zone {i}"
        model += (
            pulp.lpSum(eligible_at_zone)
            == float(demand_lookup.loc[i, "ev_sessions_per_day"]),
            f"demand_coverage_{i}",
        )

    for j in site_ids:
        capacity = float(site_lookup.loc[j, "capacity_per_charger_sessions_per_day"])
        max_chargers = int(site_lookup.loc[j, "max_chargers"])
        model += (
            pulp.lpSum(x[i, j] for i in demand_ids if (i, j) in x)
            <= capacity * chargers[j],
            f"capacity_{j}",
        )
        model += chargers[j] <= max_chargers * opened[j], f"linking_{j}"
        model += chargers[j] >= opened[j], f"minimum_one_{j}"

    model += pulp.lpSum(opened[j] for j in site_ids) >= min_stations
    model += pulp.lpSum(opened[j] for j in site_ids) <= max_stations

    if budget is not None:
        model += (
            pulp.lpSum(
                fixed_cost[j] * opened[j] + charger_cost[j] * chargers[j]
                for j in site_ids
            )
            <= budget,
            "capital_budget",
        )

    return model, opened, chargers, x, eligible, fixed_cost, charger_cost, None


def solve(args):
    demand = pd.read_csv(csv_path(args.demand))
    sites = pd.read_csv(csv_path(args.sites))
    distances = pd.read_csv(csv_path(args.distances))

    demand["demand_id"] = demand["demand_id"].astype(str)
    sites["candidate_id"] = sites["candidate_id"].astype(str)
    distances["demand_id"] = distances["demand_id"].astype(str)
    distances["candidate_id"] = distances["candidate_id"].astype(str)

    available_demand = set(demand["demand_id"])
    available_sites = set(sites["candidate_id"])
    distances = distances[
        distances["demand_id"].isin(available_demand)
        & distances["candidate_id"].isin(available_sites)
    ].copy()
    total_demand = float(demand["ev_sessions_per_day"].sum())

    # Prove basic operational feasibility before applying the optional budget.
    relaxed, opened_r, chargers_r, x_r, eligible_r, fixed_r, charger_r, error = build_model(
        demand, sites, distances, None, args.alpha, args.radius,
        args.min_stations, args.max_stations
    )
    if error:
        print(f"INFEASIBLE: {error}", file=sys.stderr)
        return 1
    relaxed_solver = pulp.PULP_CBC_CMD(msg=args.verbose)
    relaxed_status = relaxed.solve(relaxed_solver)
    if pulp.LpStatus[relaxed_status] != "Optimal":
        print(
            "INFEASIBLE under distance and capacity constraints: "
            f"{pulp.LpStatus[relaxed_status]}",
            file=sys.stderr,
        )
        return 2

    model, opened, chargers, x, eligible, fixed_cost, charger_cost, error = build_model(
        demand, sites, distances, args.budget, args.alpha, args.radius,
        args.min_stations, args.max_stations
    )
    if error:
        print(f"INFEASIBLE: {error}", file=sys.stderr)
        return 1
    status = model.solve(relaxed_solver)
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        print(
            f"INFEASIBLE with budget {args.budget:.2f}: "
            f"unbudgeted solution is operationally feasible. {status_name}",
            file=sys.stderr,
        )
        return 3

    rows = []
    for i in demand["demand_id"]:
        for j in sites["candidate_id"]:
            if (i, j) in x and x[i, j].value() and x[i, j].value() > 1e-6:
                rows.append({
                    "demand_id": i,
                    "candidate_id": j,
                    "sessions_per_day": round(x[i, j].value()),
                    "distance_km": eligible[i, j],
                })

    station_rows = []
    for j in sites["candidate_id"]:
        if opened[j].value() and opened[j].value() > 1e-6:
            station_rows.append({
                "candidate_id": j,
                "site_name": sites.loc[sites["candidate_id"] == j, "site_name"].iloc[0],
                "open": 1,
                "chargers": round(chargers[j].value()),
            })

    pd.DataFrame(rows).to_csv(args.output_allocations, index=False)
    pd.DataFrame(station_rows).to_csv(args.output_stations, index=False)

    infrastructure_cost = sum(
        fixed_cost[j] * opened[j].value() + charger_cost[j] * chargers[j].value()
        for j in sites["candidate_id"]
    )
    travel_cost = sum(eligible[i, j] * x[i, j].value() for i, j in x if x[i, j].value())
    print(f"Status: {status_name}")
    print(f"Total demand: {total_demand:.0f} sessions/day")
    print(f"Infrastructure and charger cost: {infrastructure_cost:.2f} lakh")
    print(f"Travel-distance penalty: {args.alpha * travel_cost:.2f}")
    print(f"Total objective: {pulp.value(model.objective):.2f}")
    print(f"Stations written to: {args.output_stations}")
    print(f"Allocations written to: {args.output_allocations}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize EV charging station allocation")
    parser.add_argument("--demand", type=Path, default=Path("demand_points.csv"))
    parser.add_argument("--sites", type=Path, default=Path("candidate_site_costs_capacity.csv"))
    parser.add_argument("--distances", type=Path, default=Path("distance_long.csv"))
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Travel-distance penalty weight")
    parser.add_argument("--budget", type=float, default=500.0, help="Budget in lakh; use 0 to disable")
    parser.add_argument("--min-stations", type=int, default=1)
    parser.add_argument("--max-stations", type=int, default=8)
    parser.add_argument("--output-stations", default="ev_station_solution.csv")
    parser.add_argument("--output-allocations", default="ev_allocation_solution.csv")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.budget <= 0:
        args.budget = None
    raise SystemExit(solve(args))
