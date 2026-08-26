import pandas as pd
import streamlit as st

from nuri.models import ProductMixProblem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp
from nuri.results import classify_zero_quantity_products
from nuri.scheduling import DAYS, SchedulingProblem, solve_schedule
from nuri.statistics.factorial_design import (
    FactorialExperiment,
    analyze_2k_factorial,
    best_combination,
    fit_reduced_model,
    confirmation_test,
    describe_combo,
)
from cases.furniture import PRODUCTS as FURNITURE_PRODUCTS, RESOURCES as FURNITURE_RESOURCES
from cases.case2_generic import PRODUCTS as CASE2_PRODUCTS, RESOURCES as CASE2_RESOURCES
from cases.case3_generic import PRODUCTS as CASE3_PRODUCTS, RESOURCES as CASE3_RESOURCES
from cases.case4_capital_allocation import (
    PRODUCTS as CASE4_PRODUCTS,
    RESOURCES as CASE4_RESOURCES,
)
from cases.case5_restaurant_scheduling import (
    STAFFING_REQUIREMENTS,
    WEEKDAY_HOURLY_WAGE,
    WEEKEND_HOURLY_WAGE,
    fully_staffed_employees,
)
from cases.case6_packaging_defects import (
    FACTOR_NAMES as DOE_FACTOR_NAMES,
    FACTOR_LABELS as DOE_FACTOR_LABELS,
    RAW_RUNS as DOE_RAW_RUNS,
    CONFIRMATION_RUNS as DOE_CONFIRMATION_RUNS,
    CURRENT_DEFECT_RATE as DOE_CURRENT_DEFECT_RATE,
    MONTHLY_VOLUME as DOE_MONTHLY_VOLUME,
    COST_PER_DEFECT as DOE_COST_PER_DEFECT,
)

st.set_page_config(page_title="Nuri — Optimization Engine", layout="centered")

# Each example: (products, resources, decision_label, unit_label). The decision/unit
# labels only change display wording -- the same LP/ILP math runs underneath
# regardless of whether this is a factory or a cash-strapped distributor.
PRODUCT_MIX_EXAMPLES = {
    "Furniture (tables/chairs/shelves)": (
        FURNITURE_PRODUCTS, FURNITURE_RESOURCES, "Recommended production", "produce",
    ),
    "Generic case 2 (x, y)": (
        CASE2_PRODUCTS, CASE2_RESOURCES, "Recommended values", "set",
    ),
    "Generic case 3 (x, y, integer)": (
        CASE3_PRODUCTS, CASE3_RESOURCES, "Recommended values", "set",
    ),
    "Distributor capital allocation": (
        CASE4_PRODUCTS, CASE4_RESOURCES, "Recommended purchase quantities", "buy",
    ),
    "Blank": (
        {"product_a": {"profit": 0, "max_demand": None}}, {"resource_a": 0},
        "Recommended production", "produce",
    ),
}


def products_to_df(products, resource_names):
    rows = []
    for name, attrs in products.items():
        row = {"product": name, "profit": attrs.get("profit", 0), "max_demand": attrs.get("max_demand")}
        for r in resource_names:
            row[r] = attrs.get(r, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def resources_to_df(resources):
    return pd.DataFrame(
        [{"resource": name, "limit": limit} for name, limit in resources.items()]
    )


def df_to_problem(products_df, resources_df, decision_label, unit_label):
    resource_names = [r for r in resources_df["resource"].tolist() if r]
    resources = {
        row["resource"]: row["limit"]
        for _, row in resources_df.iterrows()
        if row["resource"]
    }

    products = {}
    for _, row in products_df.iterrows():
        name = row["product"]
        if not name:
            continue
        attrs = {"profit": row["profit"]}
        max_demand = row.get("max_demand")
        if pd.notna(max_demand):
            attrs["max_demand"] = max_demand
        for r in resource_names:
            attrs[r] = row.get(r, 0) or 0
        products[name] = attrs

    return ProductMixProblem(
        products=products,
        resources=resources,
        decision_label=decision_label,
        unit_label=unit_label,
    )


def run_product_mix_app():
    st.caption("Define a product-mix problem: what you can make, what it earns, what it costs in shared resources.")

    choice = st.selectbox("Load an example", list(PRODUCT_MIX_EXAMPLES.keys()))
    example_products, example_resources, decision_label, unit_label = PRODUCT_MIX_EXAMPLES[choice]

    if st.session_state.get("_loaded_example") != choice:
        st.session_state["_loaded_example"] = choice
        st.session_state["decision_label"] = decision_label
        st.session_state["unit_label"] = unit_label
        st.session_state["resources_df"] = resources_to_df(example_resources)
        st.session_state["products_df"] = products_to_df(
            example_products, list(example_resources.keys())
        )

    st.subheader("Resources")
    resources_df = st.data_editor(
        st.session_state["resources_df"], num_rows="dynamic", key="resources_editor"
    )
    st.session_state["resources_df"] = resources_df

    resource_names = [r for r in resources_df["resource"].tolist() if r]

    # Keep the products table's columns in sync with current resource names.
    current_products_df = st.session_state["products_df"]
    for r in resource_names:
        if r not in current_products_df.columns:
            current_products_df[r] = 0
    base_cols = ["product", "profit", "max_demand"]
    current_products_df = current_products_df[base_cols + resource_names]
    st.session_state["products_df"] = current_products_df

    st.subheader("Products")
    products_df = st.data_editor(
        st.session_state["products_df"], num_rows="dynamic", key="products_editor"
    )
    st.session_state["products_df"] = products_df

    solver_choice = st.radio(
        "Solve as",
        ["LP (fractional units allowed)", "ILP (whole units only)"],
        horizontal=True,
    )

    if st.button("Optimize", type="primary"):
        problem = df_to_problem(
            products_df,
            resources_df,
            st.session_state["decision_label"],
            st.session_state["unit_label"],
        )
        result = solve_lp(problem) if solver_choice.startswith("LP") else solve_ilp(problem)

        if not result.success:
            st.error("No feasible solution found. Check your constraints.")
        else:
            st.subheader(result.decision_label)
            st.dataframe(
                pd.DataFrame(
                    [{"product": k, "quantity": v} for k, v in result.quantities.items()]
                ),
                hide_index=True,
            )
            st.metric("Expected contribution", f"{result.objective_value:,.2f}")

            st.subheader("Resource utilization")
            util_df = pd.DataFrame(
                [{"resource": k, "utilization": v} for k, v in result.utilization.items()]
            )
            st.dataframe(
                util_df.style.format({"utilization": "{:.1%}"}), hide_index=True
            )

            if result.binding_constraints:
                st.info(
                    f"**Binding constraint(s):** {', '.join(result.binding_constraints)}  \n"
                    "These are the resources actually limiting the solution — "
                    "relaxing them (more capacity) is what would increase profit."
                )

            if result.shadow_prices:
                st.subheader("Marginal value of each resource")
                if result.shadow_prices_are_approximate:
                    st.caption(
                        "Approximate, from the LP relaxation — whole-unit requirements make "
                        "the true marginal value jump discontinuously, so treat these as "
                        "directional, not exact."
                    )
                shadow_df = pd.DataFrame(
                    [{"resource": k, "value per extra unit": v} for k, v in result.shadow_prices.items()]
                )
                st.dataframe(
                    shadow_df.style.format({"value per extra unit": "{:,.2f}"}), hide_index=True
                )

            skipped, tied = classify_zero_quantity_products(result.quantities, result.reduced_costs)

            if skipped:
                st.subheader("Why other products weren't recommended")
                for name, cost in skipped.items():
                    st.write(
                        f"**{name}**: making one would cost **{-cost:,.2f}** more in resources "
                        "(at their marginal value) than it earns in profit."
                    )

            if tied:
                st.subheader("Exactly tied with the marginal value of resources")
                for name in tied:
                    st.write(
                        f"**{name}**: ties exactly with the marginal value of the resources it "
                        "would use; including it wouldn't change profit."
                    )


def employees_to_df(employees):
    rows = []
    for name, days in employees.items():
        row = {"employee": name}
        for d in DAYS:
            row[d] = d in days
        rows.append(row)
    return pd.DataFrame(rows)


def requirements_to_df(requirements):
    return pd.DataFrame(
        [
            {"day": d, "morning": requirements[d]["morning"], "evening": requirements[d]["evening"]}
            for d in DAYS
        ]
    )


def df_to_scheduling_problem(
    employees_df, requirements_df, max_hours, with_wages, weekday_wage, weekend_wage
):
    employees = {}
    for _, row in employees_df.iterrows():
        name = row["employee"]
        if not name:
            continue
        employees[name] = [d for d in DAYS if row.get(d)]

    requirements = {}
    for _, row in requirements_df.iterrows():
        day = row["day"]
        if day not in DAYS:
            continue
        requirements[day] = {"morning": row["morning"], "evening": row["evening"]}

    return SchedulingProblem(
        employees=employees,
        staffing_requirements=requirements,
        max_hours_per_employee=max_hours,
        weekday_hourly_wage=weekday_wage if with_wages else None,
        weekend_hourly_wage=weekend_wage if with_wages else None,
    )


def run_scheduling_app():
    st.caption(
        "Define a workforce scheduling problem: who's available when, how many staff "
        "each shift needs, and what the hour/cost limits are."
    )

    if "schedule_employees_df" not in st.session_state:
        st.session_state["schedule_employees_df"] = employees_to_df(fully_staffed_employees())
        st.session_state["schedule_requirements_df"] = requirements_to_df(STAFFING_REQUIREMENTS)

    st.subheader("Employee availability")
    st.caption("Restaurant hours are 10 AM-10 PM, split into a morning and evening period.")
    employees_df = st.data_editor(
        st.session_state["schedule_employees_df"], num_rows="dynamic", key="schedule_employees_editor"
    )
    st.session_state["schedule_employees_df"] = employees_df

    st.subheader("Minimum staff required per period")
    requirements_df = st.data_editor(
        st.session_state["schedule_requirements_df"], num_rows="fixed", key="schedule_requirements_editor"
    )
    st.session_state["schedule_requirements_df"] = requirements_df

    max_hours = st.number_input("Max hours per employee per week", min_value=1, value=36)

    objective = st.radio(
        "Minimize",
        ["Total labour hours", "Total labour cost"],
        horizontal=True,
    )

    weekday_wage, weekend_wage = WEEKDAY_HOURLY_WAGE, WEEKEND_HOURLY_WAGE
    with_wages = objective == "Total labour cost"
    if with_wages:
        col1, col2 = st.columns(2)
        weekday_wage = col1.number_input("Weekday hourly wage", min_value=0, value=WEEKDAY_HOURLY_WAGE)
        weekend_wage = col2.number_input("Weekend hourly wage", min_value=0, value=WEEKEND_HOURLY_WAGE)

    if st.button("Solve schedule", type="primary"):
        problem = df_to_scheduling_problem(
            employees_df, requirements_df, max_hours, with_wages, weekday_wage, weekend_wage
        )
        result = solve_schedule(problem, minimize="cost" if with_wages else "hours")

        if not result.success:
            st.error(
                "No feasible schedule found. The staffing requirements can't be met "
                "with this team under these hour limits — try relaxing a requirement, "
                "raising the hour cap, or adding staff."
            )
        else:
            st.subheader("Schedule summary")
            st.metric("Total labour hours", f"{result.total_hours:.0f}")
            if result.total_cost is not None:
                st.metric("Total labour cost", f"{result.total_cost:,.2f}")

            st.subheader("Hours per employee")
            emp_df = pd.DataFrame(
                [
                    {
                        "employee": e,
                        "hours": result.hours_by_employee[e],
                        "days worked": result.days_worked_by_employee[e],
                        **(
                            {"cost": result.cost_by_employee[e]}
                            if result.cost_by_employee
                            else {}
                        ),
                    }
                    for e in result.hours_by_employee
                ]
            )
            st.dataframe(emp_df, hide_index=True)

            if result.unused_employees:
                st.info(f"Not needed at all: {', '.join(result.unused_employees)}")
            else:
                st.info("Every employee is used at least once.")

            if result.binding_requirements:
                st.subheader("Binding requirements (zero slack)")
                st.caption(
                    "These day/period minimums are met exactly, with no spare capacity — "
                    "if anyone can't work, these are the first slots to fall short."
                )
                binding_df = pd.DataFrame(
                    [{"day": d, "period": p} for d, p in result.binding_requirements]
                )
                st.dataframe(binding_df, hide_index=True)

            with st.expander("Full assignment (who works what, when)"):
                assign_df = pd.DataFrame(
                    result.assignments, columns=["employee", "day", "shift"]
                )
                st.dataframe(assign_df, hide_index=True)


def factors_to_df(factor_labels):
    return pd.DataFrame(
        [
            {"factor": name, "low_label": labels[0], "high_label": labels[1]}
            for name, labels in factor_labels.items()
        ]
    )


def runs_to_df(runs, factor_names):
    rows = []
    for run in runs:
        row = {f: run[f] for f in factor_names}
        row["produced"] = run["produced"]
        row["rejected"] = run["rejected"]
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_runs(runs_df, factor_names):
    runs = []
    for _, row in runs_df.iterrows():
        if pd.isna(row.get("produced")) or pd.isna(row.get("rejected")):
            continue
        run = {f: int(row[f]) for f in factor_names}
        run["produced"] = int(row["produced"])
        run["rejected"] = int(row["rejected"])
        runs.append(run)
    return runs


def level_column_config(factor_names):
    return {
        f: st.column_config.SelectboxColumn(f, options=[-1, 1], required=True)
        for f in factor_names
    }


def run_doe_app():
    st.caption(
        "Design of Experiments: given raw results from a 2^k factorial experiment "
        "(every combination of two-level factors, replicated), find which factors "
        "and interactions actually matter, and the setting combination that minimizes "
        "the defect rate."
    )

    if "doe_factors_df" not in st.session_state:
        st.session_state["doe_factors_df"] = factors_to_df(DOE_FACTOR_LABELS)
        st.session_state["doe_runs_df"] = runs_to_df(DOE_RAW_RUNS, DOE_FACTOR_NAMES)
        st.session_state["doe_confirmation_df"] = runs_to_df(DOE_CONFIRMATION_RUNS, DOE_FACTOR_NAMES)

    st.subheader("Factors")
    st.caption("factor = a short code (A, B, C...) used as the column name in the runs table below.")
    factors_df = st.data_editor(
        st.session_state["doe_factors_df"], num_rows="dynamic", key="doe_factors_editor"
    )
    st.session_state["doe_factors_df"] = factors_df

    factor_names = [f for f in factors_df["factor"].tolist() if f]
    factor_labels = {
        row["factor"]: (row["low_label"], row["high_label"])
        for _, row in factors_df.iterrows()
        if row["factor"]
    }

    # Keep the runs tables' columns in sync with the current factor list.
    for key in ("doe_runs_df", "doe_confirmation_df"):
        df = st.session_state[key]
        for f in factor_names:
            if f not in df.columns:
                df[f] = -1
        df = df[factor_names + ["produced", "rejected"]]
        st.session_state[key] = df

    st.subheader("Experimental runs (raw data)")
    st.caption("Factor columns use -1 (low) / 1 (high) coding. Include every replicate as its own row.")
    runs_df = st.data_editor(
        st.session_state["doe_runs_df"],
        num_rows="dynamic",
        key="doe_runs_editor",
        column_config=level_column_config(factor_names),
    )
    st.session_state["doe_runs_df"] = runs_df

    scale = st.radio(
        "Analysis scale",
        ["logit (recommended for wide-ranging rates)", "raw rate (naive)"],
        horizontal=True,
    )
    alpha = st.number_input("Significance level (alpha)", min_value=0.001, max_value=0.5, value=0.05, step=0.01)

    st.subheader("Business impact (optional)")
    col1, col2, col3 = st.columns(3)
    current_rate = col1.number_input("Current defect rate", min_value=0.0, max_value=1.0, value=DOE_CURRENT_DEFECT_RATE, format="%.4f")
    monthly_volume = col2.number_input("Monthly volume (units)", min_value=0, value=DOE_MONTHLY_VOLUME)
    cost_per_defect = col3.number_input("Cost per defective unit", min_value=0.0, value=float(DOE_COST_PER_DEFECT))

    with st.expander("Confirmation run (optional)"):
        st.caption("A fresh run at the recommended settings, to check the prediction out-of-sample.")
        confirmation_df = st.data_editor(
            st.session_state["doe_confirmation_df"],
            num_rows="dynamic",
            key="doe_confirmation_editor",
            column_config=level_column_config(factor_names),
        )
        st.session_state["doe_confirmation_df"] = confirmation_df

    if st.button("Analyze experiment", type="primary"):
        runs = df_to_runs(runs_df, factor_names)
        experiment = FactorialExperiment(
            factor_names=factor_names, factor_labels=factor_labels, runs=runs
        )
        scale_key = "logit" if scale.startswith("logit") else "rate"
        analysis = analyze_2k_factorial(experiment, alpha=alpha, scale=scale_key)

        st.subheader("Effect significance")
        st.caption(f"F-test against pure error (replicate spread), {scale_key} scale, {len(analysis['effects'])} simultaneous tests.")
        effects_df = pd.DataFrame(
            [
                {
                    "effect": eff,
                    "estimate": r["effect"],
                    "F": r["f"],
                    "p": r["p"],
                    "significant (p<alpha)": r["significant"],
                    "survives Bonferroni": r["significant_bonferroni"],
                }
                for eff, r in sorted(analysis["effects"].items(), key=lambda kv: kv[1]["p"])
            ]
        )
        st.dataframe(
            effects_df.style.format({"estimate": "{:+.4f}", "F": "{:.2f}", "p": "{:.4f}"}),
            hide_index=True,
        )

        best_run, best_rate = best_combination(analysis)
        significant, predictions = fit_reduced_model(analysis)
        best_key = tuple(best_run[f] for f in factor_names)
        predicted_rate = predictions[best_key]

        st.subheader("Recommended settings")
        readable = describe_combo(best_run, factor_labels) if factor_labels else best_run
        st.dataframe(
            pd.DataFrame([{"factor": k, "setting": v} for k, v in readable.items()]),
            hide_index=True,
        )
        st.metric("Best observed defect rate", f"{best_rate:.2%}")
        st.metric("Reduced-model predicted defect rate", f"{predicted_rate:.2%}")
        st.caption(f"Reduced model uses only the Bonferroni-significant effects: {significant or 'none'}")

        if current_rate > 0 and monthly_volume > 0:
            st.subheader("Business impact")
            monthly_defects_current = current_rate * monthly_volume
            monthly_defects_new = predicted_rate * monthly_volume
            monthly_savings = (monthly_defects_current - monthly_defects_new) * cost_per_defect
            c1, c2, c3 = st.columns(3)
            c1.metric("Current monthly defects", f"{monthly_defects_current:,.0f}")
            c2.metric("Projected monthly defects", f"{monthly_defects_new:,.0f}")
            c3.metric("Projected monthly savings", f"{monthly_savings:,.2f}")

        confirmation_runs = df_to_runs(confirmation_df, factor_names)
        if confirmation_runs:
            st.subheader("Confirmation run")
            conf = confirmation_test(predicted_rate, confirmation_runs, alpha=alpha)
            c1, c2 = st.columns(2)
            c1.metric("Observed rate", f"{conf['observed_rate']:.2%}")
            c2.metric("Predicted rate", f"{conf['predicted_rate']:.2%}")
            ci = conf["confidence_interval"]
            st.caption(f"{1-alpha:.0%} CI on observed rate: ({ci.low:.2%}, {ci.high:.2%}); p={conf['p_value']:.4f}")
            if conf["supports_recommendation"]:
                st.success("The confirmation run is statistically consistent with the prediction.")
            else:
                st.warning(
                    "The confirmation run's observed rate is significantly different from the "
                    "prediction — the model may not generalize as well as the design-stage "
                    "analysis suggested."
                )


st.title("Nuri — Optimization Engine")

problem_type = st.selectbox(
    "Problem type", ["Product mix (LP/ILP)", "Workforce scheduling", "Factorial DOE (statistics)"]
)

st.divider()

if problem_type == "Product mix (LP/ILP)":
    run_product_mix_app()
elif problem_type == "Workforce scheduling":
    run_scheduling_app()
else:
    run_doe_app()
