import pytest

from cases.case9_pp_sack_factory import pp_sack_factory_problem
from nuri.lp import solve_lp


def test_matches_client_confirmed_optimum():
    result = solve_lp(pp_sack_factory_problem())

    assert result.success
    assert result.quantities["standard_50kg"] == pytest.approx(0, abs=1)
    assert result.quantities["printed_50kg"] == pytest.approx(41_248, abs=1)
    assert result.quantities["fertilizer_bag"] == pytest.approx(21_934, abs=1)
    assert result.quantities["feed_bag"] == pytest.approx(18_071, abs=1)
    assert result.objective_value == pytest.approx(19_844_242, abs=5)


def test_loom_and_printing_bind_polypropylene_does_not():
    result = solve_lp(pp_sack_factory_problem())

    assert set(result.binding_constraints) == {"loom", "printing"}
    assert result.utilization["polypropylene"] < 0.9


def test_standard_sack_correctly_excluded_by_reduced_cost():
    # Standard sack is feasible and has positive margin, but its reduced
    # cost should be negative -- it costs more in scarce loom/printing
    # capacity (at their marginal value) than it earns.
    result = solve_lp(pp_sack_factory_problem())
    assert result.reduced_costs["standard_50kg"] < 0


def test_printed_sack_is_not_the_best_use_of_either_single_bottleneck_alone():
    # The real lesson: printed sack has the highest raw contribution per
    # bag, but is beaten by feed bag on profit-per-loom-hour and by
    # fertilizer bag on profit-per-printing-hour. It's only optimal to
    # fill its full demand because of how both constraints interact
    # together -- neither single-resource heuristic gets the right answer.
    products = {
        "printed_50kg": {"profit": 263, "loom": 0.0020, "printing": 0.0008},
        "fertilizer_bag": {"profit": 219, "loom": 0.0017, "printing": 0.0006},
        "feed_bag": {"profit": 232, "loom": 0.0016, "printing": 0.0010},
    }
    per_loom = {p: v["profit"] / v["loom"] for p, v in products.items()}
    per_printing = {p: v["profit"] / v["printing"] for p, v in products.items()}

    assert per_loom["feed_bag"] > per_loom["printed_50kg"]
    assert per_printing["fertilizer_bag"] > per_printing["printed_50kg"]
