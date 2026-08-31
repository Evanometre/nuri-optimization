"""Single-period production/stocking quantity under uncertain demand --
the "newsvendor" problem, extended with the operational constraints a real
factory actually has: starting inventory, normal + overtime capacity (with
an overtime cost premium), and a soft cap on ending inventory.

Core idea: demand is Normal(mu, sigma). Every unit of available stock
(starting inventory + production) that goes unsold costs Co (holding).
Every unit of demand that can't be met costs Cu (stockout). The classic
result: the cost-minimizing quantity of AVAILABLE STOCK is the
critical_ratio = Cu / (Cu + Co) quantile of the demand distribution --
not the mean, and not a "safety stock on top of the mean" rule of thumb.
When Cu >> Co (as here), that quantile sits well above the mean: the
math says lean toward "produce more," not because more is always safer,
but because the two error types cost very differently.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


def critical_ratio(underage_cost, overage_cost):
    return underage_cost / (underage_cost + overage_cost)


def normal_loss(z):
    """Standard unit-normal loss function L(z) = E[(Z - z)^+] for Z ~ N(0,1).
    Used to get closed-form expected shortage/excess without simulation."""
    return stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))


def optimal_available_stock(mu, sigma, underage_cost, overage_cost):
    """The unconstrained cost-minimizing quantity of total available stock
    (starting inventory + production), ignoring capacity/inventory caps."""
    cr = critical_ratio(underage_cost, overage_cost)
    z = stats.norm.ppf(cr)
    return mu + z * sigma, z, cr


def expected_costs(available_stock, mu, sigma, underage_cost, overage_cost):
    """Expected shortage/excess units and their cost, for ANY candidate
    available-stock level -- not just the optimum. This is what lets us
    compare the recommended quantity against alternatives on a like-for-like
    basis (same formulas, different Q)."""
    z = (available_stock - mu) / sigma
    l_z = normal_loss(z)
    expected_shortage = sigma * l_z
    expected_excess = (available_stock - mu) + expected_shortage

    return {
        "available_stock": available_stock,
        "expected_shortage_units": expected_shortage,
        "expected_excess_units": expected_excess,
        "expected_shortage_cost": expected_shortage * underage_cost,
        "expected_excess_cost": expected_excess * overage_cost,
        "expected_total_cost": expected_shortage * underage_cost + expected_excess * overage_cost,
        "prob_stockout": 1 - stats.norm.cdf(z),
    }


@dataclass
class ProductionRecommendation:
    mu: float
    sigma: float
    underage_cost: float
    overage_cost: float
    starting_inventory: float
    normal_capacity: float
    overtime_capacity: float
    overtime_extra_cost: float
    max_ending_inventory: float = None

    def unconstrained_optimum(self):
        available_stock, z, cr = optimal_available_stock(
            self.mu, self.sigma, self.underage_cost, self.overage_cost
        )
        return available_stock, z, cr

    def recommend(self):
        available_stock_star, z, cr = self.unconstrained_optimum()
        production_unconstrained = available_stock_star - self.starting_inventory

        max_available_stock = self.starting_inventory + self.overtime_capacity
        capacity_binding = production_unconstrained > self.overtime_capacity

        production = min(max(production_unconstrained, 0), self.overtime_capacity)
        available_stock = self.starting_inventory + production

        overtime_units = max(0, production - self.normal_capacity)
        overtime_cost = overtime_units * self.overtime_extra_cost

        costs = expected_costs(available_stock, self.mu, self.sigma, self.underage_cost, self.overage_cost)

        result = {
            "critical_ratio": cr,
            "z": z,
            "available_stock_unconstrained_optimum": available_stock_star,
            "production_unconstrained": production_unconstrained,
            "capacity_binding": capacity_binding,
            "production": production,
            "available_stock": available_stock,
            "overtime_units": overtime_units,
            "overtime_cost": overtime_cost,
            **costs,
        }
        result["expected_total_cost_including_overtime"] = costs["expected_total_cost"] + overtime_cost

        if self.max_ending_inventory is not None:
            z_cap = (self.max_ending_inventory - (available_stock - self.mu)) / self.sigma
            # ending inventory = available_stock - demand, when demand < available_stock
            # P(ending inventory > cap) = P(demand < available_stock - cap)
            prob_breach = stats.norm.cdf((available_stock - self.max_ending_inventory - self.mu) / self.sigma)
            result["prob_ending_inventory_exceeds_cap"] = prob_breach

        return result

    def explain_calculation(self):
        """The core calculation chain, spelled out explicitly rather than
        buried in the result dict:
          September demand distribution -> critical-ratio percentile = X
          -> X - starting inventory = production
        """
        available_stock_star, z, cr = self.unconstrained_optimum()
        production_unconstrained = available_stock_star - self.starting_inventory
        return (
            f"Demand distribution: Normal(mu={self.mu:,.0f}, sigma={self.sigma:,.0f})\n"
            f"Critical ratio = Cu/(Cu+Co) = {self.underage_cost:g}/({self.underage_cost:g}+{self.overage_cost:g}) "
            f"= {cr:.4f} ({cr:.2%})\n"
            f"-> z = {z:.4f} standard deviations (norm.ppf({cr:.4f}))\n"
            f"-> {cr:.2%} percentile of demand = mu + z*sigma = "
            f"{self.mu:,.0f} + {z:.4f} x {self.sigma:,.0f} = {available_stock_star:,.0f}\n"
            f"-> Production = {available_stock_star:,.0f} (percentile) - {self.starting_inventory:,.0f} "
            f"(starting inventory) = {production_unconstrained:,.0f}"
        )

    def evaluate(self, production):
        """Expected-cost breakdown for a specific candidate production
        quantity, for comparing against the recommendation."""
        available_stock = self.starting_inventory + production
        overtime_units = max(0, production - self.normal_capacity)
        overtime_cost = overtime_units * self.overtime_extra_cost
        costs = expected_costs(available_stock, self.mu, self.sigma, self.underage_cost, self.overage_cost)
        costs["production"] = production
        costs["overtime_units"] = overtime_units
        costs["overtime_cost"] = overtime_cost
        costs["expected_total_cost_including_overtime"] = costs["expected_total_cost"] + overtime_cost
        if self.max_ending_inventory is not None:
            costs["prob_ending_inventory_exceeds_cap"] = stats.norm.cdf(
                (available_stock - self.max_ending_inventory - self.mu) / self.sigma
            )
        return costs
