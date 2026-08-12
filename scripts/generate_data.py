from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


ROW_COUNT = 10_000
UNIQUE_ROWS = 9_985
DUPLICATE_ROWS = ROW_COUNT - UNIQUE_ROWS
OUTPUT_FILE = Path("sme_gym_sales_2015_2025.csv")

YEAR_COUNTS = {
    2015: 550,
    2016: 590,
    2017: 630,
    2018: 680,
    2019: 740,
    2020: 320,
    2021: 600,
    2022: 1_000,
    2023: 1_600,
    2024: 1_450,
    2025: 1_825,
}

MONTH_WEIGHTS = {
    1: 1.10,
    2: 1.06,
    3: 1.04,
    4: 1.02,
    5: 1.00,
    6: 0.98,
    7: 0.94,
    8: 0.96,
    9: 1.00,
    10: 1.02,
    11: 1.05,
    12: 0.93,
}

INFLATION = {
    2015: 1.00,
    2016: 1.02,
    2017: 1.04,
    2018: 1.07,
    2019: 1.10,
    2020: 1.12,
    2021: 1.14,
    2022: 1.20,
    2023: 1.25,
    2024: 1.30,
    2025: 1.35,
}


def create_reference_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    products = [
        (
            "Membership",
            "Monthly Basic Membership",
            "Basic Monthly",
            45.00,
            0.22,
            True,
            0.20,
        ),
        (
            "Membership",
            "Monthly Premium Membership",
            "Premium Monthly",
            65.00,
            0.27,
            True,
            0.10,
        ),
        (
            "Membership",
            "Annual Basic Membership",
            "Basic Annual",
            450.00,
            0.23,
            False,
            0.05,
        ),
        (
            "Membership",
            "Annual Premium Membership",
            "Premium Annual",
            650.00,
            0.26,
            False,
            0.03,
        ),
        (
            "Membership",
            "Student Membership",
            "Student Monthly",
            32.00,
            0.20,
            True,
            0.07,
        ),
        (
            "Membership",
            "Corporate Membership",
            "Corporate Annual",
            480.00,
            0.24,
            True,
            0.04,
        ),
        (
            "Personal Training",
            "Single Personal Training Session",
            "Not Applicable",
            42.00,
            0.40,
            False,
            0.10,
        ),
        (
            "Personal Training",
            "10-Session Personal Training Package",
            "Not Applicable",
            360.00,
            0.38,
            False,
            0.05,
        ),
        (
            "Group Class",
            "Yoga Class",
            "Not Applicable",
            12.00,
            0.34,
            False,
            0.05,
        ),
        (
            "Group Class",
            "Spin Class",
            "Not Applicable",
            14.00,
            0.35,
            False,
            0.05,
        ),
        (
            "Group Class",
            "HIIT Class",
            "Not Applicable",
            15.00,
            0.36,
            False,
            0.05,
        ),
        (
            "Supplement",
            "Protein Powder",
            "Not Applicable",
            32.00,
            0.61,
            False,
            0.06,
        ),
        (
            "Supplement",
            "Energy Drink",
            "Not Applicable",
            3.50,
            0.58,
            False,
            0.05,
        ),
        (
            "Merchandise",
            "Gym T-Shirt",
            "Not Applicable",
            22.00,
            0.55,
            False,
            0.03,
        ),
        (
            "Merchandise",
            "Gym Bag",
            "Not Applicable",
            28.00,
            0.57,
            False,
            0.02,
        ),
        (
            "Joining Fee",
            "Joining Fee",
            "Not Applicable",
            30.00,
            0.10,
            False,
            0.04,
        ),
        (
            "Online Fitness",
            "Online Fitness Subscription",
            "Online Monthly",
            15.00,
            0.15,
            True,
            0.06,
        ),
    ]

    catalog = pd.DataFrame(
        products,
        columns=[
            "product_category",
            "product_name",
            "membership_plan",
            "base_price",
            "cost_ratio",
            "recurring_eligible",
            "base_weight",
        ],
    )

    segment_sizes = {
        "Student": 500,
        "Young Professional": 800,
        "Family": 450,
        "Corporate": 300,
        "Senior": 300,
        "General": 650,
    }

    customer_pools: dict[str, np.ndarray] = {}
    for segment, size in segment_sizes.items():
        customer_pools[segment] = np.array(
            [
                f"CUST-{uuid4().hex.upper()}"
                for _ in range(size)
            ],
            dtype=object,
        )
    return catalog, customer_pools


def apply_business_events(
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
) -> np.ndarray:
    result = weights.astype(float).copy()

    year = dates.year
    month = dates.month

    result[(year == 2020) & (month == 3)] *= 0.45
    result[
        (year == 2020) & np.isin(month, [4, 5, 6])
    ] *= 0.12
    result[
        (year == 2020) & np.isin(month, [7, 8])
    ] *= 0.45
    result[
        (year == 2020) & np.isin(month, [9, 10])
    ] *= 0.75
    result[
        (year == 2020) & np.isin(month, [11, 12])
    ] *= 0.35

    result[
        (year == 2021) & np.isin(month, [1, 2])
    ] *= 0.45
    result[(year == 2021) & (month == 3)] *= 0.65
    result[
        (year == 2021) & np.isin(month, [4, 5, 6])
    ] *= 0.85

    # A one-off corporate renewal drive. This creates a clear but less severe
    # monthly revenue spike than the leading critical anomaly, so the anomaly
    # pipeline has a secondary event to classify as a warning.
    result[(year == 2024) & (month == 10)] *= 1.55

    return result


def generate_transaction_dates(
    rng: np.random.Generator,
) -> pd.DatetimeIndex:
    sampled: list[np.datetime64] = []

    for year, count in YEAR_COUNTS.items():
        dates = pd.date_range(
            f"{year}-01-01",
            f"{year}-12-31",
            freq="D",
        )

        weights = np.array(
            [MONTH_WEIGHTS[month] for month in dates.month],
            dtype=float,
        )

        weekday = dates.dayofweek.to_numpy()

        weights *= np.where(
            weekday < 5,
            1.15,
            np.where(weekday == 5, 0.70, 0.45),
        )

        weights = apply_business_events(dates, weights)
        weights /= weights.sum()

        chosen = rng.choice(
            dates.to_numpy(),
            size=count,
            replace=True,
            p=weights,
        )

        sampled.extend(chosen)

    result = np.sort(
        np.asarray(sampled, dtype="datetime64[ns]")
    )

    result[0] = np.datetime64("2015-01-01")
    result[-1] = np.datetime64("2025-12-31")

    return pd.DatetimeIndex(result)


def product_probabilities(
    catalog: pd.DataFrame,
    date: pd.Timestamp,
) -> np.ndarray:
    weights = catalog["base_weight"].to_numpy(float).copy()
    names = catalog["product_name"].to_numpy()
    categories = catalog["product_category"].to_numpy()

    year = date.year
    month = date.month

    online = names == "Online Fitness Subscription"

    physical = np.isin(
        categories,
        [
            "Membership",
            "Personal Training",
            "Group Class",
            "Joining Fee",
        ],
    )

    premium_growth = np.isin(
        names,
        [
            "Monthly Premium Membership",
            "Annual Basic Membership",
            "Annual Premium Membership",
            "Corporate Membership",
            "10-Session Personal Training Package",
        ],
    )

    if year < 2020:
        weights[online] = 0.0

    if year <= 2017:
        elapsed = year - 2015

        weights[
            np.isin(
                names,
                [
                    "Monthly Premium Membership",
                    "Annual Premium Membership",
                ],
            )
        ] *= 1 + 0.10 * elapsed

        weights[
            categories == "Personal Training"
        ] *= 1 + 0.08 * elapsed

    if year in (2018, 2019):
        weights[premium_growth] *= 1.40
        weights[categories == "Personal Training"] *= 1.18

    if year == 2020 and month == 3:
        weights[physical] *= 0.45
        weights[online] *= 5.0

        weights[
            np.isin(
                categories,
                ["Supplement", "Merchandise"],
            )
        ] *= 1.35

    if year == 2020 and month in (4, 5, 6):
        weights[physical] *= 0.10
        weights[online] *= 12.0

        weights[
            np.isin(
                categories,
                ["Supplement", "Merchandise"],
            )
        ] *= 1.80

    if year == 2020 and month in (7, 8, 9, 10):
        weights[categories == "Membership"] *= 0.65
        weights[categories == "Personal Training"] *= 0.60
        weights[categories == "Group Class"] *= 0.35
        weights[online] *= 5.0

        weights[
            np.isin(
                categories,
                ["Supplement", "Merchandise"],
            )
        ] *= 1.30

    second_lockdown = (
        year == 2020 and month in (11, 12)
    ) or (
        year == 2021 and month in (1, 2)
    )

    if second_lockdown:
        weights[physical] *= 0.35
        weights[online] *= 7.0

        weights[
            np.isin(
                categories,
                ["Supplement", "Merchandise"],
            )
        ] *= 1.45

    if year == 2021 and month >= 3:
        weights[categories == "Personal Training"] *= 1.05
        weights[categories == "Group Class"] *= 0.62
        weights[online] *= 2.0

    if year == 2022:
        weights[categories == "Personal Training"] *= 1.22
        weights[categories == "Group Class"] *= 0.82
        weights[online] *= 1.35

    if year >= 2023:
        weights[premium_growth] *= (
            1.55 + 0.10 * (year - 2023)
        )

        weights[
            categories == "Personal Training"
        ] *= 1.30 + 0.06 * (year - 2023)

        weights[online] *= 1.15

    weights = np.clip(weights, 0.0, None)

    return weights / weights.sum()


def choose_segment(
    rng: np.random.Generator,
    product_name: str,
) -> str:
    options = [
        "Student",
        "Young Professional",
        "Family",
        "Corporate",
        "Senior",
        "General",
    ]

    if product_name == "Student Membership":
        probabilities = [
            0.82,
            0.10,
            0.01,
            0.00,
            0.00,
            0.07,
        ]
    elif product_name == "Corporate Membership":
        probabilities = [
            0.01,
            0.05,
            0.04,
            0.84,
            0.01,
            0.05,
        ]
    elif product_name in {
        "Monthly Premium Membership",
        "Annual Premium Membership",
    }:
        probabilities = [
            0.05,
            0.43,
            0.24,
            0.09,
            0.05,
            0.14,
        ]
    elif product_name == "Yoga Class":
        probabilities = [
            0.12,
            0.28,
            0.18,
            0.04,
            0.22,
            0.16,
        ]
    else:
        probabilities = [
            0.17,
            0.31,
            0.16,
            0.08,
            0.10,
            0.18,
        ]

    return str(
        rng.choice(options, p=probabilities)
    )


def choose_branch(
    rng: np.random.Generator,
    date: pd.Timestamp,
) -> str:
    options = [
        "City Centre",
        "Riverside",
        "Northside",
    ]

    if date.year <= 2017:
        probabilities = [0.52, 0.29, 0.19]
    elif date.year <= 2020:
        probabilities = [0.49, 0.30, 0.21]
    elif date.year <= 2023:
        probabilities = [0.46, 0.30, 0.24]
    else:
        probabilities = [0.43, 0.31, 0.26]

    if date.year == 2022 and date.month == 9:
        probabilities = [0.47, 0.16, 0.37]

    return str(
        rng.choice(options, p=probabilities)
    )


def choose_sales_channel(
    rng: np.random.Generator,
    date: pd.Timestamp,
    category: str,
    product_name: str,
) -> str:
    options = [
        "Front Desk",
        "Website",
        "Mobile App",
        "Corporate Partner",
        "Telephone",
    ]

    if product_name == "Corporate Membership":
        probabilities = [
            0.05,
            0.08,
            0.02,
            0.80,
            0.05,
        ]
    elif category == "Online Fitness":
        probabilities = [
            0.01,
            0.53,
            0.43,
            0.00,
            0.03,
        ]
    elif date.year <= 2018:
        probabilities = [
            0.56,
            0.20,
            0.05,
            0.08,
            0.11,
        ]
    elif date.year == 2019:
        probabilities = [
            0.50,
            0.23,
            0.08,
            0.09,
            0.10,
        ]
    elif date.year == 2020 and date.month >= 3:
        probabilities = [
            0.08,
            0.48,
            0.28,
            0.10,
            0.06,
        ]
    elif date.year <= 2022:
        probabilities = [
            0.36,
            0.31,
            0.19,
            0.08,
            0.06,
        ]
    else:
        probabilities = [
            0.28,
            0.32,
            0.29,
            0.07,
            0.04,
        ]

    return str(
        rng.choice(options, p=probabilities)
    )


def choose_payment_method(
    rng: np.random.Generator,
    date: pd.Timestamp,
    product_name: str,
    channel: str,
    recurring_eligible: bool,
) -> str:
    methods = [
        "Card",
        "Direct Debit",
        "Bank Transfer",
        "Cash",
        "Digital Wallet",
    ]

    if product_name == "Corporate Membership":
        probabilities = np.array(
            [0.08, 0.29, 0.57, 0.01, 0.05]
        )
    elif recurring_eligible:
        if date.year <= 2019:
            probabilities = np.array(
                [0.22, 0.65, 0.05, 0.04, 0.04]
            )
        else:
            probabilities = np.array(
                [0.18, 0.68, 0.04, 0.01, 0.09]
            )
    else:
        progress = (date.year - 2015) / 10

        probabilities = np.array(
            [
                0.50 - 0.06 * progress,
                0.10 + 0.08 * progress,
                0.08,
                0.28 - 0.24 * progress,
                0.04 + 0.22 * progress,
            ]
        )

        if channel == "Mobile App":
            probabilities *= [
                0.80,
                0.55,
                0.20,
                0.02,
                2.50,
            ]
        elif channel == "Website":
            probabilities *= [
                1.15,
                0.75,
                0.45,
                0.02,
                1.70,
            ]
        elif channel == "Front Desk":
            probabilities *= [
                1.00,
                0.65,
                0.35,
                1.70,
                0.70,
            ]

        probabilities /= probabilities.sum()

    return str(
        rng.choice(methods, p=probabilities)
    )


def choose_campaign(
    rng: np.random.Generator,
    date: pd.Timestamp,
    segment: str,
) -> str:
    if date.month == 1:
        new_year_probability = (
            0.82
            if date.year in (2019, 2024, 2025)
            else 0.68
        )

        return str(
            rng.choice(
                [
                    "New Year Campaign",
                    "Referral Programme",
                    "No Campaign",
                ],
                p=[
                    new_year_probability,
                    0.10,
                    0.90 - new_year_probability,
                ],
            )
        )

    if date.month == 11:
        return str(
            rng.choice(
                [
                    "Black Friday",
                    "Referral Programme",
                    "No Campaign",
                ],
                p=[0.58, 0.10, 0.32],
            )
        )

    if date.month == 9 and segment == "Student":
        return str(
            rng.choice(
                [
                    "Student Promotion",
                    "Referral Programme",
                    "No Campaign",
                ],
                p=[0.70, 0.10, 0.20],
            )
        )

    if segment == "Corporate":
        return str(
            rng.choice(
                [
                    "Corporate Wellness",
                    "Referral Programme",
                    "No Campaign",
                ],
                p=[0.47, 0.12, 0.41],
            )
        )

    if date.month in (6, 7, 8):
        return str(
            rng.choice(
                [
                    "Summer Fitness Campaign",
                    "Referral Programme",
                    "No Campaign",
                ],
                p=[0.30, 0.13, 0.57],
            )
        )

    return str(
        rng.choice(
            [
                "Referral Programme",
                "No Campaign",
            ],
            p=[0.16, 0.84],
        )
    )


def choose_discount(
    rng: np.random.Generator,
    campaign: str,
) -> float:
    choices = {
        "No Campaign": (
            [0.00, 0.03, 0.05],
            [0.76, 0.14, 0.10],
        ),
        "New Year Campaign": (
            [0.10, 0.15, 0.20],
            [0.35, 0.50, 0.15],
        ),
        "Summer Fitness Campaign": (
            [0.08, 0.12, 0.15],
            [0.35, 0.45, 0.20],
        ),
        "Student Promotion": (
            [0.10, 0.15, 0.20],
            [0.25, 0.55, 0.20],
        ),
        "Corporate Wellness": (
            [0.05, 0.08, 0.10, 0.12],
            [0.20, 0.35, 0.35, 0.10],
        ),
        "Referral Programme": (
            [0.05, 0.08, 0.10],
            [0.45, 0.35, 0.20],
        ),
        "Black Friday": (
            [0.15, 0.20, 0.25, 0.30],
            [0.20, 0.40, 0.30, 0.10],
        ),
    }

    values, probabilities = choices[campaign]

    return float(
        rng.choice(values, p=probabilities)
    )


def choose_quantity(
    rng: np.random.Generator,
    category: str,
    product_name: str,
) -> int:
    if (
        category
        in {
            "Membership",
            "Joining Fee",
            "Online Fitness",
        }
        or product_name
        == "10-Session Personal Training Package"
    ):
        return 1

    if category == "Personal Training":
        return int(
            rng.choice(
                [1, 2, 3],
                p=[0.82, 0.14, 0.04],
            )
        )

    if category == "Group Class":
        return int(
            rng.choice(
                [1, 2, 3, 4],
                p=[0.68, 0.22, 0.08, 0.02],
            )
        )

    return int(
        rng.choice(
            [1, 2, 3],
            p=[0.72, 0.22, 0.06],
        )
    )


def allocated_overhead(
    date: pd.Timestamp,
    branch: str,
) -> float:
    if date.year == 2020:
        if date.month <= 2:
            overhead = 45.0
        elif date.month == 3:
            overhead = 70.0
        elif date.month in (4, 5, 6):
            overhead = 200.0
        elif date.month in (7, 8, 9, 10):
            overhead = 95.0
        else:
            overhead = 170.0

    elif date.year == 2021:
        if date.month in (1, 2):
            overhead = 78.0
        elif date.month in (3, 4):
            overhead = 32.0
        else:
            overhead = 10.0

    elif (
        date.year == 2022
        and date.month == 9
        and branch == "Riverside"
    ):
        overhead = 58.0

    else:
        overhead = (
            4.5 + 0.25 * (date.year - 2015)
        )

    branch_multiplier = {
        "City Centre": 1.08,
        "Riverside": 1.00,
        "Northside": 0.94,
    }[branch]

    return overhead * branch_multiplier


def generate_transactions(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    catalog: pd.DataFrame,
    customer_pools: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for number, value in enumerate(
        dates,
        start=1,
    ):
        date = pd.Timestamp(value)

        product_index = int(
            rng.choice(
                len(catalog),
                p=product_probabilities(
                    catalog,
                    date,
                ),
            )
        )

        product = catalog.iloc[product_index]

        category = str(
            product["product_category"]
        )
        product_name = str(
            product["product_name"]
        )

        segment = choose_segment(
            rng,
            product_name,
        )

        branch = choose_branch(
            rng,
            date,
        )

        channel = choose_sales_channel(
            rng,
            date,
            category,
            product_name,
        )

        payment_method = choose_payment_method(
            rng,
            date,
            product_name,
            channel,
            bool(product["recurring_eligible"]),
        )

        campaign = choose_campaign(
            rng,
            date,
            segment,
        )

        discount_pct = choose_discount(
            rng,
            campaign,
        )

        quantity = choose_quantity(
            rng,
            category,
            product_name,
        )

        branch_price_multiplier = {
            "City Centre": 1.03,
            "Riverside": 1.00,
            "Northside": 0.97,
        }[branch]

        unit_price = round(
            float(product["base_price"])
            * INFLATION[date.year]
            * branch_price_multiplier
            * rng.uniform(0.98, 1.02),
            2,
        )

        gross_revenue = round(
            quantity * unit_price,
            2,
        )

        discount_amount = round(
            gross_revenue * discount_pct,
            2,
        )

        net_revenue = round(
            gross_revenue - discount_amount,
            2,
        )

        variable_cost = (
            gross_revenue
            * float(product["cost_ratio"])
            * rng.uniform(0.96, 1.04)
        )

        estimated_cost = round(
            variable_cost
            + allocated_overhead(
                date,
                branch,
            ),
            2,
        )

        abnormal_period = (
            pd.Timestamp("2020-03-01")
            <= date
            <= pd.Timestamp("2021-04-30")
        ) or (
            date.year == 2022
            and date.month == 9
            and branch == "Riverside"
        )

        if not abnormal_period:
            estimated_cost = min(
                estimated_cost,
                round(net_revenue * 0.96, 2),
            )

        profit = round(
            net_revenue - estimated_cost,
            2,
        )

        recurring_probability = (
            0.94
            if payment_method == "Direct Debit"
            else 0.72
        )

        is_recurring = (
            bool(product["recurring_eligible"])
            and bool(
                rng.random()
                < recurring_probability
            )
        )

        records.append(
            {
                "transaction_id": (
                    f"TXN-{uuid4().hex.upper()}"
                ),
                "transaction_date": date,
                "customer_id": str(
                    rng.choice(
                        customer_pools[segment]
                    )
                ),
                "branch": branch,
                "customer_segment": segment,
                "product_category": category,
                "product_name": product_name,
                "membership_plan": str(
                    product["membership_plan"]
                ),
                "sales_channel": channel,
                "payment_method": payment_method,
                "campaign": campaign,
                "quantity": quantity,
                "unit_price_gbp": unit_price,
                "discount_pct": discount_pct,
                "gross_revenue_gbp": gross_revenue,
                "discount_amount_gbp": discount_amount,
                "net_revenue_gbp": net_revenue,
                "estimated_cost_gbp": estimated_cost,
                "profit_gbp": profit,
                "is_recurring_payment": is_recurring,
            }
        )

    dataframe = pd.DataFrame(records)

    dataframe.insert(
        2,
        "year",
        dataframe[
            "transaction_date"
        ].dt.year.astype(int),
    )

    dataframe.insert(
        3,
        "quarter",
        "Q"
        + dataframe[
            "transaction_date"
        ].dt.quarter.astype(str),
    )

    dataframe.insert(
        4,
        "month",
        dataframe[
            "transaction_date"
        ].dt.month.astype(int),
    )

    dataframe.insert(
        5,
        "month_name",
        dataframe[
            "transaction_date"
        ].dt.month_name(),
    )

    return dataframe


def inject_data_quality_issues(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()
    all_rows = result.index.to_numpy()

    campaign_missing = rng.choice(
        all_rows,
        size=50,
        replace=False,
    )

    payment_candidates = np.setdiff1d(
        all_rows,
        campaign_missing,
    )

    payment_missing = rng.choice(
        payment_candidates,
        size=30,
        replace=False,
    )

    result.loc[
        campaign_missing,
        "campaign",
    ] = pd.NA

    result.loc[
        payment_missing,
        "payment_method",
    ] = pd.NA

    complete_rows = result.index[
        result["campaign"].notna()
        & result["payment_method"].notna()
    ].to_numpy()

    duplicate_sources = rng.choice(
        complete_rows,
        size=DUPLICATE_ROWS,
        replace=False,
    )

    duplicates = result.loc[
        duplicate_sources
    ].copy()

    result = pd.concat(
        [result, duplicates],
        ignore_index=True,
    )

    return (
        result.sort_values(
            [
                "transaction_date",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = [
        "transaction_id",
        "transaction_date",
        "year",
        "quarter",
        "month",
        "month_name",
        "customer_id",
        "branch",
        "customer_segment",
        "product_category",
        "product_name",
        "membership_plan",
        "sales_channel",
        "payment_method",
        "campaign",
        "quantity",
        "unit_price_gbp",
        "discount_pct",
        "gross_revenue_gbp",
        "discount_amount_gbp",
        "net_revenue_gbp",
        "estimated_cost_gbp",
        "profit_gbp",
        "is_recurring_payment",
    ]

    if len(dataframe) != ROW_COUNT:
        raise ValueError(
            f"Expected {ROW_COUNT} rows, "
            f"found {len(dataframe)}."
        )

    if list(dataframe.columns) != required_columns:
        raise ValueError(
            "Dataset columns do not match "
            "the required schema."
        )

    if (
        dataframe["transaction_date"].min()
        != pd.Timestamp("2015-01-01")
    ):
        raise ValueError(
            "The dataset must begin on "
            "2015-01-01."
        )

    if (
        dataframe["transaction_date"].max()
        != pd.Timestamp("2025-12-31")
    ):
        raise ValueError(
            "The dataset must end on "
            "2025-12-31."
        )

    if (
        dataframe["transaction_id"].isna().any()
        or (
            dataframe["transaction_id"]
            .astype(str)
            .str.len()
            == 0
        ).any()
    ):
        raise ValueError(
            "Transaction IDs must be present."
        )

    critical_columns = [
        "transaction_date",
        "quantity",
        "unit_price_gbp",
        "discount_pct",
        "gross_revenue_gbp",
        "discount_amount_gbp",
        "net_revenue_gbp",
        "estimated_cost_gbp",
        "profit_gbp",
    ]

    if (
        dataframe[critical_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Critical fields contain "
            "missing values."
        )

    if (
        dataframe["quantity"] <= 0
    ).any() or (
        dataframe["unit_price_gbp"] <= 0
    ).any():
        raise ValueError(
            "Quantities and prices "
            "must be positive."
        )

    if not dataframe[
        "discount_pct"
    ].between(
        0.0,
        0.50,
    ).all():
        raise ValueError(
            "Discount percentages must be "
            "between 0 and 0.50."
        )

    expected_gross = (
        dataframe["quantity"]
        * dataframe["unit_price_gbp"]
    ).round(2)

    expected_discount = (
        dataframe["gross_revenue_gbp"]
        * dataframe["discount_pct"]
    ).round(2)

    expected_net = (
        dataframe["gross_revenue_gbp"]
        - dataframe["discount_amount_gbp"]
    ).round(2)

    expected_profit = (
        dataframe["net_revenue_gbp"]
        - dataframe["estimated_cost_gbp"]
    ).round(2)

    calculation_checks = [
        (
            dataframe["gross_revenue_gbp"],
            expected_gross,
            "Gross revenue",
        ),
        (
            dataframe["discount_amount_gbp"],
            expected_discount,
            "Discount",
        ),
        (
            dataframe["net_revenue_gbp"],
            expected_net,
            "Net revenue",
        ),
        (
            dataframe["profit_gbp"],
            expected_profit,
            "Profit",
        ),
    ]

    for actual, expected, label in calculation_checks:
        if not np.allclose(
            actual,
            expected,
            atol=0.01,
        ):
            raise ValueError(
                f"{label} calculations "
                "are inconsistent."
            )

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    if not 10 <= duplicate_count <= 20:
        raise ValueError(
            "Expected 10-20 exact duplicate "
            f"rows, found {duplicate_count}."
        )

    missing_count = int(
        dataframe[
            ["campaign", "payment_method"]
        ]
        .isna()
        .sum()
        .sum()
    )

    if not 50 <= missing_count <= 100:
        raise ValueError(
            "Expected controlled missing "
            f"values, found {missing_count}."
        )

    normal_period = ~(
        dataframe[
            "transaction_date"
        ].between(
            "2020-03-01",
            "2021-04-30",
        )
        | (
            (dataframe["year"] == 2022)
            & (dataframe["month"] == 9)
            & (
                dataframe["branch"]
                == "Riverside"
            )
        )
    )

    if (
        dataframe.loc[
            normal_period,
            "estimated_cost_gbp",
        ]
        > dataframe.loc[
            normal_period,
            "net_revenue_gbp",
        ]
    ).any():
        raise ValueError(
            "Normal transactions must not "
            "have costs above net revenue."
        )



def save_dataset(
    dataframe: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> None:
    output = dataframe.copy()

    output["transaction_date"] = (
        output["transaction_date"]
        .dt.strftime("%Y-%m-%d")
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    # No fixed seed is supplied. NumPy obtains fresh entropy
    # from the operating system, so every run is different.
    rng = np.random.default_rng()

    catalog, customer_pools = (
        create_reference_data()
    )

    dates = generate_transaction_dates(rng)

    dataset = generate_transactions(
        rng,
        dates,
        catalog,
        customer_pools,
    )

    dataset = inject_data_quality_issues(
        rng,
        dataset,
    )

    validate_dataset(dataset)
    save_dataset(dataset)

    yearly_summary = (
        dataset.groupby(
            "year",
            as_index=False,
        )
        .agg(
            transaction_count=(
                "transaction_id",
                "size",
            ),
            net_revenue_gbp=(
                "net_revenue_gbp",
                "sum",
            ),
            estimated_cost_gbp=(
                "estimated_cost_gbp",
                "sum",
            ),
            profit_gbp=(
                "profit_gbp",
                "sum",
            ),
        )
        .round(2)
    )

    minimum_date = (
        dataset["transaction_date"]
        .min()
        .date()
    )

    maximum_date = (
        dataset["transaction_date"]
        .max()
        .date()
    )

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Dataset shape: {dataset.shape}")
    print(
        f"Date range: {minimum_date} "
        f"to {maximum_date}"
    )
    print(
        "Exact duplicate rows: "
        f"{int(dataset.duplicated().sum())}"
    )
    print(
        "Missing values: "
        f"{int(dataset.isna().sum().sum())}"
    )
    print("\nYearly summary:")
    print(
        yearly_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()