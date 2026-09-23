"""
Generates realistic benchmark tabular datasets for AutoDataScientist:
1. Customer Churn (Binary classification, mixed types, missing values, ID column)
2. Credit Risk (Imbalanced classification, numerical & categorical)
3. Housing Price (Regression, continuous & discrete)
"""

import os
import numpy as np
import pandas as pd

def generate_customer_churn(n_samples=1200, random_state=42) -> pd.DataFrame:
    np.random.seed(random_state)
    customer_ids = [f"CUST_{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 72, size=n_samples)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    monthly_charges = np.where(
        internet_service == "Fiber optic",
        np.random.normal(85, 12, n_samples),
        np.where(internet_service == "DSL", np.random.normal(55, 10, n_samples), np.random.normal(25, 5, n_samples))
    ).round(2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

    total_charges = (tenure_months * monthly_charges + np.random.normal(0, 15, n_samples)).round(2)
    total_charges = np.clip(total_charges, 18.0, None)
    
    # Introduce ~3% missing values in total_charges
    mask = np.random.rand(n_samples) < 0.03
    total_charges = np.where(mask, np.nan, total_charges)

    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])

    # Realistic churn probability calculation
    logit = (
        -1.2
        - 0.04 * tenure_months
        + 0.02 * monthly_charges
        + 0.45 * support_tickets
        + 0.6 * (contract_type == "Month-to-month")
        - 0.8 * (contract_type == "Two year")
        + 0.5 * (payment_method == "Electronic check")
        + 0.3 * senior_citizen
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "paperless_billing": paperless_billing,
        "senior_citizen": senior_citizen,
        "churn": churn
    })
    return df

def generate_housing_prices(n_samples=1000, random_state=42) -> pd.DataFrame:
    np.random.seed(random_state)
    sqft_living = np.random.gamma(shape=5.0, scale=350, size=n_samples).astype(int)
    sqft_living = np.clip(sqft_living, 500, 6000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = np.round(np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n_samples), 1)
    location_grade = np.random.choice(["Tier1", "Tier2", "Tier3", "Rural"], size=n_samples, p=[0.25, 0.35, 0.3, 0.1])
    year_built = np.random.randint(1950, 2024, size=n_samples)
    has_garage = np.random.choice([1, 0], size=n_samples, p=[0.75, 0.25])
    distance_to_center_km = np.random.exponential(scale=12.0, size=n_samples).round(1)

    grade_mult = {"Tier1": 1.4, "Tier2": 1.1, "Tier3": 0.9, "Rural": 0.7}
    multiplier = np.array([grade_mult[g] for g in location_grade])

    price = (
        100000 
        + sqft_living * 180 
        + bedrooms * 15000 
        + bathrooms * 22000 
        + (year_built - 1950) * 800 
        + has_garage * 25000 
        - distance_to_center_km * 2500
    ) * multiplier + np.random.normal(0, 35000, size=n_samples)
    price = np.round(np.clip(price, 80000, 1800000), -2)

    df = pd.DataFrame({
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location_grade": location_grade,
        "year_built": year_built,
        "has_garage": has_garage,
        "distance_to_center_km": distance_to_center_km,
        "price": price
    })
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(out_dir, exist_ok=True)
    
    churn_df = generate_customer_churn()
    churn_path = os.path.join(out_dir, "customer_churn.csv")
    churn_df.to_csv(churn_path, index=False)
    print(f"Generated {churn_path} with {len(churn_df)} rows. Churn rate: {churn_df['churn'].mean():.2%}")

    house_df = generate_housing_prices()
    house_path = os.path.join(out_dir, "housing_prices.csv")
    house_df.to_csv(house_path, index=False)
    print(f"Generated {house_path} with {len(house_df)} rows. Price mean: ${house_df['price'].mean():,.0f}")
