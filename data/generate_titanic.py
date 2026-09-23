"""
Generates a realistic Titanic benchmark dataset and places it in data/samples/titanic.csv.
"""

import os
import numpy as np
import pandas as pd

def generate_titanic_benchmark(n=891, random_state=42) -> pd.DataFrame:
    np.random.seed(random_state)
    p_ids = list(range(1, n + 1))
    pclasses = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sexes = np.random.choice(["male", "female"], size=n, p=[0.65, 0.35])
    
    # Age with ~20% missing values
    ages = np.where(pclasses == 1, np.random.normal(38, 14, n), np.where(pclasses == 2, np.random.normal(29, 13, n), np.random.normal(25, 12, n)))
    ages = np.clip(np.round(ages, 1), 0.5, 80.0)
    mask_age = np.random.rand(n) < 0.20
    ages = np.where(mask_age, np.nan, ages)

    sibsp = np.random.choice([0, 1, 2, 3], size=n, p=[0.68, 0.23, 0.06, 0.03])
    parch = np.random.choice([0, 1, 2], size=n, p=[0.76, 0.16, 0.08])

    fares = np.where(
        pclasses == 1, np.random.exponential(65, n) + 25,
        np.where(pclasses == 2, np.random.exponential(18, n) + 10, np.random.exponential(8, n) + 7)
    )
    fares = np.round(np.clip(fares, 4.0, 512.0), 2)

    embarked = np.random.choice(["S", "C", "Q"], size=n, p=[0.72, 0.20, 0.08])
    # 2 missing values in Embarked like real Titanic
    embarked[0] = np.nan
    embarked[1] = np.nan

    # Survival logic based on historic facts: women and 1st class survived more
    logit = (
        -1.2 
        + 2.4 * (sexes == "female") 
        + 1.3 * (pclasses == 1) 
        + 0.5 * (pclasses == 2) 
        - 0.03 * np.nan_to_num(ages, nan=28.0) 
        + 0.005 * fares
    )
    prob_surv = 1 / (1 + np.exp(-logit))
    survived = (np.random.rand(n) < prob_surv).astype(int)

    names = [f"Passenger, Person {i}" for i in p_ids]
    tickets = [f"TICKET_{1000 + i}" for i in p_ids]

    df = pd.DataFrame({
        "PassengerId": p_ids,
        "Survived": survived,
        "Pclass": pclasses,
        "Name": names,
        "Sex": sexes,
        "Age": ages,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": tickets,
        "Fare": fares,
        "Embarked": embarked
    })
    return df

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "samples", "titanic.csv")
    df = generate_titanic_benchmark()
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df)} rows. Survival rate: {df['Survived'].mean():.2%}")
