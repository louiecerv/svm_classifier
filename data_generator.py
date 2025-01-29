import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

def generate_simulated_data():
    """Generates a simulated business classification dataset and saves it to a CSV file."""
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
    df["Target"] = y
    
    # Save to CSV
    df.to_csv("business_data.csv", index=False)

if __name__ == "__main__":
    generate_simulated_data()
    print("Simulated business dataset saved as 'business_data.csv'.")
