# Section C - Mini Capstone Project
# Food Delivery Analytics Console - Full EDA Report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(30)

# Generate the dataset at program startup
restaurants = ["Spice Hub", "Urban Bites", "Tasty Corner", "Food Street", "Royal Kitchen"]
cities = ["Vadodara", "Ahmedabad", "Surat"]
cuisines = ["Indian", "Chinese", "Fast Food"]

df = pd.DataFrame({
    "restaurant_name": np.random.choice(restaurants, 250),
    "city": np.random.choice(cities, 250),
    "order_value": np.random.uniform(150, 1000, 250),
    "delivery_time_mins": np.random.normal(30, 7, 250),
    "rating": np.random.uniform(1.0, 5.0, 250),
    "cuisine_type": np.random.choice(cuisines, 250)
})

# Create a few missing numeric values for the required null-handling step
for column in ["order_value", "delivery_time_mins", "rating"]:
    missing_rows = np.random.choice(df.index, size=5, replace=False)
    df.loc[missing_rows, column] = np.nan

numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


def summary_statistics():
    print("\n--- Summary Statistics ---")
    print(f"Mean order value: Rs {np.mean(df['order_value']):.2f}")
    print(f"Mean delivery time: {np.mean(df['delivery_time_mins']):.2f} minutes")
    print(f"Mean rating: {np.mean(df['rating']):.2f}")
    print(f"Delivery time standard deviation: {np.std(df['delivery_time_mins']):.2f}")


def distribution_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["order_value"], bins=15)
    axes[0].set_title("Order Value Distribution")
    axes[0].set_xlabel("Order Value (Rs)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["delivery_time_mins"], bins=15)
    axes[1].set_title("Delivery Time Distribution")
    axes[1].set_xlabel("Delivery Time (minutes)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("distribution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nDistribution chart saved as distribution_analysis.png")


def correlation_heatmap():
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Food Delivery Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap_capstone.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nCorrelation heatmap saved as correlation_heatmap_capstone.png")


def restaurant_performance():
    result = (
        df.groupby("restaurant_name")
          .agg(
              mean_order_value=("order_value", "mean"),
              mean_delivery_time=("delivery_time_mins", "mean"),
              mean_rating=("rating", "mean")
          )
          .sort_values("mean_rating", ascending=False)
    )

    print("\n--- Restaurant Performance Report ---")
    print(result.to_string(float_format=lambda x: f"{x:.2f}"))


def final_report():
    performance = (
        df.groupby("restaurant_name")["rating"]
          .mean()
          .sort_values(ascending=False)
          .head(3)
    )

    corr = df.select_dtypes(include=np.number).corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    strongest_pair = corr.stack().idxmax()
    strongest_value = corr.stack().max()

    mean_delivery = np.mean(df["delivery_time_mins"])
    std_delivery = np.std(df["delivery_time_mins"])

    print("\n" + "=" * 50)
    print("FINAL SUMMARY REPORT")
    print("=" * 50)

    print("\nTop 3 restaurants by mean rating:")
    for i, (restaurant, rating) in enumerate(performance.items(), 1):
        print(f"{i}. {restaurant}: {rating:.2f}")

    print(f"\nHighest absolute Pearson correlation:")
    print(f"{strongest_pair[0]} vs {strongest_pair[1]}: {strongest_value:.2f}")

    print(f"\nOverall mean delivery time: {mean_delivery:.2f} minutes")
    print(f"Overall delivery time standard deviation: {std_delivery:.2f} minutes")


while True:
    print("\n===== FOOD DELIVERY ANALYTICS CONSOLE =====")
    print("1. Summary Statistics")
    print("2. Distribution Analysis")
    print("3. Correlation Heatmap")
    print("4. Restaurant Performance Report")
    print("5. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        summary_statistics()
    elif choice == "2":
        distribution_analysis()
    elif choice == "3":
        correlation_heatmap()
    elif choice == "4":
        restaurant_performance()
    elif choice == "5":
        final_report()
        print("\nProgram ended.")
        break
    else:
        print("Please enter a valid choice.")
