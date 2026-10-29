
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape)

# Parameters
alpha = 0.05

# Decision function
def decide(p):
    return "Reject H0" if p < alpha else "Fail to reject H0"

# 1. Welch's t-test: Age by Sex
g1 = df.loc[df["Sex"] == "male", "Age"].dropna()
g2 = df.loc[df["Sex"] == "female", "Age"].dropna()

print(g1.mean(), g2.mean())

sns.boxplot(data=df, x="Sex", y="Age")
plt.title("Age by Sex")
plt.show()

t_stat, t_p = ttest_ind(g1, g2, equal_var=False)
print("T-test:", t_stat, t_p, decide(t_p))

# 2. Chi-square: Sex vs Survived
chi_df = df[["Sex", "Survived"]].dropna()
observed = pd.crosstab(chi_df["Sex"], chi_df["Survived"])

sns.heatmap(observed, annot=True, fmt="d")
plt.title("Sex vs Survived")
plt.show()

chi2, chi_p, dof, expected = chi2_contingency(observed)
print("Chi-square:", chi2, dof, chi_p, decide(chi_p))
print(pd.DataFrame(expected, index=observed.index,
                   columns=observed.columns).round(2))

# 3. One-way ANOVA: Fare by Pclass
anova_df = df[["Fare", "Pclass"]].dropna()
samples = [
    anova_df.loc[anova_df["Pclass"] == g, "Fare"]
    for g in sorted(anova_df["Pclass"].unique())
]

print(anova_df.groupby("Pclass")["Fare"].agg(["count", "mean"]))

sns.boxplot(data=anova_df, x="Pclass", y="Fare")
plt.title("Fare by Passenger Class")
plt.show()

f_stat, anova_p = f_oneway(*samples)
print("ANOVA:", f_stat, anova_p, decide(anova_p))

# Tukey HSD
tukey = pairwise_tukeyhsd(
    anova_df["Fare"], anova_df["Pclass"], alpha=alpha
)
print(tukey)

# Final summary
results = pd.DataFrame({
    "Test": ["Welch's t-test", "Chi-square test", "One-way ANOVA"],
    "Statistic": [t_stat, chi2, f_stat],
    "p-value": [t_p, chi_p, anova_p]
})
results["Decision"] = results["p-value"].apply(decide)
print(results)
results.to_csv("hypothesis_testing_results.csv", index=False)
