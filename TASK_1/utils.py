"""UTILITY FUNCTIONS"""
from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns


# DATA PREPROCESSING FUNCTIONS 

#correlation function 
def correlations(dataset: pandas.DataFrame) -> pandas.DataFrame:
    correlations_dictionary = {
        correlation_type: dataset.corr(numeric_only=True, method=correlation_type)
        for correlation_type in ("kendall", "pearson", "spearman")
    }
    for i, k in enumerate(correlations_dictionary.keys()):
        correlations_dictionary[k].loc[:, "correlation_type"] = k
    correlations_matrix = pandas.concat(correlations_dictionary.values())

    return correlations_matrix

def plot_correlations(correlations: pandas.DataFrame, fig_size: tuple = (18,6)) -> None:
    # Plot heatmaps for each correlation type
    correlation_types = correlations['correlation_type'].unique()

    plt.figure(figsize=fig_size)

    for i, corr_type in enumerate(correlation_types):
        plt.subplot(1, len(correlation_types), i + 1)
        corr_matrix = correlations[correlations['correlation_type'] == corr_type].drop(columns='correlation_type').astype(float)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'{corr_type.capitalize()} Correlation Matrix')

    plt.tight_layout()
    plt.show()

# single feature transformation function
def __transform_single_features(dataset: pandas.DataFrame, transformation: str) -> Tuple[
    pandas.DataFrame, Dict[str, Any]]:
    match transformation:
        case "standard":
            transformed_dataset = dataset.copy().select_dtypes(exclude=["object", "category", "bool", "datetime64"])
            transformations = dict()

            for feature in transformed_dataset.columns:
                transformations[feature] = StandardScaler()
                transformed_feature = transformations[feature].fit_transform(transformed_dataset[[feature]]).squeeze()
                transformed_dataset = transformed_dataset.astype({feature: transformed_feature.dtype})
                transformed_dataset.loc[:, feature] = transformed_feature
        case _:
            raise ValueError(f"Unknown transformation: {transformation}")

    return transformed_dataset, transformations

# center and scale function
def center_and_scale(dataset: pandas.DataFrame) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
    """Shifts data to the origin: removes mean and scales by standard deviation all numeric features. Returns a copy of the dataset."""
    return __transform_single_features(dataset, "standard")

# outlier counts function
def count_outliers(column: pandas.DataFrame) -> pandas.Series:
    """Counts the number of outliers in each numeric feature using the IQR method. Returns the counts."""
    # Printing number of outliers basing on the IQR method
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    outliers = column[(column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))]

    print(f"Number of outliers: {len(outliers)} over {len(column) - column.isnull().sum()} values")
    return None
