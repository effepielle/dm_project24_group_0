"""UTILITY FUNCTIONS"""

from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import bisect
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, get_scorer
import os


# ----- DATA PREPROCESSING FUNCTIONS -----

#correlation function 
def correlations(dataset: pd.DataFrame, corr_idx) -> pd.DataFrame:
    correlations_dictionary = {
        correlation_type: dataset.corr(numeric_only=True, method=correlation_type)
        for correlation_type in corr_idx
    }
    for i, k in enumerate(correlations_dictionary.keys()):
        correlations_dictionary[k].loc[:, "correlation_type"] = k
    correlations_matrix = pd.concat(correlations_dictionary.values())

    return correlations_matrix


def plot_correlations(correlations: pd.DataFrame, fig_size: tuple = (18,6)) -> None:
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
def __transform_single_features(dataset: pd.DataFrame, transformation: str) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
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
def center_and_scale(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Shifts data to the origin: removes mean and scales by standard deviation all numeric features. Returns a copy of the dataset."""
    return __transform_single_features(dataset, "standard")


# outlier counts function
def count_outliers(column: pd.DataFrame) -> pd.Series:
    """Counts the number of outliers in each numeric feature using the IQR method. Returns the counts."""
    # Printing number of outliers basing on the IQR method
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    outliers = column[(column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))]

    print(f"Number of outliers: {len(outliers)} over {len(column) - column.isnull().sum()} values using the IQR method")
    return None


# ----- FEATURE ENGINEERING FUNCTIONS ----- 

# BMI index function
def compute_bmi(weight, height):
    if pd.notnull(weight) and pd.notnull(height):
        return round(weight / ((height / 100) ** 2), 2)
    else:
        return np.nan
    
    
# Difficulty index function
def compute_difficulty_index(length, climb_total, profile):
    # total meters climbed is more impacting, then length of the race and
    # finally profile value as competitions at altitude are more complex 
    difficulty_index = length*0.3 + climb_total*0.6 + profile*0.1
    return round(difficulty_index, 2)


def compute_cyclist_performance(races_df: pd.DataFrame, WEIGHTS) -> Dict:
    # Initialize the dictionary
    cyclist_performance = {}

    # iteration in all races to compute the cyclist level
    for _, row in races_df.iterrows():
        
        cyclist = row['cyclist']
        position = row['position']
        points = row['points']
        num_cyclists = row['num_cyclists']


        # Initialize the nested dictionary if the cyclist is not already in the dictionary
        if cyclist not in cyclist_performance:
            cyclist_performance[cyclist] = {'total_races': 0}
            # a dictionary is created for each cyclist with the keys '1_points', '2_points', '3_points', ... and 'total'

        cyclist_performance[cyclist]['total_races'] += 1
        
        # Add the date and points as a tuple to the appropriate list based on the position
        position_key = f'{position + 1}_points'
        if position_key in cyclist_performance[cyclist]:
            cyclist_performance[cyclist][position_key].append((points, num_cyclists))
        else:
            cyclist_performance[cyclist][position_key] = [(points, num_cyclists)]
    
    return cyclist_performance


def compute_cyclist_cumulative_performance(races_df: pd.DataFrame, WEIGHTS):
    # Initialize the dictionary
    cyclist_performance = {}

    # iteration in all races to compute the cyclist level
    for _, row in races_df.iterrows():
        
        cyclist = row['cyclist']
        position = row['position']
        points = row['points']
        num_cyclists = row['cyclist_number']

        # Initialize the nested dictionary if the cyclist is not already in the dictionary
        if cyclist not in cyclist_performance:
            cyclist_performance[cyclist] = {'total_races': 0}
            # a dictionary is created for each cyclist with the keys '1_points', '2_points', '3_points', ... and 'total'

        cyclist_performance[cyclist]['total_races'] += 1

        # Add the date and points as a tuple to the appropriate list based on the position
        position_key = f'{position + 1}_points'
        if position_key in cyclist_performance[cyclist]:
            cyclist_performance[cyclist][position_key].append((points, num_cyclists))
        else:
            cyclist_performance[cyclist][position_key] = [(points, num_cyclists)]
        
        normalized_level = 0.0

        # Sum of placements weighted by their position and race score before the date
        placement_sum = 0
        for position, weight in WEIGHTS.items():
            if position in cyclist_performance[cyclist]:
                for points, num_cyclists in cyclist_performance[cyclist][position]:
                    placement_sum += weight * points
            
            normalized_level = placement_sum / cyclist_performance[cyclist]['total_races']
        
        # Update the 'cyclist_level' column for the current row
        races_df.at[row.name, 'cyclist_level'] = normalized_level


def get_season(date):
    month = date.month
    day = date.day

    if (month == 12 and day >= 21) or (month <= 2) or (month == 3 and day < 20):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month <= 5) or (month == 6 and day < 21):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month <= 8) or (month == 9 and day < 22):
        return 'Summer'
    else:
        return 'Autumn'
    
        
def compute_average_on_column(df, column, cluster_labels):
    # Filtra i ciclisti per cluster 0 e cluster 1
    cluster_0 = df[df[cluster_labels] == 0]
    cluster_1 = df[df[cluster_labels] == 1]

    # Calcola la media dell'esperienza dei ciclisti per ciascun cluster
    mean_cluster_0 = cluster_0[column].mean()
    mean_cluster_1 = cluster_1[column].mean()

    print(f"Average Value in column '{column}' in cluster 0: {mean_cluster_0}")
    print(f"Average Value in column '{column}' in cluster 1: {mean_cluster_1}")

def plot_parallel_cluster_distribution(best_df, rest_df, outlier_df, column):
    
    # Remove underscores from the column name for plotting
    column_str = column.replace('_', ' ')
    column_str = column_str.capitalize()
    column_str = " "+column_str+" "
    # Plot the distribution of stages won for all clusters
    plt.figure(figsize=(18, 6))

    # Plot for best cyclists
    plt.subplot(1, 3, 1)
    sns.histplot(best_df[column], bins=100, stat='percent')
    plt.title('Distribution of ' + column_str + ' for Best Cyclists')
    plt.xlabel(column_str)
    plt.ylabel('Percentage of Cyclists')

    # Plot for intermediate cyclists
    plt.subplot(1, 3, 2)
    sns.histplot(rest_df[column], bins=100, stat='percent')
    plt.title('Distribution of ' + column_str + ' for Rest of Cyclists')
    plt.xlabel(column_str)
    plt.ylabel('Percentage of Cyclists')

    # Plot for outlier cyclists
    plt.subplot(1, 3, 3)
    sns.histplot(outlier_df[column], bins=100, stat='percent')
    plt.title('Distribution of ' + column_str + ' for Outliers')
    plt.xlabel(column_str)
    plt.ylabel('Percentage of Cyclists')

    plt.tight_layout()
    plt.show()  

# Defining sensitivity and specificity as metrics of sklearn

def specificity_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn / (tn + fp)    

def sensitivity_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tp / (tp + fn)

# Funzioni personalizzate con zero_division
def precision_with_zero_division(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def recall_with_zero_division(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

# Scoring Dictionary
scoring = {
    'sensitivity': make_scorer(sensitivity_score),
    'specificity': make_scorer(specificity_score),
    'accuracy': get_scorer("accuracy"),
    'precision': make_scorer(precision_with_zero_division),
    'recall': make_scorer(recall_with_zero_division),
    'roc_auc': get_scorer("roc_auc"),
    'f1': get_scorer("f1"),
}

def save_results(model, resampling, mean_test_scores, std_test_scores, accuracy, recall, precision, sensitivity_score, specificity_score, f1, roc_auc, report):    
    # Create a folder with the model name if it does not exist
    model_folder = f"model_results/{model}"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save the results to a file
    results_file = os.path.join(model_folder, f"{model}_{resampling}.txt")
    with open(results_file, "w") as file:
        file.write(f"MODEL: {model} ")
        file.write(f"- Resampling: {resampling}\n\n")
        file.write(f"Validation Results:\n\n")

        for metric in scoring.keys():
            file.write(f"{metric.capitalize()} - Mean: {mean_test_scores[metric]:.4f}, Std: {std_test_scores[metric]:.4f}\n")
   
        file.write(f"\nTest Results:\n\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Sensitivity: {sensitivity_score}\n")
        file.write(f"Specificity: {specificity_score}\n")
        file.write(f"F1 Score: {f1}\n")
        file.write(f"ROC AUC: {roc_auc}\n")
        file.write(f"\nClassification Report:\n{report}\n")