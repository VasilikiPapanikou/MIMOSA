import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportions_ztest
from contextlib import redirect_stdout
import json
from collections import defaultdict
import os 
from scipy.stats import wasserstein_distance

class FairnessMetrics:
    def __init__(self, X_test, y_test, y_pred, sensitive_attr, attr_values, attr_names, output_dir, colors=None):
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.y_pred = y_pred
        self.sensitive_attr = sensitive_attr
        self.attr_values = attr_values  # e.g., [1, 0]
        self.attr_names = attr_names    # e.g., ['Male', 'Female']
        self.output_dir = output_dir
        self.colors = colors or {}
        self.metrics = {}

    def compute_confusion_matrix_values(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp, fn, fp, tn

    def compute_metrics(self):
        for val, name in zip(self.attr_values, self.attr_names):
            idx = self.X_test[self.sensitive_attr] == val
            y_true = self.y_test[idx]
            y_pred = self.y_pred[idx]
            tp, fn, fp, tn = self.compute_confusion_matrix_values(y_true, y_pred)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            pr = np.mean(y_pred == 1)

            self.metrics[name] = {
                "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "TPR": tpr, "FPR": fpr, "PR": pr,
                "Count": sum(y_pred == 1), "Nobs": len(y_pred)
            }
    
    def perform_z_tests(self):
        names = self.attr_names
        values = self.metrics

        pr_count = [values[n]["Count"] for n in names]
        pr_nobs = [values[n]["Nobs"] for n in names]
        stat_pr, p_pr = proportions_ztest(pr_count, pr_nobs)

        tpr_count = [values[n]["TP"] for n in names]
        tpr_nobs = [values[n]["TP"] + values[n]["FN"] for n in names]
        stat_tpr, p_tpr = proportions_ztest(tpr_count, tpr_nobs)

        fpr_count = [values[n]["FP"] for n in names]
        fpr_nobs = [values[n]["FP"] + values[n]["TN"] for n in names]
        stat_fpr, p_fpr = proportions_ztest(fpr_count, fpr_nobs)

        return {
            "PR": (stat_pr, p_pr),
            "TPR": (stat_tpr, p_tpr),
            "FPR": (stat_fpr, p_fpr)
        }

    def save_results(self, file_name):
        try:
            group_1 = self.attr_names[0]
            group_2 = self.attr_names[1]

            # Differences
            diffs = {
                "PR": self.metrics[group_1]["PR"] - self.metrics[group_2]["PR"],
                "TPR": self.metrics[group_1]["TPR"] - self.metrics[group_2]["TPR"],
                "FPR": self.metrics[group_1]["FPR"] - self.metrics[group_2]["FPR"],
            }

        except KeyError as e:
            raise KeyError(f"Missing key in self.metrics: {e}. Check that attr_names and sensitive_attr values match.") from e

        ztest_results = self.perform_z_tests()
        path = f"{self.output_dir}/fairness_metrics_{file_name}.txt"

        with open(path, "w") as f:
            with redirect_stdout(f):
                print("=== Fairness Metrics by Group ===")
                for group in self.attr_names:
                    print(f"\nGroup: {group}")
                    print(f"  PR:  {self.metrics[group]['PR']:.6f}")
                    print(f"  TPR: {self.metrics[group]['TPR']:.6f}")
                    print(f"  FPR: {self.metrics[group]['FPR']:.6f}")

                print("\n=== Differences (Group1 - Group2) ===")
                for metric, diff in diffs.items():
                    print(f"{metric} Difference ({group_1} - {group_2}): {diff:.6f}")
                    stat, p = ztest_results[metric]
                    print(f"Z-statistic for {metric}: {stat:.4f}, p-value: {p:.4f}")

    def plot_metrics(self, file_name):
        labels = ["PR", "TPR", "FPR"]
        group_values = {
            name: [self.metrics[name][m] for m in labels] for name in self.attr_names
        }

        bar_width = 0.35
        index = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (group, values) in enumerate(group_values.items()):
            color = self.colors.get(group, None)
            ax.bar(index + i * bar_width, values, bar_width, label=group, color=color)

        ax.set_xlabel("Metrics", fontsize=25)
        ax.set_ylabel("Values", fontsize=25)
        ax.set_xticks(index + bar_width * (len(self.attr_names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fairness_metrics_{file_name}.pdf", format="pdf")
        plt.show()


    ########################################Fairness metrics based on explanations########################################
    def load_explanations(self, save_file_path, save_file_name):
        """
        Loads explanations from a JSON file.

        Args:
            save_file_path: The directory path where the explanations file is stored.
            save_file_name: The name of the file that contains the saved explanations.
        """

        save_file = f"{save_file_path}{save_file_name}"
        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                explanations = json.load(f)
            print(f"Explanations loaded from {save_file}")
            return explanations
        return None

    def compute_attribution_distribution_difference(self, explanations, X_test, sensitive_attr, attr_names, value_to_name, distance_func=wasserstein_distance):
        """
        Computes the distributional difference in feature attributions between two groups 
        using a distance metric like Wasserstein distance.

        Args:
            explanations: List of explanations (e.g., SHAP or LIME) for each instance. 
                        Each explanation is a list of (feature_name, value) pairs.
            X_test: The test dataset containing feature values for each instance.
            sensitive_attr: The name of the sensitive attribute column (e.g., "Sex").
            attr_names: The two group names to compare (e.g., ["Male", "Female"]).
            value_to_name: Dict mapping sensitive attribute values to group names.
            distance_func: A function to compute distance between two lists of numbers. 
                        Default: Wasserstein distance (Earth Mover's Distance).

        """

        
        def extract_feature_name(feature_text):
            feature_text_lower = feature_text.lower()
            matches = [f for f in  X_test.columns if f.lower() in feature_text_lower]
            return max(matches, key=len) if matches else feature_text

        group_attributions = {name: defaultdict(list) for name in attr_names}

        for i, exp in enumerate(explanations):
            group_value = X_test.iloc[i][sensitive_attr]
            group_name = value_to_name.get(group_value)
            if group_name not in attr_names:
                continue
            for feature, value in exp:
                feature_group = extract_feature_name(feature)
                group_attributions[group_name][feature_group].append(value)

        all_features = set(group_attributions[attr_names[0]].keys()) | set(group_attributions[attr_names[1]].keys())
        results = []

        for feature in sorted(all_features):
            group1_vals = group_attributions[attr_names[0]].get(feature, [])
            group2_vals = group_attributions[attr_names[1]].get(feature, [])

            if group1_vals and group2_vals:
                dist = distance_func(group1_vals, group2_vals)
            else:
                dist = float('nan')  # If either group has no data

            results.append({
                'Feature': feature,
                f'{attr_names[0]}_count': len(group1_vals),
                f'{attr_names[1]}_count': len(group2_vals),
                'Distribution_Distance': dist
            })

        return pd.DataFrame(results).set_index('Feature')
    
    def compute_attribution_difference(self, explanations, X_test, sensitive_attr, attr_names, value_to_name):

        """
        Computes the difference in feature attributions between two groups (e.g., Male vs Female).

        This method calculates both:
        - The average contribution of each feature to predictions for each group
        - The average absolute contribution (magnitude) for each feature per group

        Args:
            explanations: List of explanations for each instance. 
            X_test: The test dataset containing feature values for each instance.
            sensitive_attr: The name of the sensitive attribute column (e.g., "Sex", "Race").
            attr_names: The two group names to compare (e.g., ["Male", "Female"] or ["White", "Black"]).

        """
        
        def extract_feature_name(feature_text):
            feature_text_lower = feature_text.lower()
            matches = [f for f in  X_test.columns if f.lower() in feature_text_lower]
            return max(matches, key=len) if matches else feature_text

        group_attributions = {name: defaultdict(list) for name in attr_names}

        for i, exp in enumerate(explanations):
            group_value = X_test.iloc[i][sensitive_attr]
            group_name = value_to_name.get(group_value, None)
            if group_name is None:
                continue   
            
            for feature, value in exp:
                feature_group = extract_feature_name(feature)
                group_attributions[group_name][feature_group].append(value)

        means = {name: {} for name in attr_names}
        abs_means = {name: {} for name in attr_names}

        for group_name in attr_names:
            for feature, contribs in group_attributions[group_name].items():
                means[group_name][feature] = sum(contribs) / len(contribs)
                abs_means[group_name][feature] = sum(abs(x) for x in contribs) / len(contribs)

        all_features = set(means[attr_names[0]].keys()) | set(means[attr_names[1]].keys())

        comparison_df = pd.DataFrame(index=sorted(all_features))
        comparison_df[f'{attr_names[0]}_avg'] = comparison_df.index.map(lambda x: means[attr_names[0]].get(x, 0))
        comparison_df[f'{attr_names[1]}_avg'] = comparison_df.index.map(lambda x: means[attr_names[1]].get(x, 0))
        comparison_df['Difference'] = comparison_df[f'{attr_names[0]}_avg'] - comparison_df[f'{attr_names[1]}_avg']

        comparison_df[f'{attr_names[0]}_abs_avg'] = comparison_df.index.map(lambda x: abs_means[attr_names[0]].get(x, 0))
        comparison_df[f'{attr_names[1]}_abs_avg'] = comparison_df.index.map(lambda x: abs_means[attr_names[1]].get(x, 0))
        comparison_df['Abs_Difference'] = comparison_df[f'{attr_names[0]}_abs_avg'] - comparison_df[f'{attr_names[1]}_abs_avg']

        return comparison_df
    
    def save_attribution_difference(self, df, save_path, file_name):
        """
        Saves the attribution difference results to a file.

        Args:
            df including attribution differences.
            file_name.
           
        """
        os.makedirs(save_path, exist_ok=True)

        
        path = os.path.join(save_path, f"{file_name}.txt")
        with open(path, "w") as f:
            with redirect_stdout(f):
                print(df.round(6))
    
    def compute_counterfactual_difference(self, explanations, X_test, sensitive_attr, attr_names):
        """
        Computes the difference in counterfactual changes between two groups (e.g., Male vs Female).

        This method calculates both:
        - The percentage of times a feature was changed in counterfactuals for each group.
        - The average magnitude of change for each feature per group.

        Args:
            explanations: List of DiCE counterfactuals for each instance.
            X_test: The test dataset.
            sensitive_attr: The name of the sensitive attribute (e.g., "Sex", "Race").
            attr_names: The two group names to compare (e.g., ["Male", "Female"]).
            
        """

        change_counts = {name: defaultdict(int) for name in attr_names}
        change_magnitudes = {name: defaultdict(float) for name in attr_names}
        group_counts = {name: 0 for name in attr_names}

        for i, cf_list in enumerate(explanations):
            if not cf_list:
                continue   

            original = X_test.iloc[i]
            group_name = original[sensitive_attr]   

            if group_name not in attr_names:
                continue

            group_counts[group_name] += 1

            for cf in cf_list:
                for feature in X_test.columns:
                    if feature in cf and cf[feature] != original[feature]:
                        change_counts[group_name][feature] += 1
                        if pd.api.types.is_numeric_dtype(type(original[feature])):
                            change_magnitudes[group_name][feature] += abs(cf[feature] - original[feature])
                        else:
                            change_magnitudes[group_name][feature] += 1

        change_percent = {name: {} for name in attr_names}
        avg_mags = {name: {} for name in attr_names}

        for group in attr_names:
            for feature in change_counts[group]:
                change_percent[group][feature] = 100 * change_counts[group][feature] / group_counts[group]
                avg_mags[group][feature] = change_magnitudes[group][feature] / change_counts[group][feature]

        all_features = set(change_percent[attr_names[0]].keys()) | set(change_percent[attr_names[1]].keys())

        comparison_df = pd.DataFrame(index=sorted(all_features))
        comparison_df[f'{attr_names[0]}_change_percent'] = comparison_df.index.map(lambda x: change_percent[attr_names[0]].get(x, 0))
        comparison_df[f'{attr_names[1]}_change_percent'] = comparison_df.index.map(lambda x: change_percent[attr_names[1]].get(x, 0))
        comparison_df['Change_Diff'] = comparison_df[f'{attr_names[0]}_change_percent'] - comparison_df[f'{attr_names[1]}_change_percent']

        comparison_df[f'{attr_names[0]}_avg_mag'] = comparison_df.index.map(lambda x: avg_mags[attr_names[0]].get(x, 0))
        comparison_df[f'{attr_names[1]}_avg_mag'] = comparison_df.index.map(lambda x: avg_mags[attr_names[1]].get(x, 0))
        comparison_df['Mag_Diff'] = comparison_df[f'{attr_names[0]}_avg_mag'] - comparison_df[f'{attr_names[1]}_avg_mag']

        return comparison_df

    def save_counterfactual_difference(self, df, file_name):
        """
        Saves the counterfactual difference results to a file.

        Args:
            df including difference in percentages of changes and magnitudes.
            file_name: The name to use for the saved file (without extension).
        """
        os.makedirs(self.output_dir, exist_ok=True)

        path = os.path.join(self.output_dir, f"counterfactual_difference_{file_name}.txt")
        with open(path, "w") as f:
            with redirect_stdout(f):
                print(df.round(6))