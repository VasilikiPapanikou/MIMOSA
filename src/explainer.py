# lime_explainer.py
import streamlit as st
from lime import lime_tabular
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from matplotlib.ticker import MultipleLocator, FuncFormatter
import shap
from collections import defaultdict
import dice_ml
from dice_ml.utils import helpers
from scipy.stats import wasserstein_distance
import textwrap
import warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
class Explainer:
    
    def __init__(self, data_df, data_encoded, class_names, feature_names, categorical_names, categorical_indices, continuous_features=None):
        """
        Generic initialization for any explanation method.
        """
        self.data_df = data_df
        self.data_encoded = data_encoded
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.categorical_indices = categorical_indices
        self.continuous_features = continuous_features if continuous_features else []
        self.lime_explainer = None
        self.shap_explainer = None
        self.dice_explainer = None

    def init_lime(self):
        """Initializes the LIME explainer if it hasn't been already."""
        if self.lime_explainer is None:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.data_encoded.iloc[:,:-1].values, 
                feature_names=self.feature_names,
                class_names=self.class_names,
                categorical_features=self.categorical_indices,
                categorical_names=self.categorical_names,
                kernel_width=3
            )

    def init_shap(self, model):
        """Initializes the SHAP explainer if it hasn't been already."""
        if self.shap_explainer is None:
            self.shap_explainer = shap.Explainer(model, self.data_encoded.iloc[:,:-1])

    def init_dice(self, model):
        """Initializes the DiCE explainer if it hasn't been already."""
        if self.dice_explainer is None:
      
            train_dataset = self.data_df[self.feature_names + ["Target"]].copy()

            
            target_name = "Target" 
            
            d = dice_ml.Data(
                dataframe=train_dataset, 
                continuous_features=self.continuous_features,  
                outcome_name=target_name
            )            
            m = dice_ml.Model(model=model, backend="sklearn")
            self.dice_explainer = dice_ml.Dice(d, m, method="random")

    def _load_saved_explanations(self, save_path, data_subset):
        """Load only explanations that correspond exactly to the current rows."""
        with open(save_path, 'r') as f:
            explanations_list = json.load(f)

        if not isinstance(explanations_list, list):
            raise ValueError("Saved explanations must be a list.")

        if hasattr(data_subset, "index"):
            expected_ids = [int(instance_id) for instance_id in data_subset.index]
        else:
            expected_ids = list(range(len(data_subset)))

        actual_ids = [int(item["instance_id"]) for item in explanations_list]
        if actual_ids != expected_ids:
            raise ValueError("Saved explanations belong to a different instance subset.")

        return explanations_list

    def generate_lime_explanations_for_group(self, data_subset, predict_fn, save_path=None, num_features=10):
        """
        Generates LIME explanations for a group. 
        If 'save_path' is provided and the file exists, it loads from disk.
        Otherwise, it generates them and saves to disk.
        
        Args:
            data_subset (pd.DataFrame): The data to explain.
            predict_fn (function): The model's predict_proba function.
            save_path (str, optional): Full path (dir/filename.json) to save/load.
            num_features (int): Number of features per explanation.

        Returns:
            list: List of explanation dictionaries.
        """
        
       
        if save_path and os.path.exists(save_path):
            try:
                explanations_list = self._load_saved_explanations(save_path, data_subset)
                st.success(f"Explanations loaded.")
                return explanations_list
            except Exception as e:
                st.warning(f"Could not load existing file: {e}. Regenerating...")

        self.init_lime()
       
        explanations_list = []
        
       
        data_subset_values = data_subset.values if hasattr(data_subset, "values") else data_subset
        
        progress_bar = st.progress(0, text="Generating LIME explanations... (0%)")
        total = len(data_subset_values)
        
        for i in tqdm(range(total), desc="Generating LIME explanations"):
            instance = data_subset_values[i]
            instance_id = data_subset.index[i] if hasattr(data_subset, "index") else i
            
            exp = self.lime_explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            
            # We construct the dictionary exactly as you requested
            exp_dict = {
                "instance_id": int(instance_id),
                "explanation": exp.as_list(), 
            }
            explanations_list.append(exp_dict)
            
            
            percent_complete = (i + 1) / total
            progress_bar.progress(percent_complete, text=f"Generating LIME explanations... ({int(percent_complete * 100)}%)")
        
        progress_bar.empty() 

        
        if save_path:
            try:
                
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                
                with open(save_path, 'w') as f:
                    json.dump(explanations_list, f, indent=4)
                
            except Exception as e:
                st.warning(f"Failed to save explanations to file: {e}")

        return explanations_list
    
    def generate_shap_explanations_for_group(self, data_subset, model, save_path=None):
        """
        Generates SHAP explanations with a progress bar by processing in batches.
        """
        
        
        if save_path and os.path.exists(save_path):
            try:
                explanations_list = self._load_saved_explanations(save_path, data_subset)
                st.success(f"Explanations loaded.")
                return explanations_list
            except Exception as e:
                st.warning(f"Could not load SHAP file: {e}. Regenerating...")

        
        self.init_shap(model)

        
        if isinstance(data_subset, np.ndarray):
            data_subset_df = pd.DataFrame(data_subset, columns=self.feature_names)
        else:
            data_subset_df = data_subset

        
        dynamic_cat_map = {}
        if self.categorical_names and self.categorical_indices:
            idx_to_name = {i: name for i, name in enumerate(self.feature_names)}
            for feat_idx, labels in self.categorical_names.items():
                feat_name = idx_to_name.get(feat_idx)
                if feat_name:
                    dynamic_cat_map[feat_name] = {i: label for i, label in enumerate(labels)}

        
        explanations_list = []
        
        
        BATCH_SIZE = 20 
        total_rows = len(data_subset_df)
        
        progress_bar = st.progress(0, text="Calculating SHAP values... (0%)")

        for start_idx in range(0, total_rows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            
            
            batch_df = data_subset_df.iloc[start_idx:end_idx]
            
            
            shap_values_obj = self.shap_explainer(batch_df)
            
            
            values_batch = shap_values_obj.values
            if values_batch.ndim == 3:
                values_batch = values_batch[:, :, 1] 
            
            
            raw_data_batch = shap_values_obj.data
            if hasattr(raw_data_batch, "values"):
                 raw_data_batch = raw_data_batch.values

            
            for i in range(len(batch_df)):
               
                instance_id = batch_df.index[i]
                
                explanation_tuples = []
                for feat_idx, feat_name in enumerate(self.feature_names):
                    shap_val = float(values_batch[i, feat_idx])
                    raw_val = raw_data_batch[i][feat_idx]
                    
                    
                    if feat_name in dynamic_cat_map:
                        mapping = dynamic_cat_map[feat_name]
                        label = mapping.get(int(raw_val), str(raw_val))
                        feature_str = f"{feat_name}={label}"
                    else:
                        feature_str = f"{feat_name}={round(float(raw_val), 2)}"

                    explanation_tuples.append([feature_str, shap_val])
                
                
                explanation_tuples.sort(key=lambda x: abs(x[1]), reverse=True)

                exp_dict = {
                    "instance_id": int(instance_id),
                    "explanation": explanation_tuples
                }
                explanations_list.append(exp_dict)

            
            percent_complete = end_idx / total_rows
            progress_bar.progress(percent_complete, text=f"Calculating SHAP values... ({int(percent_complete * 100)}%)")

        progress_bar.empty()

        
        if save_path:
            try:
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(explanations_list, f, indent=4)
            except Exception as e:
                st.warning(f"Failed to save SHAP explanations: {e}")

        return explanations_list

    def generate_dice_explanations_for_group(
        self,
        data_subset,
        model,
        save_path=None,
        total_CFs=1,
        desired_class="opposite",
        features_to_vary=None,
        sample_size=500,
    ):        
        """
        Generates DiCE explanations for a group.
        
        Args:
            data_subset (pd.DataFrame): The data (original features + target) to explain.
            model: The trained model object.
            ... (other args)
        """ 
       
        if save_path and os.path.exists(save_path):
            try:
                explanations_list = self._load_saved_explanations(save_path, data_subset)
                st.success(f"Explanations loaded.")
                return explanations_list
            except Exception as e:
                st.warning(f"Could not load DiCE file: {e}. Regenerating...")

        
        self.init_dice(model)
        
       
        data_subset_df = data_subset.copy()

        
        query_instances = data_subset_df[self.feature_names] 

        
        total = len(query_instances)
        st.write(f"Calculating DiCE counterfactuals for {total} instances...")
        progress_bar = st.progress(0, text="Calculating DiCE counterfactuals... (0%)")
        explanations_list = []
        failed_count = 0
        
        for i in tqdm(range(total), desc="Generating DiCE counterfactuals"):
            original_index = data_subset_df.index[i]
            query_instance = query_instances.iloc[[i]]
            cfs_data = []

            try:
                dice_kwargs = {
                    "total_CFs": total_CFs,
                    "desired_class": desired_class,
                    "diversity_weight": 0,
                    "sample_size": sample_size,
                    "random_seed": 42,
                }
                if features_to_vary:
                    dice_kwargs["features_to_vary"] = features_to_vary

                dice_exp = self.dice_explainer.generate_counterfactuals(
                    query_instance,
                    **dice_kwargs
                )

                cfe = dice_exp.cf_examples_list[0] if dice_exp.cf_examples_list else None
                if cfe and cfe.final_cfs_df is not None:
                    cfs_data = cfe.final_cfs_df.to_dict(orient="records")
            except Exception:
                failed_count += 1

            explanations_list.append({
                "instance_id": int(original_index),
                "cfs": cfs_data
            })
            
            percent_complete = (i + 1) / total
            progress_bar.progress(
                percent_complete,
                text=f"Calculating DiCE counterfactuals... ({int(percent_complete * 100)}%)"
            )

        progress_bar.empty()
        if failed_count:
            st.warning(
                f"DiCE did not find counterfactuals for {failed_count} of {total} instances. "
                "Those instances were skipped in DiCE feature-change summaries."
            )


        
        if save_path:
            try:
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(explanations_list, f, indent=4)
                st.success("✅ DiCE explanations generation complete.")
            except Exception as e:
                st.warning(f"Failed to save DiCE explanations: {e}")

        return explanations_list
     
    def generate_explanations(self, method, data_subset, model_or_predict_fn, save_path=None, num_features=10):
        """
        Generic function that calls either LIME or SHAP based on the 'method' string.
        """
        if method == "LIME":
            return self.generate_lime_explanations_for_group(
                data_subset, 
                model_or_predict_fn, 
                save_path=save_path,
                num_features=num_features
            )
        elif method == "SHAP":
            return self.generate_shap_explanations_for_group(
                data_subset, 
                model_or_predict_fn, 
                save_path=save_path
            )
        elif method == "DiCE":
            return self.generate_dice_explanations_for_group(
                data_subset, model_or_predict_fn, save_path=save_path
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_counterfactual_difference(self, explanations_per_group, group_names, data_subsets):
        """
        Calculates the percentage of times each feature changed in the counterfactuals.
        """
        change_percent = {}

        for group in group_names:
            explanations = explanations_per_group.get(group, [])
            original_data = data_subsets.get(group)  
            
            if not explanations or original_data is None:
                change_percent[group] = {}
                continue

            feature_changes = defaultdict(int)
            valid_cf_count = 0
            
            for i, exp_dict in enumerate(explanations):
                cfs_list = exp_dict.get('cfs', [])
                if not cfs_list: 
                    continue
                
                cf = cfs_list[0]
                valid_cf_count += 1

                
                if i >= len(original_data): break
                original_row = original_data.iloc[i]

                
                for feature in original_data.columns:
                    if feature not in cf: continue

                    val_orig = original_row[feature]
                    val_cf = cf[feature]

                    
                    is_different = False
                    try:
                        
                        if abs(float(val_orig) - float(val_cf)) > 1e-5:
                            is_different = True
                    except:
                        
                        if str(val_orig) != str(val_cf):
                            is_different = True
                    
                    if is_different:
                        feature_changes[feature] += 1
            group_percent = {}
            if valid_cf_count > 0:
                for feature, count in feature_changes.items():
                    group_percent[feature] = (count / valid_cf_count) * 100
            
            change_percent[group] = group_percent

        return pd.DataFrame(change_percent).fillna(0)
    
    def extract_feature_name(self, feature_text):
        feature_text = str(feature_text).strip()
        # Prefer the actual feature vocabulary so names such as Credit-Amount
        # and multi-word categorical features remain intact.
        for feature_name in sorted(self.feature_names, key=len, reverse=True):
            suffix = feature_text[len(feature_name):len(feature_name) + 1]
            if feature_text == feature_name or (
                feature_text.startswith(feature_name)
                and suffix in {" ", "=", "<", ">"}
            ):
                return feature_name

        match = re.search(r'[A-Za-z_][A-Za-z0-9_\s]*', feature_text)
        if match:
            return match.group(0).strip()  
        
        if '=' in feature_text:
            return feature_text.split('=')[0].strip()
        if '<' in feature_text or '>' in feature_text:
            return feature_text.split()[0].strip()
        
        return feature_text.strip()
    
    def plot_violin_distributions(self, explanations_per_group, group_names, custom_colors, feature_order=None):
        df = self._prepare_plot_data(explanations_per_group, group_names)
        if df.empty: return None
        
        if feature_order:
            df = df[df['Feature'].isin(feature_order)]
        
        fig, ax = plt.subplots(figsize=(5, 3))
        
        sns.violinplot(
            data=df, x="Feature", y="Contribution", hue="Group",
            hue_order=group_names, palette=custom_colors,
            order=feature_order, 
            split=True, inner="quart", scale="width", linewidth=1.0, ax=ax
        )
        
        ax.set_ylabel("Contribution", fontsize=7)
        ax.set_xlabel("")
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
       
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = [
            textwrap.fill(text, width=10, break_long_words=False) 
            for text in labels
        ]
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=6)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        ax.legend(
            title=None,
            fontsize=6,
            loc="upper right",        
            frameon=True,             
            borderaxespad=0.2         
        )
        
        plt.tight_layout()
        return fig
    
    def compute_wasserstein_table(self, explanations_per_group, group_names, feature_order=None):
        from scipy.stats import wasserstein_distance  
        
        
        df = self._prepare_plot_data(explanations_per_group, group_names)
        if feature_order:
            df = df[df['Feature'].isin(feature_order)]
            features_to_process = feature_order
        else:
            features_to_process = df['Feature'].unique()

        results = []
        g1_name, g2_name = group_names[0], group_names[1]

        for feature in features_to_process:
            vals_g1 = df[(df['Feature'] == feature) & (df['Group'] == g1_name)]['Contribution']
            vals_g2 = df[(df['Feature'] == feature) & (df['Group'] == g2_name)]['Contribution']
            
            dist = 0.0
            if not vals_g1.empty and not vals_g2.empty:
                dist = wasserstein_distance(vals_g1, vals_g2)
            
            results.append({
                "Feature": feature,
                "Wasserstein Difference": dist
            })

        
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.set_index("Feature")
            
        return res_df

    def compute_mean_diff_table(self, explanations_per_group, group_names, feature_order=None):
        """Calculates the absolute difference between the Mean contribution of Group 1 and Group 2."""
        df = self._prepare_plot_data(explanations_per_group, group_names)
        
        if feature_order:
            df = df[df['Feature'].isin(feature_order)]
            features_to_process = feature_order
        else:
            features_to_process = df['Feature'].unique()

        results = []
        g1_name, g2_name = group_names[0], group_names[1]

        for feature in features_to_process:
            vals_g1 = df[(df['Feature'] == feature) & (df['Group'] == g1_name)]['Contribution']
            vals_g2 = df[(df['Feature'] == feature) & (df['Group'] == g2_name)]['Contribution']
            
            diff = 0.0
            if not vals_g1.empty and not vals_g2.empty:
                diff = abs(vals_g1.mean() - vals_g2.mean())
            
            results.append({"Feature": feature, "Difference": diff})

        res_df = pd.DataFrame(results)
        if not res_df.empty: res_df = res_df.set_index("Feature")
        return res_df

    def compute_mean_abs_diff_table(self, explanations_per_group, group_names, feature_order=None):
        """Calculates the difference between the Mean Absolute Importance of Group 1 and Group 2."""
        df = self._prepare_plot_data(explanations_per_group, group_names)
        
        if feature_order:
            df = df[df['Feature'].isin(feature_order)]
            features_to_process = feature_order
        else:
            features_to_process = df['Feature'].unique()

        results = []
        g1_name, g2_name = group_names[0], group_names[1]

        for feature in features_to_process:
            vals_g1 = df[(df['Feature'] == feature) & (df['Group'] == g1_name)]['Contribution']
            vals_g2 = df[(df['Feature'] == feature) & (df['Group'] == g2_name)]['Contribution']
            
            diff = 0.0
            if not vals_g1.empty and not vals_g2.empty:
                mean_abs_g1 = vals_g1.abs().mean()
                mean_abs_g2 = vals_g2.abs().mean()
                diff = abs(mean_abs_g1 - mean_abs_g2)
            
            results.append({"Feature": feature, "Difference": diff})

        res_df = pd.DataFrame(results)
        if not res_df.empty: res_df = res_df.set_index("Feature")
        return res_df
    
    
    def plot_instance_heatmap(self, explanations_per_group, group_names, n_instances=10, feature_order=None):
        """
        Generates a side-by-side heatmap of instance explanations.
        Ensures consistent cell sizes and a shared colorbar.
        """
        import matplotlib.gridspec as gridspec
        
        dfs = {}
        all_values = []
        
        for group in group_names:
            exps = explanations_per_group.get(group, [])[:n_instances]
            if not exps: 
                dfs[group] = pd.DataFrame()
                continue
                
            rows_data = []
            for exp in exps:
                row_dict = {'ID': str(exp.get('instance_id', '?'))}
                for feat_str, val in exp['explanation']:
                    clean_feat = self.extract_feature_name(feat_str)
                    row_dict[clean_feat] = val
                    all_values.append(val)
                rows_data.append(row_dict)
            
            df = pd.DataFrame(rows_data).set_index('ID')
            
            if feature_order:
                df = df.reindex(columns=feature_order, fill_value=0)
            
            dfs[group] = df

        if not all_values: return None
        
        
        vmin = min(all_values)
        vmax = max(all_values)
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit 

        
        n_groups = len(group_names)
        fig_height = max(4, 0.5 * n_instances) 
        fig = plt.figure(figsize=(6 * n_groups + 1, fig_height)) 
        
       
        gs = gridspec.GridSpec(1, n_groups + 1, width_ratios=[1]*n_groups + [0.05], wspace=0.1)
        
        axes = []
        
        
        for i, group in enumerate(group_names):
            ax = plt.subplot(gs[0, i])
            axes.append(ax)
            
            df = dfs[group]
            
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.axis('off')
                continue
            
            
            sns.heatmap(
                df, 
                cmap="vlag", 
                vmin=vmin, vmax=vmax,
                center=0,
                annot=True, 
                fmt=".2f",
                cbar=False, 
                linewidths=0.5, 
                linecolor='white',
                annot_kws={"size": 8},
                ax=ax
            )
            
            ax.set_title(f"{group}", fontsize=14, pad=10)
            
            
            import textwrap
            x_labels = [item.get_text() for item in ax.get_xticklabels()]
            wrapped_labels = [textwrap.fill(text, width=12) for text in x_labels]
            ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=9)
            
            
            if i == 0:
                ax.set_ylabel("Instance", fontsize=10)
                ax.set_yticks([])
            else:
                ax.set_ylabel("")
                ax.set_yticks([]) 
            
            ax.set_xlabel("")

        
        cbar_ax = plt.subplot(gs[0, n_groups])
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Feature Contribution", fontsize=10)
        cbar.outline.set_visible(False)

        return fig 
    
    
    def get_global_feature_order(self, explanations_per_group, group_names, top_n=10):
        """Calculates the top N features based on total absolute importance across ALL groups."""
        df = self._prepare_plot_data(explanations_per_group, group_names)
        if df.empty: return []
        
        feature_importance = df.groupby('Feature')['Contribution'].apply(lambda x: x.abs().sum())
        return feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
    
    def get_global_feature_order_dice(self, explanations_per_group, group_names, data_subsets, top_n=10):
        """
        Computes global feature importance for DiCE explanations.
        Importance = sum of abs(Delta) across all CFs.
        """
        df = self._prepare_dice_plot_data(explanations_per_group, group_names, data_subsets)

        if df.empty:
            return []

        
        df["AbsDelta"] = df["Delta"].abs()

        feature_importance = (
            df.groupby("Feature")["AbsDelta"]
            .sum()
            .sort_values(ascending=False)
        )

        return feature_importance.head(top_n).index.tolist()

    @staticmethod
    def _prediction_probabilities(predict_fn, data):
        """Return a 2-D class-probability array for a model or prediction wrapper."""
        # Some model wrappers add a temporary target column; isolate that mutation.
        model_input = data.copy() if hasattr(data, "copy") else data
        probabilities = predict_fn(model_input) if callable(predict_fn) else predict_fn.predict_proba(model_input)
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("AOPC requires binary class probabilities.")
        return probabilities

    @staticmethod
    def _feature_baselines(reference_data, feature_names, categorical_features=None):
        """Build valid replacement values for numeric and categorical columns."""
        categorical_features = set(categorical_features or [])
        baselines = {}
        for feature in feature_names:
            if feature not in reference_data.columns:
                continue
            series = reference_data[feature].dropna()
            if series.empty:
                continue
            if feature in categorical_features:
                baselines[feature] = series.mode().iloc[0]
            else:
                baselines[feature] = series.median()
        return baselines

    def compute_aopc_global(
        self,
        predict_fn,
        X,
        feature_ranking,
        K,
        reference_data=None,
        categorical_features=None,
    ):
        """Compute the global confidence-drop AOPC used to compare rankings."""
        if X is None or len(X) == 0:
            return np.nan

        X = X.copy()
        valid_ranking = [feature for feature in feature_ranking if feature in X.columns]
        K = min(max(int(K), 0), len(valid_ranking))
        if K == 0:
            return 0.0

        reference_data = reference_data if reference_data is not None else X
        baselines = self._feature_baselines(
            reference_data, X.columns, categorical_features
        )
        original_probs = self._prediction_probabilities(predict_fn, X)
        predicted_classes = original_probs.argmax(axis=1)
        original_class_probs = original_probs[
            np.arange(len(X)), predicted_classes
        ]

        cumulative_diffs = np.zeros(len(X), dtype=float)
        for k in range(K + 1):
            X_modified = X.copy()
            for feature in valid_ranking[:k]:
                if feature in baselines:
                    X_modified.loc[:, feature] = baselines[feature]

            new_probs = self._prediction_probabilities(predict_fn, X_modified)
            new_class_probs = new_probs[np.arange(len(X)), predicted_classes]
            cumulative_diffs += original_class_probs - new_class_probs

        return float(np.mean(cumulative_diffs) / (K + 1))

    def compare_aopc_methods(
        self,
        predict_fn,
        X,
        rankings,
        K_max,
        reference_data=None,
        categorical_features=None,
        random_runs=5,
        random_state=42,
    ):
        """Compare explanation rankings and a shuffled ranking using global AOPC."""
        rng = np.random.default_rng(random_state)
        results = {}
        curves = {}

        for method, ranking in rankings.items():
            curve = [
                self.compute_aopc_global(
                    predict_fn, X, ranking, k, reference_data, categorical_features
                )
                for k in range(K_max + 1)
            ]
            curves[method] = curve
            results[method] = curve[-1] if curve else np.nan

        random_curves = []
        for _ in range(random_runs):
            random_ranking = list(X.columns)
            rng.shuffle(random_ranking)
            random_curves.append([
                self.compute_aopc_global(
                    predict_fn, X, random_ranking, k, reference_data, categorical_features
                )
                for k in range(K_max + 1)
            ])

        random_mean = np.nanmean(random_curves, axis=0) if random_curves else []
        random_std = np.nanstd(random_curves, axis=0) if random_curves else []
        curves["Random"] = random_mean.tolist() if len(random_mean) else []
        results["Random"] = random_mean[-1] if len(random_mean) else np.nan

        best_method = max(
            rankings,
            key=lambda method: results[method] if not np.isnan(results[method]) else -np.inf,
        )
        return {
            "best_method": best_method,
            "scores": results,
            "curves": curves,
            "random_std": random_std.tolist() if len(random_std) else [],
        }

    def _prepare_plot_data(self, explanations_per_group, group_names, data_per_group=None):
        """
        Helper to aggregate explanation data for plots.
        If data_per_group is provided, it extracts the raw feature value for coloring.
        """
        rows = []
        for group in group_names:
            exps = explanations_per_group.get(group, [])
            
            
            group_df = None
            if data_per_group and group in data_per_group:
                group_df = data_per_group[group]

            for i, exp_dict in enumerate(exps):
                
                data_row = None
                if group_df is not None and i < len(group_df):
                    data_row = group_df.iloc[i]

                
                for feat_str, val in exp_dict["explanation"]:
                    clean_feat = self.extract_feature_name(feat_str)
                    
                    
                    raw_val = np.nan
                    if data_row is not None and clean_feat in data_row:
                        raw_val = data_row[clean_feat]
                    
                    rows.append({
                        'Group': group,
                        'Feature': clean_feat,
                        'Contribution': val,
                        'Value': raw_val  
                    })
                    
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    
    def plot_mean_aggregation(self, explanations_per_group, group_names, custom_colors, feature_order=None):
        df = self._prepare_plot_data(explanations_per_group, group_names)
        if df.empty: return None

        mean_df = df.groupby(['Group', 'Feature'])['Contribution'].mean().reset_index(name='Mean')
        
        if feature_order:
            mean_df = mean_df[mean_df['Feature'].isin(feature_order)]

        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        sns.barplot(
            data=mean_df, x='Feature', y='Mean', hue='Group', 
            hue_order=group_names, palette=custom_colors, 
            order=feature_order, 
            ax=ax
        )

       
        ax.set_ylabel("Mean Contribution", fontsize=7)
        ax.set_xlabel("")
        ax.axhline(0, color='black', linewidth=0.8)

       
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = [
            textwrap.fill(text, width=10, break_long_words=False) 
            for text in labels
        ]
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=6)
        
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        
         
        ax.legend(
            title=None,
            fontsize=6,
            loc="upper right",        
            frameon=True,             
            borderaxespad=0.2         
        )

        plt.tight_layout()
        return fig

    def plot_mean_abs_aggregation(self, explanations_per_group, group_names, custom_colors, feature_order=None):
        df = self._prepare_plot_data(explanations_per_group, group_names)
        if df.empty: return None

        df['Abs_Contribution'] = df['Contribution'].abs()
        mean_abs_df = df.groupby(['Group', 'Feature'])['Abs_Contribution'].mean().reset_index(name='Abs_Mean')

        if feature_order:
            mean_abs_df = mean_abs_df[mean_abs_df['Feature'].isin(feature_order)]

         
        fig, ax = plt.subplots(figsize=(5, 3))
        
        sns.barplot(
            data=mean_abs_df, x='Feature', y='Abs_Mean', hue='Group', 
            hue_order=group_names, palette=custom_colors, 
            order=feature_order, 
            ax=ax
        )

    
        
        ax.set_ylabel("Mean |Contribution|", fontsize=7)
        ax.set_xlabel("")
        
       
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = [
            textwrap.fill(text, width=10, break_long_words=False) 
            for text in labels
        ]
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=6)
        
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
         
        ax.legend(
            title=None,
            fontsize=6,
            loc="upper right",        
            frameon=True,             
            borderaxespad=0.2         
        )

        plt.tight_layout()
        return fig
    
    def plot_beeswarm_comparison(
        self, 
        explanations_per_group, 
        group_names, 
        custom_colors, 
        feature_order=None, 
        data_per_group=None
    ):
        """
        Generates side-by-side SHAP-style beeswarm plots.
        - Numerical features: Colored by Feature Value (Blue=Low, Red=High).
        - Categorical features: Colored by Group Color (Solid).
        """

        df = self._prepare_plot_data(explanations_per_group, group_names, data_per_group)
        if df.empty:
            return None

        if feature_order:
            df = df[df['Feature'].isin(feature_order)]

        is_numeric = df['Feature'].isin(self.continuous_features)

        df['Value_Normalized'] = np.nan

        if is_numeric.any() and 'Value' in df.columns:
            df.loc[is_numeric, 'Value_Normalized'] = (
                df.loc[is_numeric]
                .groupby('Feature')['Value']
                .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
            )
            df.loc[is_numeric, 'Value_Normalized'] = df.loc[is_numeric, 'Value_Normalized'].fillna(0.5)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
        for i, group in enumerate(group_names):
            ax = axes[i]

            group_data = df[df['Group'] == group]

            numeric_rows = group_data[group_data['Feature'].isin(self.continuous_features)]
            categorical_rows = group_data[~group_data['Feature'].isin(self.continuous_features)]

            if not numeric_rows.empty:
                sns.stripplot(
                    data=numeric_rows,
                    y="Feature",
                    x="Contribution",
                    order=feature_order,
                    hue="Value_Normalized",
                    palette="coolwarm",
                    size=4,
                    alpha=0.8,
                    jitter=0.25,
                    ax=ax,
                    legend=False,
                    zorder=2
                )
            if not categorical_rows.empty:
                solid_color = custom_colors.get(group, "#333333")
                sns.stripplot(
                    data=categorical_rows,
                    y="Feature",
                    x="Contribution",
                    order=feature_order,
                    color=solid_color,
                    size=4,
                    alpha=0.6,
                    jitter=0.25,
                    ax=ax,
                    zorder=2
                )

            ax.set_title(f"{group}", fontsize=14)
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.2)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            ax.set_xlabel("Contribution")

            if i == 0:
                ax.set_ylabel("Feature", fontsize=12)
            else:
                ax.set_ylabel("")
                ax.tick_params(left=False)

        if is_numeric.any():
            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            sm.set_array([])

            cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Feature Value", fontsize=10)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Low', 'High'])

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        return fig


    def plot_dice_comparison(self, df_changes, group_names, custom_colors, feature_order=None):
        """
        Plots the counterfactual feature changes.
        Accepts optional feature_order to explicitly order the bars.
        """
        if df_changes.empty:
            return None

        
        cols_to_check = [g for g in group_names if g in df_changes.columns]
        if not cols_to_check:
            return None

        
        df_changes = df_changes[(df_changes[cols_to_check] != 0).any(axis=1)]
        if df_changes.empty:
            return None

        
        ordered_feats = [f for f in feature_order if f in df_changes.index]

       
        if len(ordered_feats) == 0:
            use_order = False
        else:
            use_order = True
         

        if use_order:
            df_changes = df_changes.loc[ordered_feats]
            df_changes = df_changes.head(10)   
        else:
            
            df_changes['max_change'] = df_changes[cols_to_check].max(axis=1)
            df_changes = df_changes.sort_values('max_change', ascending=False)
            df_changes = df_changes.drop(columns='max_change')
            df_changes = df_changes.head(10)

        
        features = df_changes.index
        n_features = len(features)
        n_groups = len(group_names)

        fig, ax = plt.subplots(figsize=(6, 4))

        bar_width = 0.8 / n_groups
        x_pos = np.arange(n_features)

        for i, group in enumerate(group_names):
            if group not in df_changes.columns:
                continue

            values = df_changes[group].values
            ax.bar(
                x_pos + (i * bar_width),
                values,
                width=bar_width,
                label=group,
                color=custom_colors.get(group, "#333")
            )

        ax.set_xticks(x_pos + bar_width * (n_groups - 1) / 2)

        wrapped_labels = [
            textwrap.fill(str(text), width=12, break_long_words=False) 
            for text in features
        ]

        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=9)
        ax.set_ylabel("Frequency of Change (%)", fontsize=10)
        ax.legend(title=None, fontsize=9, ncol=2, loc="upper right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}%'))
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_dice_comparison(self, df_changes, group_names, custom_colors, feature_order=None):
        """
        Plots the counterfactual feature changes.
        Accepts optional feature_order to explicitly order the bars.
        """
        if df_changes.empty:
            return None
        
        cols_to_check = [g for g in group_names if g in df_changes.columns]
        if not cols_to_check:
            return None

      
        df_changes = df_changes[(df_changes[cols_to_check] != 0).any(axis=1)]
        if df_changes.empty:
            return None

       
        use_order = False
        
        if feature_order:
            ordered_feats = [f for f in feature_order if f in df_changes.index]
            if len(ordered_feats) > 0:
                use_order = True

        if use_order:
            df_changes = df_changes.loc[ordered_feats]
            df_changes = df_changes.head(10)   
        else:
            df_changes['max_change'] = df_changes[cols_to_check].max(axis=1)
            df_changes = df_changes.sort_values('max_change', ascending=False)
            df_changes = df_changes.drop(columns='max_change')
            df_changes = df_changes.head(10)

       
        features = df_changes.index
        n_features = len(features)
        n_groups = len(group_names)

        
        fig, ax = plt.subplots(figsize=(6, 4))

        bar_width = 0.8 / n_groups
        x_pos = np.arange(n_features)

        for i, group in enumerate(group_names):
            if group not in df_changes.columns:
                continue

            values = df_changes[group].values
            ax.bar(
                x_pos + (i * bar_width),
                values,
                width=bar_width,
                label=group,
                color=custom_colors.get(group, "#333")
            )

      
        ax.set_xticks(x_pos + bar_width * (n_groups - 1) / 2)

       
        wrapped_labels = [
            textwrap.fill(str(text), width=12, break_long_words=False) 
            for text in features
        ]

       
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=6)
        ax.set_ylabel("Frequency of Change (%)", fontsize=10)
        
       

        ax.legend(title=None, fontsize=6, ncol=2, loc="upper right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}%'))
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def _prepare_dice_plot_data(self, explanations_per_group, group_names, data_subsets):
        """
        Prepares instance-level data for DiCE Beeswarm plots.
        Returns DataFrame with: [Group, Feature, Delta, FactualValue]
        """
        rows = []
        
        for group in group_names:
            explanations = explanations_per_group.get(group, [])
            original_data = data_subsets.get(group)
            
            if not explanations or original_data is None: continue
            
            for i, exp_dict in enumerate(explanations):
                cfs_list = exp_dict.get('cfs', [])
                if not cfs_list: continue
                
                if i >= len(original_data): break
                original_row = original_data.iloc[i]
                
                cf = cfs_list[0] 
                
                for feature in original_data.columns[:-1]: 
                    if feature not in cf: continue
                    
                    val_orig = original_row[feature]
                    val_cf = cf[feature]
                    
                    delta = 0
                    is_numeric = feature in self.continuous_features
                    
                    if is_numeric:
                        try:
                            delta = float(val_cf) - float(val_orig)
                        except:
                            delta = 0
                    else:
                        delta = 1 if str(val_orig) != str(val_cf) else 0
                    
                    rows.append({
                        "Group": group,
                        "Feature": feature,
                        "Delta": delta,          
                        "FactualValue": val_orig, 
                        "IsNumeric": is_numeric
                    })
                        
        return pd.DataFrame(rows)
    

    def plot_dice_beeswarm(
        self,
        explanations_per_group,
        group_names,
        custom_colors,
        data_subsets,
        feature_order=None
    ):
        """
        DiCE Beeswarm plot with SAME STYLE as plot_beeswarm_comparison().
        - X-axis: Delta (CF - factual)
        - Y-axis: Features
        - Numeric features: colored by normalized factual value
        - Categorical: solid group color
        """

    
        df = self._prepare_dice_plot_data(explanations_per_group, group_names, data_subsets)
        if df.empty:
            return None

       
        df["Value_Normalized"] = np.nan
        numeric_mask = df["IsNumeric"]

        if numeric_mask.any():
            vals = df.loc[numeric_mask, "FactualValue"].astype(float)
            df.loc[numeric_mask, "Value_Normalized"] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

            df.loc[numeric_mask, "Value_Normalized"] = df.loc[numeric_mask, "Value_Normalized"].fillna(0.5)

        
        feature_order = [f for f in feature_order if f in df["Feature"].unique()]

        df = df[df["Feature"].isin(feature_order)]

  
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

        
        for i, group in enumerate(group_names):
            ax = axes[i]

            group_df = df[df["Group"] == group]
            df_num = group_df[group_df["IsNumeric"]]
            df_cat = group_df[~group_df["IsNumeric"]]

           
            if not df_num.empty:
                sns.stripplot(
                    data=df_num,
                    x="Delta",
                    y="Feature",
                    order=feature_order,
                    hue="Value_Normalized",
                    palette="coolwarm",
                    jitter=0.25,
                    size=4,
                    alpha=0.8,
                    ax=ax,
                    legend=False,
                    zorder=2
                )

            
            if not df_cat.empty:
                sns.stripplot(
                    data=df_cat,
                    x="Delta",
                    y="Feature",
                    order=feature_order,
                    color=custom_colors.get(group, "#333"),
                    jitter=0.25,
                    size=4,
                    alpha=0.6,
                    ax=ax,
                    zorder=2
                )

           
            ax.set_title(group, fontsize=14)
            ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.2)
            ax.grid(True, axis="x", linestyle="--", alpha=0.5)
            ax.set_xlabel("Change required (CF - Original)")

            if i == 0:
                ax.set_ylabel("Feature", fontsize=12)
            else:
                ax.set_ylabel("")
                ax.tick_params(left=False)

        
        if numeric_mask.any():
            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            sm.set_array([])

            cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Feature Value", fontsize=10)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Low", "High"])

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        return fig
