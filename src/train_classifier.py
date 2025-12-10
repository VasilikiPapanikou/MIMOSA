import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.base import clone

# AIF360 & TensorFlow
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class DiceModelWrapper:
    def __init__(self, model, label_encoders=None, model_type="random_forest", session=None, feature_names=None):
        self.model = model
        self.label_encoders = label_encoders or {}
        self.model_type = model_type  
        self.session = session
        self.feature_names = feature_names  

    def predict_proba(self, x):
        # 1. Prepare DataFrame from input (numpy or df)
        if isinstance(x, pd.DataFrame):
            x_df = x.copy()
        else:
            x_df = pd.DataFrame(x, columns=self.feature_names)

        # 2. Encode categorical features if needed
        # (DiCE generates decoded counterfactuals, but models often need encoded values)
        for feature, encoder in self.label_encoders.items():
            if feature in x_df.columns:
                try:
                    # Convert to string to match encoder's expected input type
                    x_df[feature] = encoder.transform(x_df[feature].astype(str))
                except:
                    pass

        # 3. Prediction Logic
        if self.model_type == "MLP":
            # AIF360 / TensorFlow Path
            dummy_labels = np.zeros(len(x_df))
            
            # Detect protected attributes dynamically for the wrapper
            protected_attrs = [c for c in x_df.columns if "Sex" in c or "Race" in c]
            
            aif_data = BinaryLabelDataset(
                df=x_df.assign(target=dummy_labels),
                label_names=['target'],
                protected_attribute_names=protected_attrs,
                favorable_label=1, unfavorable_label=0
            )

            # Use the active TF session
            with self.session.as_default():
                scores = self.model.predict(aif_data).scores
            
            # Return [Prob(0), Prob(1)]
            return np.column_stack([1 - scores, scores])

        else:
            # Standard Sklearn Path
            return self.model.predict_proba(x_df)


class ClassifierTrainer:
    def __init__(self, dataset_name, feature_names, categorical_features, label_encoders, privileged_groups=None, unprivileged_groups=None, debias=False, model_dir="saved_models"):
        self.dataset_name = dataset_name
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.label_encoders = label_encoders
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.debias = debias  # This determines if we run Adversarial (True) or Standard (False) logic in MLP
        self.model_dir = model_dir
        
        self.model = None
        self.model_type = None
        self.session = None
        os.makedirs(self.model_dir, exist_ok=True)

    def get_model_and_params(self, model_type):
        if model_type == 'Random Forest':
            model = RandomForestClassifier(random_state=42, n_jobs=1)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
        elif model_type == 'Logistic Regression':
            model = LogisticRegression(max_iter=5000)
            param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        elif model_type == 'XGBoost':
            model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        elif model_type == 'MLP':
            # We return None because we handle MLP construction manually in train()
            return None, None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model, param_grid

    def train(self, data_df, data_encoded, model=None, model_type='random_forest', enable_plots=False, adversarial_sensitive_attrs=None):
        self.model_type = model_type
        y = data_df["Target"]
        X = data_encoded.drop(columns=["Target"])

        model_path = os.path.join(self.model_dir, f"{model_type}_{self.dataset_name}_best_model.joblib")
        param_path = os.path.join(self.model_dir, f"{model_type}_{self.dataset_name}_best_params.joblib")

        split_seed = 42 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_seed, stratify=y
        ) 

        best_params = "Default/Loaded"

        if model is not None:
            print(f"--- Using pre-loaded {model_type} model ---")
            clf = model 
            self.model = clf
            best_params = "Loaded from disk"
            
            # Re-init session for loaded MLP models
            if model_type == 'MLP':
                try:
                    tf.reset_default_graph()
                    sess = tf.Session()
                    self.session = sess
                    sess.run(tf.global_variables_initializer())
                except: pass
        else:
            # --- TRAINING NEW MODEL ---
            print(f"--- Training new {model_type} model (Debias={self.debias}) ---")
            
            if model_type == 'MLP':
                # ==========================================
                # AIF360 ADVERSARIAL DEBIASING (For BOTH Standard & Debiased)
                # ==========================================
                tf.reset_default_graph()
                sess = tf.Session()
                self.session = sess

                # 1. Identify Protected Attributes
                # If passed explicitly (Screen 3), use them. Otherwise detect (Screen 1).
                if adversarial_sensitive_attrs:
                    protected_attrs = adversarial_sensitive_attrs
                else:
                    protected_attrs = [col for col in X_train.columns if "Sex" in col or "Race" in col]
                
                # 2. Define Groups (Required for AIF360 even if debias=False)
                # Assuming 1 is privileged. Adjust if your encoding differs.
                privileged_groups = [{attr: 1 for attr in protected_attrs}]
                unprivileged_groups = [{attr: 0 for attr in protected_attrs}]

                # 3. Create Dataset
                train_df = X_train.copy()
                train_df['target'] = y_train.values
                
                aif_dataset = BinaryLabelDataset(
                    df=train_df, 
                    label_names=["target"],
                    protected_attribute_names=protected_attrs,
                    favorable_label=1, 
                    unfavorable_label=0
                )
                
                # 4. Train Model
                # self.debias controls if the adversary is active (True) or inactive (False)
                self.model = AdversarialDebiasing(
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    scope_name='debiased_classifier',
                    debias=self.debias, 
                    adversary_loss_weight=0.1, 
                    sess=sess
                )
                
                clf = self.model
                clf.fit(aif_dataset)
                best_params = f"AdversarialDebiasing(debias={self.debias})"
                
                # Note: We do NOT use joblib to save this TF model to disk here 
                # because sessions don't pickle well. It stays in memory.

            else:
                # ==========================================
                # STANDARD SCIKIT-LEARN MODELS
                # ==========================================
                base_model, param_grid = self.get_model_and_params(model_type)
                
                grid_search = GridSearchCV(
                    estimator=base_model, param_grid=param_grid,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring='accuracy', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train) 
                clf = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                joblib.dump(clf, model_path)
                joblib.dump(best_params, param_path)

            self.model = clf 

        print(f"Best Parameters: {best_params}")

        # --- PREDICTIONS ---
        y_pred_train = None 
        
        if model_type == 'MLP':
            # Prediction for AIF360
            test_df = X_test.copy()
            test_df['target'] = y_test.values
            
            # Re-detect attributes for consistency
            if adversarial_sensitive_attrs:
                protected_attrs = adversarial_sensitive_attrs
            else:
                protected_attrs = [col for col in X_test.columns if "Sex" in col or "Race" in col]

            aif_test = BinaryLabelDataset(
                df=test_df, label_names=['target'],
                protected_attribute_names=protected_attrs,
                favorable_label=1, unfavorable_label=0
            )
            y_pred = clf.predict(aif_test).labels.ravel()
            
            # For train accuracy
            train_df = X_train.copy()
            train_df['target'] = y_train.values
            aif_train = BinaryLabelDataset(
                df=train_df, label_names=['target'],
                protected_attribute_names=protected_attrs,
                favorable_label=1, unfavorable_label=0
            )
            y_pred_train = clf.predict(aif_train).labels.ravel()
            
        else:
            y_pred = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)

        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        
        # --- METRICS & PLOTS ---
        report = classification_report(y_test, y_pred, output_dict=True) 
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = None 
        if enable_plots:
            fig_cm, ax = plt.subplots(figsize=(4, 3)) 
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel('Predicted') 
            ax.set_ylabel('True') 
            plt.tight_layout()
        
        # --- PREDICTION FUNCTION WRAPPERS (Closure for Explainer) ---
        def predict_lime(x):
            if isinstance(x, np.ndarray):
                x_df = pd.DataFrame(x, columns=X_train.columns)
            else:
                x_df = x
            
            if model_type == "MLP":
                x_df['target'] = 0 
                # Re-use detection logic
                p_attrs = adversarial_sensitive_attrs if adversarial_sensitive_attrs else [c for c in x_df.columns if "Sex" in c or "Race" in c]
                
                aif_d = BinaryLabelDataset(
                    df=x_df, label_names=['target'], protected_attribute_names=p_attrs,
                    favorable_label=1, unfavorable_label=0
                )
                probs = clf.predict(aif_d).scores
                return np.column_stack([1 - probs, probs])
            else:
                return clf.predict_proba(x_df)

        def predict_shap(x):
            return predict_lime(x)

        # --- DICE WRAPPER ---
        dice_wrapper = DiceModelWrapper(
            model=self.model,
            label_encoders=self.label_encoders,
            model_type=self.model_type,
            session=self.session if model_type == 'MLP' else None,
            feature_names=X_train.columns.tolist()
        )

        results = {
            "model": clf,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "predict_lime": predict_lime,
            "predict_shap": predict_shap,
            "dice_model": dice_wrapper,
            "classification_report": report,
            "fig_cm": fig_cm,
            "accuracy": acc
        }

        return results