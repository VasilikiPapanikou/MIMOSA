import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append('../src') 
from data_loader import DataLoader
import datetime
import numpy as np
import plotly.express as px
import joblib
from train_classifier import ClassifierTrainer
from fairness_metrics import FairnessMetrics
import base64


from explainer import Explainer   
import importlib

st.set_page_config(layout="wide", page_title="Fairness Dashboard")
@st.cache_data
def get_img_as_base64(file):
    if not os.path.exists(file):
        return None
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache_data
def load_logo():
    return get_img_as_base64("media/logo.png")

 
def clear_selected_state():
    st.session_state.selected_state = None
    if 'active_tab' in st.session_state:
        st.session_state.active_tab = "Model Performance"

def shorten_group_name(name, default_label):
    """
    Use actual name only if <= 2 words. Otherwise return the default label.
    """
    if not isinstance(name, str):
        return default_label
    if len(name.split()) <= 2:
        return name
    return default_label

def display_fairness_metrics(
    fm ,
    focus_group_name,
    rest_group_name,
    metrics_to_display = ['PR', 'TPR', 'FPR']
):
    """
    Computes and displays fairness metrics (PR, TPR, FPR) in a 3-column layout.

    Args:
        fm: The object containing fairness computation methods (compute_metrics, perform_z_tests, metrics).
        focus_group_name: The name of the focus group (G1).
        rest_group_name: The name of the rest group (G2).
        metrics_to_display: List of metrics to show ('PR', 'TPR', 'FPR').
        title: The title for the section.
    """
    
 
    
    try:
        fm.compute_metrics()
        z_tests = fm.perform_z_tests()
        
        g1_metrics = fm.metrics[focus_group_name]
        g2_metrics = fm.metrics[rest_group_name]

        cols = st.columns(3)
        
        metric_definitions = {
            'PR': {
                'title': "Demographic Parity (PR)",
                'help': "The rate of positive predictions (e.g., 'loan approved') for each group. **Difference = (Focus Group - Rest)**",
                'g1_error': f"**Unfair:** Siginicant disparity in positive rates, disadvantaging the **{focus_group_name}** group.",
                'g2_error': f"**Unfair:** Siginicant disparity in positive rates, disadvantaging the **{rest_group_name}** group.",
                'success': "**Fair:** The difference is not statistically significant."
            },
            'TPR': {
                'title': "Equal Opportunity (TPR)",
                'help': "Measures how often each group correctly receives a positive outcome when they should. **Difference = (Focus Group - Rest)**",
                'g1_error': f"**Unfair:** Significant disparity in true positive rates, disadvantaging the **{focus_group_name}** group.",
                'g2_error': f"**Unfair:** Significant disparity in true positive rates, disadvantaging the **{rest_group_name}** group.",
                'success': "**Fair:** The difference is not statistically significant."
            },
            'FPR': {
                'title': "Predictive Equality (FPR)",
                'help': "Measures how often each group is wrongly given a positive outcome. **Difference = (Focus Group - Rest)**",
                'g1_error': f"**Unfair:** Significant disparity in false positive rates, disadvantaging the **{focus_group_name}** group.",
                'g2_error': f"**Unfair:** Significant disparity in false positive rates, disadvantaging the **{rest_group_name}** group.",
                'success': "**Fair:** The difference is not statistically significant."
            }
        }
        
        for i, metric_key in enumerate(metrics_to_display):
            if metric_key not in metric_definitions:
                continue

            with cols[i]:
                defs = metric_definitions[metric_key]
                g1_val = g1_metrics[metric_key]
                g2_val = g2_metrics[metric_key]
                metric_diff = g1_val - g2_val
                p_value = z_tests[metric_key][1]
                
                st.markdown(f"##### {defs['title']}", help=defs['help'])

                #short1 = "Focus Group"
                #short2 = "Rest of Population"

                fig = create_mini_bar_plot(g1_val, g2_val, focus_group_name, rest_group_name)
                st.pyplot(fig)
                
                st.markdown(
                    f"**Difference:** <span style='color:red;'>{metric_diff:+.3f}</span> (p-value: {format_p_value(p_value)})", 
                    unsafe_allow_html=True
                )
                
                if p_value < 0.05:
                    if metric_diff < 0:
                        st.error(defs['g1_error'])
                    else:
                        st.error(defs['g2_error'])
                else:
                    st.success(defs['success'])

        st.divider()

    except Exception as e:
        st.error(f"An error occurred during fairness metric display: {e}")
        st.code(e)

def format_p_value(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"

def display_fairness_delta(
    fm_base,
    fm_intervene,
    focus_group_name,
    rest_group_name,
    metrics_to_display=['PR', 'TPR', 'FPR']
):
    """
    Displays the change in disparity metrics between a base model and an intervened model, 
    separating the change text from the Streamlit metric for no arrow display.
    """
    
     
    st.markdown("### Change in Disparity (Intervention vs. Base Model)")
    st.info(f"This shows the change in disparity (Difference: {focus_group_name} - {rest_group_name}) after applying the intervention.")
    
    # Ensure all metrics are computed
    fm_base.compute_metrics()
    fm_intervene.compute_metrics()

    cols = st.columns(len(metrics_to_display))

    # Retrieve full metrics objects for clarity and the expander
    g1_metrics_base = fm_base.metrics[focus_group_name]
    g2_metrics_base = fm_base.metrics[rest_group_name]
    g1_metrics_intervene = fm_intervene.metrics[focus_group_name]
    g2_metrics_intervene = fm_intervene.metrics[rest_group_name]

    for i, metric_key in enumerate(metrics_to_display):
        with cols[i]:
            st.markdown(f"**{metric_key} Disparity Change**")
            
            # Base Model Difference (G1 - G2)
            diff_base = g1_metrics_base[metric_key] - g2_metrics_base[metric_key]
            
            # Intervened Model Difference (G1 - G2)
            diff_intervene = g1_metrics_intervene[metric_key] - g2_metrics_intervene[metric_key]
            
            # Improvement calculation (reduction in absolute difference)
            improvement = abs(diff_base) - abs(diff_intervene)
            
            delta_label = "Reduction in Disparity" if improvement >= 0 else "Increase in Disparity"
            color = "green" if improvement >= 0 else "red"

            # 1. Display Base Metric
            st.metric(
                label=f"Base Disparity (G1 - G2)",
                value=f"{diff_base:+.3f}"
            )
            # 2. Display Intervened Metric WITHOUT the delta parameter
            st.metric(
                label=f"Intervened Disparity (G1 - G2)",
                value=f"{diff_intervene:+.3f}",
            )
            
            # 3. Display the separate change information (no arrow)
            st.markdown(f"**Change:** <span style='color:{color}'>{abs(improvement):.3f} {delta_label}</span>", unsafe_allow_html=True)
            
            # 4. Display Individual Group Metric Details
            with st.expander("Metric Details"):
                st.markdown(f"**{focus_group_name} ({metric_key})**")
                st.markdown(f"Base: `{g1_metrics_base[metric_key]:.3f}` | Intervened: `{g1_metrics_intervene[metric_key]:.3f}`")
                st.markdown(f"**{rest_group_name} ({metric_key})**")
                st.markdown(f"Base: `{g2_metrics_base[metric_key]:.3f}` | Intervened: `{g2_metrics_intervene[metric_key]:.3f}`")
    
def get_focus_group_name(filters):
    str_parts = []
    num_parts = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple)):
            num_parts.append(f"{key} {value[0]}-{value[1]}")
        else:
            str_parts.append(str(value))
    parts = str_parts + num_parts
    if not parts:
        return "Selected Subgroup"
    return " ".join(parts)
            
def get_rest_group_name(filters, df):
    """
    Checks if the "Rest" group can be simply named.
    """
    default_name = "Rest of Population"
    
   
    if len(filters) != 1:
        return default_name 

    attr_name = list(filters.keys())[0]
    attr_value = list(filters.values())[0]

    
    if isinstance(attr_value, (list, tuple)):
        return default_name 
    
    
    unique_vals = df[attr_name].unique()
    if len(unique_vals) == 2:
        other_val = [val for val in unique_vals if val != attr_value]
        if len(other_val) == 1:
            return str(other_val[0]) # Returns "Male"

    
    return f"non-{attr_value}" # Returns "non-Black"

@st.cache_data(show_spinner="Downloading dataset...")
def load_dataset(name, state=None, year=None, protected_to_remove=[]):
    data_loader = DataLoader()
    

    try:
        if name == "Adult":
            data_initial, data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = data_loader.load_dataset(
                "../data/",year,protected_to_remove, datasetName="Adult", min_max_scale=True
            )
            target = "Target"
        
        elif name == "COMPAS":
            data_initial,data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = data_loader.load_dataset(
                "",year,protected_to_remove, datasetName="Compas", min_max_scale=True
            )
            target = "Target"
        
        elif name == "German Credit":
            data_initial,data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = data_loader.load_dataset(
                "../data/german.data",year,protected_to_remove, datasetName="GermanCredit", min_max_scale=True
            )
            target = "Target"
        
        elif name == "ACS Data" or state is not None:
             
            data_initial, data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = data_loader.load_dataset(
                "", year,protected_to_remove, datasetName="ACS Data", min_max_scale=True, state=state
            )
            target = "Target"  
                
        elif name == "ACS Data" and state is None:
       
            st.info("Please select a state on the map to load ACS Data.")
            
            return None, None, None, None, None, None, None, None, None, None
            
    except Exception as e:
        st.write("--------------------------------------------------")
        st.error(f"Error loading {name}: {e}")
        
        return None, None, None, None, None, None, None, None, None, None
    
    return data_initial, data_df, data_encoded,target, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders

if 'app_started' not in st.session_state:
    st.session_state.app_started = False

if 'selected_state' not in st.session_state:
    st.session_state.selected_state = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Model Performance"

@st.cache_resource(show_spinner="")
def train_and_predict_model(
    model_name,
    dataset_name,
    data_df,
    data_encoded,
    feature_names,
    categorical_features,
    _label_encoders, 
    removed_attributes=[],
    selected_state=None,
    selected_year=None, 
    debias=False,
    adversarial_sensitive_attrs=None  
):
    """
    Trains/loads the specified model and returns all necessary prediction results.
    Uses model_name and dataset_name (or state/year) for caching.
    """
    
    if debias:
        training_model_name = 'MLP'
        adv_suffix = "_adv"
    else:
        training_model_name = model_name
        adv_suffix = ""

   
    removal_suffix = "_".join(sorted(removed_attributes)) if removed_attributes else "FULL"
    
    if selected_state:
        MODEL_FILENAME = f"{training_model_name}_{selected_state}_{selected_year}_{removal_suffix}{adv_suffix}_best_model.joblib"
    else:
        MODEL_FILENAME = f"{training_model_name}_{dataset_name}_{removal_suffix}{adv_suffix}_best_model.joblib"
        
    MODEL_DIR = "saved_models"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

    
    trainer = ClassifierTrainer(
        dataset_name=dataset_name,
        feature_names=feature_names,
        categorical_features=categorical_features,
        label_encoders=_label_encoders, 
        privileged_groups=None,   
        unprivileged_groups=None, 
        debias=debias             
    )

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = joblib.load(MODEL_PATH)
        except Exception as e:
            pass
           

    display_name = "Adversarial Debiasing" if debias else training_model_name
    
    with st.spinner(f"Training {display_name}... This may take a moment."):
        results = trainer.train(
            data_df, 
            data_encoded, 
            model=model,
            model_type=training_model_name, 
            enable_plots=True,
            adversarial_sensitive_attrs=adversarial_sensitive_attrs 
        )
    
    if model is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            joblib.dump(results['model'], MODEL_PATH)
        except:
            print(f"Warning: Could not save model to disk.")
        
    return results

@st.cache_data(show_spinner=False)
def get_cached_explanations(
    _explainer, method, model_name, _model_input, data_subset, group_name, outcome_group, num_features=10
):
    
    
    os.makedirs("saved_explanations", exist_ok=True)
    
    file_name = f"saved_explanations/{method}_{model_name}_{group_name}_{outcome_group}_explanations.json"
    
    return _explainer.generate_explanations(
        method,
        data_subset,
        _model_input,
        save_path=file_name,
        num_features=num_features
    )

def create_mini_bar_plot(val1, val2, name1, name2):
    
    fig, ax = plt.subplots(figsize=(3, 1)) 

    name1_bar = "Focus Group"
    name2_bar = "Rest of\nPopulation"

    name1_bar = (name1[:15] + '..') if len(name1) > 17 else name1
    name2_bar = (name2[:15] + '..') if len(name2) > 17 else name2
    bars = ax.barh([name2_bar, name1_bar], [val2, val1], color=["#1B587C", "#F07F09"], height=0.6)
    ax.bar_label(bars, fmt='%.3f', padding=5)
    max_val = max(val1, val2, 0.01)
    ax.set_xlim(0, max_val * 1.3)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_xticks([]) 
    ax.tick_params(axis='y', length=0) 
    plt.tight_layout()
    return fig

def next_step():
    if st.session_state.current_step < 3:
        st.session_state.current_step += 1

def prev_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1

def update_dataset_choice():
    st.session_state.saved_dataset_name = st.session_state.sidebar_dataset
    st.session_state.saved_attributes_to_filter = []

def update_attributes_filter():
    st.session_state.saved_attributes_to_filter = st.session_state.attributes_widget_key

def save_widget_state(saved_key, widget_key):
    """
    General callback to sync a widget's value to a permanent session state variable.
    """
    st.session_state[saved_key] = st.session_state[widget_key]

def main():

    if not st.session_state.app_started:

        @st.cache_data
        def load_bg_image():
            return get_img_as_base64("media/mimosa.jpg")

        img_str = load_bg_image()
        image_path = "media/mimosa.jpg"

        if img_str:
            page_bg_img = f"""
            <style>
            
            [data-testid="stAppViewContainer"] {{
                background: 
                    linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
                    url("data:image/png;base64,{img_str}") no-repeat center center fixed;
                background-size: cover;
            }}

           
            .block-container {{
                padding-top: 0rem;
                padding-bottom: 0rem;
                min-height: 100vh;
            }}

           
            [data-testid="stSidebar"] {{
                display: none !important;
            }}

            [data-testid="stHeader"] {{
                background-color: rgba(0,0,0,0) !important;
            }}

            
            .splash-title {{
                font-size: 3.5rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #ffffff;
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);
                margin-bottom: 0.3rem;
                
                margin-top: 150px;
            }}

            .splash-subtitle {{
                font-size: 1.4rem;
                font-weight: 300;
                max-width: 670px;
                color: #f5f5f5;
                text-align: center;
                font-family: "Times New Roman", Times, serif;
                margin: 0 auto 2.0rem auto;
                line-height: 1.6;
                text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.7);
                
                margin-top: 2px;
            }}
            .splash-footnote {{
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 1.1rem;
                color: #f0f0f0;
                font-family: "Times New Roman", Times, serif;
                text-align: center;
                text-shadow: 1px 1px 5px rgba(0,0,0,0.6);
                opacity: 0.85;

                white-space: nowrap;   
                width: 100%;          
                overflow: hidden;      
            }}
            div.stButton {{
                display: flex;
                justify-content: center;
                margin-bottom: 50px;
                transform: translateX(-25px);
                width: 100%;
            }}

            .stButton button,
            [data-testid="stButton"] > button,
            [data-testid="stFormSubmitButton"] > button,
            button[kind="secondary"],
            button[kind="primary"],
            div.stButton > button {{
                font-family: "Times New Roman", Times, serif !important;
                font-size: 1.25rem !important;
                font-weight: 600 !important;
            }}
          
           
            .btn-center {{
                text-align: center;
                width: 100%;
                display: flex;
                justify-content: center;
            }}
        
            .btn-center {{
                display: flex;
                justify-content: center;
                width: 100%;
            }}
            div.stButton > button {{
                margin-top: -40px !important;  
            }}
        
            </style>
            """
        else:
            st.warning(f"Splash screen image not found at {image_path}. Please create this file.")
            page_bg_img = "" 

        st.markdown(page_bg_img, unsafe_allow_html=True)

        st.markdown('<div class="splash-title">MIMOSA</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="splash-subtitle">'
            'Use explanations to detect and understand bias in AI decision models.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="splash-footnote">*The mimosa flower symbolizes purity, innocence and sensitivity, values that are essential in the pursuit of truth, justice and fairness.*</div>', unsafe_allow_html=True)

         
        with st.container():
            st.markdown('<div class="btn-center">', unsafe_allow_html=True)

            if st.button("Start Bias Detection", key="start_btn"):
                st.session_state.app_started = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        #st.stop() 

    else:
        
        if 'selected_state' not in st.session_state:
            st.session_state.selected_state = None

        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1  
            
        if 'trainer' not in st.session_state:
            st.session_state.trainer = None
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = {}
        if 'fm' not in st.session_state:
            st.session_state.fm = None
        if 'explanations_per_group' not in st.session_state:
            st.session_state.explanations_per_group = None
        if 'base_model_results' not in st.session_state:
            st.session_state.base_model_results = {}
        
        
        st.markdown("""
            <style>
            
            [data-testid="stSidebar"] .stButton > button {
                background-color: transparent;
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 20px; 
                color: inherit;
                transition: all 0.3s ease;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                background-color: rgba(128, 128, 128, 0.1);
                border-color: rgba(128, 128, 128, 0.5);
                transform: scale(1.05);
            }
            
            [data-testid="stSidebar"] .stButton > button:disabled {
                border: none;
                color: rgba(128, 128, 128, 0.3);
            }
            </style>
        """, unsafe_allow_html=True)

    
        col_prev, col_logo, col_next = st.sidebar.columns([1, 4, 1])
        
        with col_prev:
            if st.session_state.current_step > 1:
                st.button("◀", on_click=prev_step, help="Go to Previous Step")
            else:
                st.button("◀", disabled=True)

        with col_logo:
            logo = get_img_as_base64("media/logo.png")
            if logo:
                st.markdown(
                    f"<div style='display:flex; justify-content:center; margin-top:8px;'>"
                    f"<img src='data:image/png;base64,{logo}' width='150px'>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        with col_next:
            if st.session_state.current_step < 3:
                st.button("▶", on_click=next_step, help="Go to Next Step")
            else:
                st.button("▶", disabled=True)

        steps = {1: "Training & Statistical Fairness", 2: "Explain", 3: "Fairness Interventions"}
        st.sidebar.markdown(
            f"<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 10px;'>"
            f"Step {st.session_state.current_step} of 3: <b>{steps[st.session_state.current_step]}</b>"
            f"</div>", 
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("---") 
        
        
        # ==============================================================
        # SCREEN 1: ANALYSIS & TRAINING
        # ==============================================================
        if st.session_state.current_step == 1:
            st.sidebar.header("Dataset & Model Configuration")
            if 'saved_dataset_name' not in st.session_state:
                st.session_state.saved_dataset_name = "Please select..."
             
            
            if 'saved_model_name' not in st.session_state:
                st.session_state.saved_model_name = "Please select..."
            if 'selected_model_name' not in st.session_state:
                st.session_state.selected_model_name = "Please select..."
            if 'saved_attributes_to_filter' not in st.session_state:
                st.session_state.saved_attributes_to_filter = []  # Default is an empty list
            if 'selected_value' not in st.session_state:
                st.session_state.selected_value = "Please select..."
            
            dataset_options = ["Please select...", "ACS Data", "Adult", "COMPAS", "German Credit", "🗂️ Load from file"]
            
            try:
                default_index = dataset_options.index(st.session_state.saved_dataset_name)
            except ValueError:
                default_index = 0

        
            dataset_name = st.sidebar.selectbox(
                "Select Dataset",
                dataset_options,
                index=default_index,         
                key="sidebar_dataset",       
                on_change=update_dataset_choice 
            )
            
            if dataset_name == "Please select...":
                
                st.markdown("""
                    ***This tool allows you to use explanations to detect and understand bias in AI decision models.***
                        
                """)    
                
                st.header("How to Get Started")
                st.markdown("""
                    1.  **Select a Dataset:** Use the `Select Dataset` dropdown in the sidebar to load a dataset. You can:
                        * Load a pre-defined dataset (like 'Adult' or 'COMPAS').
                        * Select 'ACS Data' from US Census surveys and choose a specific US state from the map.
                        * Upload your own CSV file.
                    2.  **Define a Focus Group:** In the `Define Focus Group` section (which appears in the sidebar after loading data), choose the attributes you want to investigate (e.g., 'Sex' = 'Female').
                    3.  **Train a Model:** Choose a model to train (e.g. 'Random Forest') from the `Select Model` dropdown or upload your own.
                    4.  **Analyze the Results:** After the model is ready, two main tabs will appear:
                        * **Model Performance:** Review the model's overall accuracy, classification report, and confusion matrix.
                        * **Fairness Analysis:**  
                            * **Measure:** Explore group fairness metrics (like Demographic Parity) to see *if* your focus group is being treated unfairly.
                            * **Select Outcome Group:** Choose an outcome to investigate (e.g., False Positives, True Positives).
                            * **Diagnose with Explanations:** Use individual and group-level explanations (LIME, SHAP) to understand *why* the model produces biased outcomes for that specific group.
                    
                """)
                st.info("Please select a dataset from the sidebar to begin.")
                st.stop()
            st.header("Training & Statistical Fairness")
            data_initial = None    
            data_encoded = None
            data_df = None
            target = None
            selected_year = None
            selected_state = None
            sensitive_attributes = []  
            focus_group_filters = {}

            if dataset_name == "ACS Data":

                st.subheader("ACS Data (American Community Survey)")
                st.markdown("""
                    You have selected the **American Community Survey (ACS)** dataset. This is a demographic survey
                    by the U.S. Census Bureau that provides vital, up-to-date information about the nation's people.
                    
                    The datasets available here are commonly used in fairness research to predict outcomes like **income level**.
                """)
                
            
                st.info("To begin, please **select a year** and then **click a state on the map below** to load its dataset.")
                current_year = datetime.datetime.now().year 
                year_list = list(range(current_year-2, 2013, -1)) 
                
                selected_year = st.selectbox(
                    "Select Year",
                    year_list, key = "acs_year_selector"
                )
                
                if 'started' not in st.session_state:
                    st.session_state.started = False
                
                state_data = {
                    'state': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
                    'value': [1]*50 
                }
                df = pd.DataFrame(state_data)

                fig = px.choropleth(
                    df,
                    locations='state',           
                    locationmode="USA-states",   
                    color='value',               
                    scope="usa",              
                    color_continuous_scale="Blues",  
                )
                
                fig.update_layout(
                    title_text='Click to select a state.',
                    geo=dict(
                        lakecolor='rgb(255, 255, 255)'  
                    ),
                    margin={"r":0,"t":50,"l":0,"b":0} ,
                    coloraxis_showscale=False
                )
                
                fig.update_traces(showscale=False)
                
                selection = st.plotly_chart(
                    fig, 
                    on_select="rerun", 
                    key="state_selection",
                    use_container_width=True
                )
                
                
                if selection and selection.selection and selection.selection['points']:
                    try:
                        
                        st.session_state.selected_state = selection.selection['points'][0]['location']
                    except (KeyError, IndexError):
                        st.warning("Could not retrieve state. Please try clicking again.")
                
                
                if st.session_state.selected_state:
                    selected_state = st.session_state.selected_state
                    st.success(f"You selected: **{selected_state}**")
                    
                    
                    data_initial, data_df, data_encoded, target, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = load_dataset(dataset_name, state=selected_state, year = selected_year)
                else:
                    
                    st.stop()

            elif dataset_name == "🗂️ Load from file":
                uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
                
                if uploaded_file:
                    try:
                        
                        if 'custom_df' not in st.session_state:
                            st.session_state.custom_df = pd.read_csv(uploaded_file)
                            st.session_state.custom_cols = ["Please select..."] + list(st.session_state.custom_df.columns)
                        
                        data_df = st.session_state.custom_df
                        
                        
                        
                        
                        data_df = data_df.copy()
                        for col in data_df.columns:
                            if data_df[col].dtype == 'object':
                                data_df[col] = data_df[col].astype('category').cat.codes

                    except Exception as e:
                        st.sidebar.error(f"Error reading file: {e}")
                        st.session_state.custom_df = None
                        st.session_state.custom_cols = []
                        st.stop()
                else:
                    st.info("Please upload a CSV file from the sidebar.")
                    st.stop()

            else:
                data_initial, data_df, data_encoded, target, class_names, feature_names, categorical_features, categorical_indices, categorical_names, label_encoders = load_dataset(dataset_name)
            
            st.subheader("Load Dataset & Model Training")
            focus_group_mask = pd.Series(True, index=data_df.index)
            focus_group_defined = False

            st.sidebar.subheader("Define Focus Group")
            st.session_state.data_loader["data_encoded"] = data_encoded
            st.session_state.data_loader["data_df"] = data_df
            st.session_state.data_loader["feature_names"] = feature_names
            st.session_state.data_loader["categorical_features"] = categorical_features
            st.session_state.data_loader["label_encoders"] = label_encoders
            st.session_state.data_loader["class_names"] = class_names
            st.session_state.data_loader["categorical_indices"] = categorical_indices
            st.session_state.data_loader["categorical_names"] = categorical_names
            st.session_state.data_loader["target"] = target
            st.session_state.data_loader["dataset_name"] = dataset_name
            st.session_state.data_loader["selected_state"] = selected_state
            st.session_state.data_loader["selected_year"] = selected_year
            
            all_columns = data_df.columns.tolist()
            if target in all_columns:
                all_columns.remove(target)

            attributes_to_filter = st.sidebar.multiselect(
                "Select attributes to filter",
                all_columns,
                default=st.session_state.saved_attributes_to_filter,   
                key="attributes_widget_key",                           
                on_change=update_attributes_filter                    
            )

            if attributes_to_filter:
                focus_group_defined = True
                
                for attr in attributes_to_filter:
                
                    
                    col_ui = data_initial[attr] 
                    
                    
                    if pd.api.types.is_numeric_dtype(col_ui):
                        
                        
                        is_int_col = False
                        if pd.api.types.is_integer_dtype(col_ui):
                            is_int_col = True
                        else:
                            try:
                                if col_ui.dropna().apply(lambda x: float(x).is_integer()).all():
                                    is_int_col = True
                            except:
                                pass

                        
                        if is_int_col:
                            min_val, max_val = int(col_ui.min()), int(col_ui.max())
                            step = 1  
                        else:
                            min_val, max_val = float(col_ui.min()), float(col_ui.max())
                            step = None  

                        saved_key = f"range_filter_{attr}"
                        widget_key = f"widget_range_filter_{attr}"
                        
                        if saved_key not in st.session_state:
                            st.session_state[saved_key] = (min_val, max_val)
                        
                        
                        current_min, current_max = st.session_state[saved_key]
                        if is_int_col:
                            current_val = (int(current_min), int(current_max))
                        else:
                            current_val = (float(current_min), float(current_max))

                        selected_range = st.sidebar.slider(
                            f"Range for {attr}",
                            min_value=min_val,
                            max_value=max_val,
                            value=current_val,
                            step=step,                  
                            key=widget_key,
                            on_change=save_widget_state,
                            args=(saved_key, widget_key)
                        )
                        
                        
                        focus_group_mask &= col_ui.between(selected_range[0], selected_range[1])
                        focus_group_filters[attr] = selected_range

                    
                    else:
                        unique_vals = sorted(col_ui.unique().astype(str)) 
                        
                        saved_key = f"focus_filter_{attr}"
                        widget_key = f"widget_focus_filter_{attr}"
                        
                        
                        if saved_key not in st.session_state:
                            st.session_state[saved_key] = unique_vals[0] if len(unique_vals) > 0 else None
                        
                        
                        try:
                            current_val = str(st.session_state[saved_key])
                            if current_val in unique_vals:
                                specific_index = unique_vals.index(current_val)
                            else:
                                specific_index = 0
                        except (ValueError, TypeError):
                            specific_index = 0

                        selected_val = st.sidebar.selectbox(
                            f"Value for {attr}",
                            unique_vals,
                            index=specific_index,
                            key=widget_key,
                            on_change=save_widget_state,
                            args=(saved_key, widget_key)
                        )
                        
                        focus_group_mask &= (col_ui.astype(str) == selected_val)
                        focus_group_filters[attr] = selected_val

                st.session_state.data_loader["focus_group_mask"] = focus_group_mask
                st.session_state.data_loader["focus_group_filters"] = focus_group_filters
                st.subheader(f"📊 Dataset: {dataset_name}")
            
            DATASET_DESCRIPTIONS = {
                "Adult": "The UCI Adult dataset is used to predict whether an individual's income exceeds $50K/yr based on census data.",
                "COMPAS": "The COMPAS dataset is used to predict recidivism risk for criminal defendants.",
                "German Credit": "The German Credit dataset is used to classify individuals as 'good' or 'bad' credit risks based on a set of attributes.",
                "ACS Data": f"A dataset from the {selected_year} American Community Survey (ACS) for **{selected_state}** state. It is used to predict the income.",
                "🗂️ Load from file": "A custom dataset uploaded by the user."
            }

            TARGET_NAME_MAP = {
                "Adult": "Income (<=50K or >50K)",
                "COMPAS": "Two-Year Recidivism",
                "German Credit": "Credit Risk (Good/Bad)",
                "ACS Data": "Income (<=50K or >50K)",
                "🗂️ Load from file": target
            }
            
            display_target_name = TARGET_NAME_MAP.get(dataset_name, target)
            st.markdown(DATASET_DESCRIPTIONS.get(dataset_name, "No description available."))
            continuous_features = [feature for feature in feature_names if feature not in categorical_features]
            with st.expander("View Dataset Details"):
                st.write(f"**Number of samples:** `{data_df.shape[0]}`")
                st.write(f"**Number of features:** `{data_df.shape[1]-1}`")
                st.write(f"**Target variable:** `{display_target_name}`")
                st.write(f"**Categorical Features:** `{', '.join(categorical_features)}`")
                st.write(f"**Numerical Features:**`{', '.join(continuous_features)}`")
                
            st.dataframe(data_initial.head(), hide_index=True)
            sns.set_context("paper", font_scale=0.8)

            group_order = ['Focus Group', 'Rest of Population']
            
            if focus_group_defined:
                focus_group_name = st.session_state.get('focus_group_name', 'Focus Group')
                rest_group_name = st.session_state.get('rest_group_name', 'Rest of Population')
                
                focus_plot_name = shorten_group_name(focus_group_name, "Focus Group")
                rest_plot_name  = shorten_group_name(rest_group_name, "Rest of Population")
                
                custom_colors = {focus_group_name: "#F07F09", rest_group_name: "#1B587C"}
            
                plot_df = data_initial.copy() 
                plot_df['focus_group'] = np.where(focus_group_mask, focus_plot_name, rest_plot_name)
                group_order_with_names = [focus_plot_name, rest_plot_name]
                
                plot_col1, plot_col2 = st.columns(2)
                
                with plot_col1:
                    st.subheader(f"👥 Selected group distribution")
                    fig, ax = plt.subplots(figsize=(3, 2))
                    
                    sns.countplot(
                        x='focus_group', 
                        data=plot_df, 
                        palette="pastel", 
                        ax=ax, 
                        order=group_order_with_names
                    )
                    
                    ax.set_ylabel("Count", fontsize=8)
                    ax.set_xlabel("Group", fontsize=8)
                    ax.tick_params(axis='x', labelrotation=0, labelsize=7) 
                    ax.tick_params(axis='y', labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)

                count_df = plot_df.groupby(['focus_group', target]).size().reset_index(name="count")
                total_df = plot_df.groupby('focus_group').size().reset_index(name="total")
                prop_df = pd.merge(count_df, total_df, on='focus_group')
                prop_df["proportion"] = prop_df["count"] / prop_df["total"]

                with plot_col2: 
                    st.subheader(f"🎯 Target distribution by selected group")
                    fig, ax = plt.subplots(figsize=(3, 2)) 
                    sns.barplot(
                        data=prop_df,
                        x='focus_group', 
                        y="proportion", 
                        hue=target,
                        palette="Set2", 
                        ax=ax,
                        order=group_order_with_names
                    )
                    ax.set_ylabel("Proportion", fontsize=8)
                    ax.set_xlabel("Group", fontsize=8)
                    ax.tick_params(axis='x', labelrotation=0, labelsize=7)
                    ax.tick_params(axis='y', labelsize=7)
                    ax.legend(title=target, fontsize=5, title_fontsize=6, loc='upper right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
            else:
                st.info("Select attributes from the 'Define Focus Group' to define your subgroup of interest.")

            model_options = ["Please select...", "Logistic Regression", "Random Forest", "XGBoost", "MLP"]
            try:
                model_index = model_options.index(st.session_state.saved_model_name)
            except ValueError:
                model_index = 0

      
            model_name = st.sidebar.selectbox(
                "Select Model",
                model_options,
                index=model_index,
                key="widget_model_name",           
                on_change=save_widget_state,       
                args=("saved_model_name", "widget_model_name")   
            )
            st.session_state.data_loader["model_name"] = model_name
            rf_model, dice_model, predict_lime, predict_shap = None, None, None, None
            X_test, y_test, y_pred, best_split = None, None, None, None
            
            if 'model_trained' not in st.session_state:
                st.session_state.model_trained = False
            if 'fairness_calculated' not in st.session_state:
                st.session_state.fairness_calculated = False

            if dataset_name == "ACS Data":
                dataset_name = selected_state

            if model_name != "Please select...":
                train_btn = st.sidebar.button("Train Model", key="train_model_btn")
                calc_fairness_btn = False
                if st.session_state.model_trained and focus_group_defined:
                    calc_fairness_btn = st.sidebar.button("Calculate Fairness Metrics", key="calc_fairness_btn")

                if train_btn:
                    with st.spinner(f"Training {model_name} model..."):


                        st.info(f"Preparing **{model_name}** model...")
                        results = train_and_predict_model(
                            model_name=model_name,
                            dataset_name=dataset_name,  
                            data_df=data_df, 
                            data_encoded=data_encoded, 
                            feature_names=feature_names, 
                            categorical_features=categorical_features, 
                            _label_encoders=label_encoders,
                            
                            removed_attributes=[], 
                            selected_state=selected_state,
                            selected_year=selected_year
                        )
                
                        st.session_state.base_model_results = results
                        st.session_state.model_name = model_name
                        st.session_state.model_trained = True
                        st.session_state.fairness_calculated = False  
                        st.rerun()
                if calc_fairness_btn:
                    st.session_state.fairness_calculated = True
                     

                if st.session_state.model_trained and st.session_state.base_model_results is not None:
                    results = st.session_state.base_model_results

                    rf_model = results["model"]
                    dice_model = results["dice_model"]
                    X_test = results["X_test"]
                    y_test = results["y_test"]
                    y_pred = results["y_pred"]

                    st.subheader(f"📈 Model performance for: {model_name}")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.metric(label="Model Accuracy", value=f"{results['accuracy']:.2%}")
                        
                        st.subheader("Confusion Matrix")
                        st.pyplot(results["fig_cm"], use_container_width=True) 

                
                    st.divider()
                
                if st.session_state.fairness_calculated:
                    
                    if not focus_group_defined:
                        st.warning("⚠️ Focus Group definition changed or lost. Please redefine in sidebar.")
                        st.session_state.fairness_calculated = False
                        st.stop()    
                    if calc_fairness_btn:
                        st.session_state.fairness_calculated = True
                        
                     
                        focus_group_name = get_focus_group_name(focus_group_filters)
                        rest_group_name = get_rest_group_name(focus_group_filters, data_df)

                        st.session_state['focus_group_name'] = focus_group_name   
                        st.session_state['rest_group_name'] = rest_group_name  

                        test_indices = X_test.index
                        test_focus_mask = focus_group_mask.reindex(test_indices, fill_value=False)
                        X_test_with_focus = X_test.copy()
                        X_test_with_focus['focus_group'] = np.where(test_focus_mask, 'Focus Group', 'Rest of Population')

                        if test_focus_mask.sum() == 0:
                            st.error(f"⚠️ The group '{focus_group_name}' is empty in the Test Set.")
                            st.warning("Please select a broader group.")
                            st.stop()
                        try:
                            with st.spinner("Calculating fairness metrics..."):
                                fm = FairnessMetrics(
                                    X_test=X_test_with_focus,
                                    y_test=y_test,
                                    y_pred=y_pred,
                                    sensitive_attr='focus_group',
                                    attr_values=['Focus Group', 'Rest of Population'],
                                    attr_names=[focus_group_name, rest_group_name],  
                                    output_dir="fairness_reports"
                                )
                                st.session_state.fm = fm
                        except Exception as e:
                            st.error(f"Error instantiating FairnessMetrics: {e}")
                            st.stop()
                
                        help_text = (
                            "**Group Fairness** assumes that individuals are partitioned into groups based on the value of one"
                            " or more protected attributes and requires that these groups are treated similarly by the model."
                            "**Group Fairness Metrics** measure statistical disparities among groups in model outcomes "
                            "(e.g., Demographic Parity, True Positive Rate, False Positive Rate) between "
                            "groups. These differences reflect potential bias in the model's predictions."
                        )
                
                        st.markdown("### Statistical Group Fairness Metrics", help=help_text)

                
                        display_fairness_metrics(
                            fm=fm,
                            focus_group_name=focus_group_name,
                            rest_group_name=rest_group_name
                        )
                    elif not st.session_state.fairness_calculated and focus_group_defined:
                        st.info("👈 Click **'Calculate Fairness Metrics'** in the sidebar to analyze fairness.")
                

        # ==============================================================
        # SCREEN 2: EXPLANATIONS
        # ==============================================================
        elif st.session_state.current_step == 2:
            st.title("Explain")
            st.write("Diagnose *why* the model produces certain outcomes for your focus group.")

            if (not st.session_state.data_loader 
                or "data_df" not in st.session_state.data_loader):
                st.warning("⚠️ No dataset loaded. Please return to Step 1.")
                st.stop()
            if ("model" not in st.session_state.base_model_results
                or st.session_state.base_model_results["model"] is None):
                st.warning("⚠️ No model trained. Please return to Step 1 and train a model first.")
                st.stop()
            if 'saved_explanation_type' not in st.session_state:
                st.session_state.saved_explanation_type = "Please select..."
            if 'saved_explanation_method' not in st.session_state:
                st.session_state.saved_explanation_method = "Please select..."
            if 'saved_outcome_group' not in st.session_state:
                st.session_state.saved_outcome_group = "Please select..."
            if 'saved_viz_type' not in st.session_state:
                st.session_state.saved_viz_type = "Please select..."
            
            if 'saved_dice_viz_type' not in st.session_state:
                st.session_state.saved_dice_viz_type = "Please select..."

            if 'saved_explanation_method' not in st.session_state:
                st.session_state.saved_explanation_method = "Please select..."
            st.sidebar.header("Explanation Settings")

            explanation_type_options = ["Please select...", "Individual", "Group"]
            try:
                exp_type_index = explanation_type_options.index(st.session_state.saved_explanation_type)
            except ValueError:
                exp_type_index = 0 
            exp_type = st.sidebar.selectbox(
                "Select Explanation Type",
                explanation_type_options,
                index=exp_type_index,
                key="widget_exp_type_name",          
                on_change=save_widget_state,       
                args=("saved_explanation_type", "widget_exp_type_name")   
            )
            if st.session_state.saved_explanation_type == "Individual":
                method_options = ["Please select...", "LIME", "SHAP", "DiCE", "Best"]
            else: 
                method_options = ["Please select...", "FACEGroup"]
            
            try:
                method_index = method_options.index(st.session_state.saved_explanation_method)
            except ValueError:
                method_index = 0
                st.session_state.saved_explanation_method = "Please select..."

            exp_method = st.sidebar.selectbox(
                "Explanation Method", 
                method_options,
                index=method_index,
                key="widget_exp_method",
                on_change=save_widget_state,
                args=("saved_explanation_method", "widget_exp_method")
            )

            if st.session_state.saved_explanation_type == "Please select...":
                st.info("Please select an explanation type above to generate explanations.")
                st.stop()
          
            outcome_options = [
                    "Please select...",  
                    "All Predictions", 
                    "All Positives (P)", 
                    "All Negatives (N)",
                    "True Positives (TP)", 
                    "False Positives (FP)", 
                    "True Negatives (TN)", 
                    "False Negatives (FN)"
                ]
             
            try:
                outcome_index = outcome_options.index(st.session_state.saved_outcome_group)
            except ValueError:
                outcome_index = 0
            outcome_group = st.sidebar.selectbox(
                "Analyze explanations for which group?", 
                outcome_options,
                index=outcome_index,
                key="widget_outcome_group",
                on_change=save_widget_state,
                args=("saved_outcome_group", "widget_outcome_group")
            )

            if st.session_state.saved_outcome_group == "Please select...":
                st.info("Please select an outcome group above to generate explanations.")
                st.stop()

            y_pred = st.session_state.base_model_results["y_pred"]
            X_test = st.session_state.base_model_results["X_test"]
            y_test = st.session_state.base_model_results["y_test"]
            predict_lime = st.session_state.base_model_results["predict_lime"]
            predict_shap = st.session_state.base_model_results["predict_shap"]
            dice_model = st.session_state.base_model_results["dice_model"]
            data_encoded = st.session_state.data_loader["data_encoded"]
            data_df = st.session_state.data_loader["data_df"]
            class_names = st.session_state.data_loader["class_names"]
            feature_names = st.session_state.data_loader["feature_names"]
            categorical_names = st.session_state.data_loader["categorical_names"]
            categorical_indices = st.session_state.data_loader["categorical_indices"]
            categorical_features = st.session_state.data_loader["categorical_features"]
            continuous_features = [feature for feature in feature_names if feature not in categorical_features]
            focus_group_mask = st.session_state.data_loader["focus_group_mask"]
            focus_group_name = st.session_state.get('focus_group_name', 'Focus Group')
            rest_group_name = st.session_state.get('rest_group_name', 'Rest of Population')

            focus_plot_name = shorten_group_name(focus_group_name, "Focus Group")
            rest_plot_name  = shorten_group_name(rest_group_name, "Rest of Population")
            is_positive_pred = (y_pred == 1)
            is_negative_pred = (y_pred == 0)
            is_true_positive = (y_test == 1) & (y_pred == 1)
            is_false_positive = (y_test == 0) & (y_pred == 1)
            is_true_negative = (y_test == 0) & (y_pred == 0)
            is_false_negative = (y_test == 1) & (y_pred == 0)

            X_test_outcome_subset = X_test.copy()
            
            if outcome_group == "All Positives (P)":
                X_test_outcome_subset = X_test[is_positive_pred]
            elif outcome_group == "All Negatives (N)":
                X_test_outcome_subset = X_test[is_negative_pred]
            elif outcome_group == "True Positives (TP)":
                X_test_outcome_subset = X_test[is_true_positive]
            elif outcome_group == "False Positives (FP)":
                X_test_outcome_subset = X_test[is_false_positive]
            elif outcome_group == "True Negatives (TN)":
                X_test_outcome_subset = X_test[is_true_negative]
            elif outcome_group == "False Negatives (FN)":
                X_test_outcome_subset = X_test[is_false_negative]

            test_indices = X_test_outcome_subset.index
            focus_mask_on_subset = focus_group_mask.reindex(test_indices, fill_value=False)
            
            X_test_focus_group = X_test_outcome_subset[focus_mask_on_subset]
            X_test_rest_group = X_test_outcome_subset[~focus_mask_on_subset]

            n_focus = len(X_test_focus_group)
            n_rest = len(X_test_rest_group)
            max_possible = max(n_focus, n_rest)

            if max_possible == 0:
                st.warning("No instances found in the selected outcome group.")
                st.stop()

            st.sidebar.markdown("---")
            st.sidebar.markdown("##### ⏱️ Sampling")
            st.sidebar.write("Some explanation methods can be slow on large datasets. Select a sample to save time.")
            
            default_limit = max_possible

            user_sample_limit = st.sidebar.slider(
                "Max instances per group:",
                min_value=5,
                max_value=max_possible if max_possible > 5 else 10,
                value=default_limit,
                step=5,
                key="user_sample_limit_slider_step2" 
            )
       
            if len(X_test_focus_group) > user_sample_limit:                    
                X_test_focus_group_sampled = X_test_focus_group.sample(user_sample_limit, random_state=42)
            else:
                X_test_focus_group_sampled = X_test_focus_group

            if len(X_test_rest_group) > user_sample_limit:                   
                X_test_rest_group_sampled = X_test_rest_group.sample(user_sample_limit, random_state=42)
            else:               
                X_test_rest_group_sampled = X_test_rest_group

            if X_test_focus_group_sampled.empty and X_test_rest_group_sampled.empty:            
                st.warning("Selected samples resulted in empty dataframes.")
                st.stop()

            if "explanations_generated_step2" not in st.session_state:
                st.session_state.explanations_generated_step2 = False

            if st.sidebar.button("Generate Explanations", key="gen_exp_btn_step2"):
                st.session_state.explanations_generated_step2 = True

            if not st.session_state.explanations_generated_step2:
                st.stop()
           
            data_df_no_focus = data_df.drop(columns=["focus_group"], errors="ignore")
            try:
                explainer_obj = Explainer(
                    data_df=data_df_no_focus,
                    data_encoded=data_encoded,
                    class_names=class_names,
                    feature_names=feature_names,
                    categorical_names=categorical_names,
                    categorical_indices=categorical_indices,
                    continuous_features=continuous_features
                )
            except Exception as e:
                st.error(f"Error initializing Explainer: {e}")
                st.stop()

            if exp_method == "LIME":
                model_input = predict_lime
            elif exp_method == "SHAP":
                model_input = predict_shap  
            elif exp_method == "DiCE":
                model_input = dice_model     
            
            viz_type = "Please select..."  
            current_method = st.session_state.saved_explanation_method
           
            
            model_name = st.session_state.saved_model_name


            st.markdown("---")
            st.markdown(f"### Generating **{exp_method}** Explanations")
            if st.session_state.explanations_generated_step2 == True:
                
                if exp_method == "LIME" or exp_method == "SHAP":
                    st.markdown("---")
                    st.subheader(f"Processing: {focus_group_name}")
                    focus_explanations = get_cached_explanations(
                        explainer_obj,           
                        exp_method,   
                        model_name,        
                        model_input,            
                        X_test_focus_group_sampled,
                        focus_group_name,
                        outcome_group
                    )
                        
                    st.markdown("---")
                    st.subheader(f"Processing: {rest_group_name}")
                    rest_explanations = get_cached_explanations(
                        explainer_obj,          
                        exp_method,   
                        model_name,          
                        model_input,             
                        X_test_rest_group_sampled,
                        rest_group_name,
                        outcome_group
                    )
                        
                    st.divider()
                    st.success(f"✅ {exp_method} Explanations ready! Choose a visualization below.")
                

                 
                    aggregation_choices = [
                        "Please select...",
                        "Distributions (Beeswarm Plot)", 
                        "Distributions (Violin Plot)", 
                        "Instance Heatmap",
                        "Mean Aggregation", 
                        "Mean Absolute Aggregation"
                    ]

                    try:
                        viz_index = aggregation_choices.index(st.session_state.saved_viz_type)
                    except ValueError:
                        viz_index = 0
                    
                    if st.session_state.saved_explanation_type == "Individual":
                        viz_type = st.sidebar.selectbox(
                            "Select Visualizations", 
                            aggregation_choices,
                            index=viz_index,
                            key="widget_viz_type",
                            on_change=save_widget_state,
                            args=("saved_viz_type", "widget_viz_type")
                        )

                    if st.session_state.saved_viz_type == "Please select...":
                        st.info("Please select a visualization to visualize explanations.")
                        st.stop()

                    custom_colors = {focus_group_name: "#F07F09", rest_group_name: "#1B587C"}
                    group_names = [focus_group_name, rest_group_name]
                    explanations_per_group = {
                        focus_group_name: focus_explanations,
                        rest_group_name: rest_explanations
                    }
                    if explanations_per_group:
                        if 'base_explanations' not in st.session_state:
                            st.session_state['base_explanations'] = {}
                            
                        st.session_state['base_explanations'][exp_method] = {
                            'explainer_obj': explainer_obj, 
                            'explanations': explanations_per_group,
                            'focus_group_name': focus_group_name,
                            'rest_group_name': rest_group_name,
                            'outcome_group': outcome_group,
                            'model_name': 'Base Model',
                        }
                    
                    top_features = explainer_obj.get_global_feature_order(
                        explanations_per_group, group_names, top_n=10
                    )
                    data_subsets = {
                        focus_group_name: X_test_focus_group_sampled,  
                        rest_group_name: X_test_rest_group_sampled
                    }
                    fig = None
                    if viz_type == "Distributions (Violin Plot)":
                        st.subheader("Feature Contribution Distributions")
                        
                        fig = explainer_obj.plot_violin_distributions(
                            explanations_per_group, group_names, custom_colors, feature_order=top_features
                        )
                        
                        if fig:
                            st.pyplot(fig, use_container_width=False)
                            st.markdown(
                                "### 📏 Distribution Differences", 
                                help="Distribution distance among groups usings Wasserstein distance. Larger values indicate a greater disparity between the two groups for that feature. Red number: Highest disparity."
                            )
                            
                            
                            diff_df = explainer_obj.compute_wasserstein_table(
                                explanations_per_group, 
                                group_names, 
                                feature_order=top_features
                            )
                            if not diff_df.empty:
                                max_val = diff_df["Wasserstein Difference"].max()
                                font_size = "18px" 
                                markdown_lines = []
                                
                                for feature_name, row in diff_df.iterrows():
                                    val = row["Wasserstein Difference"]
                                    if val == max_val:
                                        val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{val:.3f}</span>'
                                    else:
                                        val_str = f"{val:.3f}"
                                    
                                    line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}</span>'
                                    
                                    markdown_lines.append(line)
                    
                                st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                            
                    elif viz_type == "Distributions (Beeswarm Plot)":
                        st.subheader("Feature Contribution Distributions")
                        fig = explainer_obj.plot_beeswarm_comparison(
                            explanations_per_group, group_names, custom_colors, feature_order=top_features, data_per_group=data_subsets
                        )

                        if fig:
                            st.pyplot(fig, use_container_width=False)
                            st.markdown(
                                "### 📏 Distribution Differences", 
                                help="Distribution distance among groups usings Wasserstein distance. Larger values indicate a greater disparity between the two groups for that feature. Red number: Highest disparity."
                            )
                            
                            
                            diff_df = explainer_obj.compute_wasserstein_table(
                                explanations_per_group, 
                                group_names, 
                                feature_order=top_features
                            )
                            if not diff_df.empty:
                                max_val = diff_df["Wasserstein Difference"].max()
                                font_size = "18px" 
                                markdown_lines = []
                                
                                for feature_name, row in diff_df.iterrows():
                                    val = row["Wasserstein Difference"]
                                    if val == max_val:
                                        val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{val:.3f}</span>'
                                    else:
                                        val_str = f"{val:.3f}"
                                    
                                    line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}</span>'
                                    
                                    markdown_lines.append(line)
                    
                                st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                    
                    elif viz_type == "Instance Heatmap":
                        st.subheader("Feature Contribution Heatmap")
                        n_instances = st.slider("Number of Instances", 5, 50, 10)
                        fig = explainer_obj.plot_instance_heatmap(
                            explanations_per_group, 
                            [focus_group_name, rest_group_name], 
                            n_instances=n_instances,
                            feature_order=top_features
                        )
                        
                    
                        if fig:
                            st.pyplot(fig, use_container_width=False)
                        else:
                            st.warning("Could not generate plot. Data might be empty.")
                    
                    elif viz_type == "Mean Aggregation":
                        st.subheader("Mean Feature Contributions")
                        fig = explainer_obj.plot_mean_aggregation(
                            explanations_per_group, group_names, custom_colors, feature_order=top_features
                        )
                        if fig:
                            st.pyplot(fig, use_container_width=False)
                            
                            
                            st.markdown("#### 📏 Differences in Mean Conribution", help="Absolute difference between the average contribution of the two groups.")
                            diff_df = explainer_obj.compute_mean_diff_table(explanations_per_group, group_names, feature_order=top_features)
                            
                            if not diff_df.empty:
                                max_val = diff_df["Difference"].max()
                                font_size = "18px"
                                markdown_lines = []
                                for feature_name, row in diff_df.iterrows():
                                    val = row["Difference"]
                                    val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{val:.3f}</span>' if val == max_val else f"{val:.3f}"
                                    markdown_lines.append(f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}</span>')
                                st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        else:
                            st.warning("Could not generate plot. Data might be empty.")

                    elif viz_type == "Mean Absolute Aggregation":
                        st.subheader("Mean Absolute Feature Importance")
                        fig = explainer_obj.plot_mean_abs_aggregation(
                            explanations_per_group, group_names, custom_colors, feature_order=top_features
                        )
                        if fig:
                            st.pyplot(fig, use_container_width=False)

                            
                            st.markdown("#### 📏  Differences in Absolute Features Importance", help="Difference in the average magnitude (importance) of the feature.")
                            diff_df = explainer_obj.compute_mean_abs_diff_table(explanations_per_group, group_names, feature_order=top_features)
                            
                            if not diff_df.empty:
                                max_val = diff_df["Difference"].max()
                                font_size = "18px"
                                markdown_lines = []
                                for feature_name, row in diff_df.iterrows():
                                    val = row["Difference"]
                                    val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{val:.3f}</span>' if val == max_val else f"{val:.3f}"
                                    markdown_lines.append(f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}</span>')
                                st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        else:
                            st.warning("Could not generate plot. Data might be empty.")
                
                elif exp_method == "DiCE":
                    focus_indices = X_test_focus_group_sampled.index
                    rest_indices = X_test_rest_group_sampled.index
                    
                        
                    df_focus_original = data_df.loc[focus_indices].drop(columns=["focus_group"], errors="ignore")
                    
                    df_rest_original = data_df.loc[rest_indices].drop(columns=["focus_group"], errors="ignore")

                    st.markdown("---")
                   
                    st.subheader(f"Processing: {focus_group_name}")
                    focus_explanations = get_cached_explanations(
                        explainer_obj, "DiCE",model_name, dice_model, 
                        df_focus_original,  
                        focus_group_name, outcome_group
                    )
                    
                    st.markdown("---")
                    st.subheader(f"Processing: {rest_group_name}")
                    rest_explanations = get_cached_explanations(
                        explainer_obj, "DiCE",model_name, dice_model, 
                        df_rest_original,  
                        rest_group_name, outcome_group
                    )

                    st.divider()
                    st.success("✅ DiCE Explanations ready!")
                    viz_dice_type = st.session_state.get("saved_dice_viz_type", "Please select...")

                    viz_choices = [
                        "Please select...",
                        "Beeswarm Plot (Magnitude of Changes)", 
                        "Bar Plot (Percentages of Feature Change)"
                    ]

                    try:
                        viz_dice_index = viz_choices.index(st.session_state.saved_dice_viz_type)
                    except ValueError:
                        viz_dice_index = 0
                    
                    if st.session_state.saved_explanation_type == "Individual":
                        viz_dice_type = st.sidebar.selectbox(
                            "Select Visualizations", 
                            viz_choices,
                            index=viz_dice_index,
                            key="widget_dice_viz_type",
                            on_change=save_widget_state,
                            args=("saved_dice_viz_type", "widget_dice_viz_type")
                        )

                    if st.session_state.saved_dice_viz_type  == "Please select...":
                        st.info("Please select a visualization to visualize explanations.")
                        st.stop()

                    
                    
                    custom_colors = {focus_group_name: "#F07F09", rest_group_name: "#1B587C"}
                    group_names = [focus_group_name, rest_group_name]
                    
                    explanations_per_group = {
                        focus_group_name: focus_explanations,
                        rest_group_name: rest_explanations
                    }
                    
                    if 'base_explanations' not in st.session_state:
                        st.session_state['base_explanations'] = {}
                        
                    st.session_state['base_explanations']['DiCE'] = {
                        'explainer_obj': explainer_obj, 
                        'explanations': explanations_per_group,
                        'focus_group_name': focus_group_name,
                        'rest_group_name': rest_group_name,
                        'outcome_group': outcome_group,
                        'model_name': 'Base Model',
                        'df_focus_original': df_focus_original,  
                        'df_rest_original': df_rest_original,    
                    }
                  
                    data_subsets = {
                        focus_group_name: df_focus_original, 
                        rest_group_name: df_rest_original
                    }
                    top_features = explainer_obj.get_global_feature_order_dice(explanations_per_group, group_names,data_subsets, top_n=10)

                    df_changes = explainer_obj.compute_counterfactual_difference(
                        explanations_per_group, 
                        group_names,
                        data_subsets
                    )
                    
                    if focus_group_name in df_changes.columns and rest_group_name in df_changes.columns:
                        df_changes['Change_Diff'] = df_changes[focus_group_name] - df_changes[rest_group_name]
                    else:
                        df_changes['Change_Diff'] = 0
                    # ----------------------------------------------------
                    if viz_dice_type == "Beeswarm Plot (Magnitude of Changes)":
                        st.subheader("Feature Contribution Distributions")

                        fig = explainer_obj.plot_dice_beeswarm(explanations_per_group, group_names, custom_colors, data_subsets, feature_order = top_features)
                    elif viz_dice_type == "Bar Plot (Percentages of Feature Change)":
                        st.subheader("Counterfactual Feature Changes")
                        st.write("This plot shows **how often** has a feature has to change to flip the prediction.")

                     
                        fig = explainer_obj.plot_dice_comparison(
                            df_changes, group_names, custom_colors, feature_order=top_features
                        )
                    
                    if fig:
                        st.pyplot(fig, use_container_width=False)
                    
                        st.markdown(
                            "#### 📏 Frequency Differences", 
                            help="Difference in percentage of times a feature changed to flip prediction."
                        )
                        
                        if not df_changes.empty:

                            display_df = df_changes.copy()
                            display_df['Abs_Diff'] = display_df['Change_Diff'].abs()
                            
                            max_val = display_df['Abs_Diff'].max()
                            font_size = "18px"
                            markdown_lines = []
                            
                            display_df = display_df.sort_values('Abs_Diff', ascending=False)
                            
                            for feature_name, row in display_df.iterrows():
                                val = row['Abs_Diff']
                                
                                # Highlight Max
                                if val == max_val:
                                    val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{val:.1f}%</span>'
                                else:
                                    val_str = f"{val:.1f}%"
                                
                                line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}</span>'
                                markdown_lines.append(line)
                            
                            st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                    else:
                        st.warning("No feature changes found.")

        # ==============================================================
        # SCREEN 3: FAIRNESS INTERVENTIONS
        # ==============================================================
        elif st.session_state.current_step == 3:
            st.title("Fairness Interventions")
            
            if st.session_state.data_loader is None:
                st.warning("⚠️ No dataset loaded. Please go to 'Step 1'.")
                st.stop()
            if st.session_state.base_model_results is None:
                st.warning("⚠️ No model trained. Please go back to 'Step 1' and train a model first in order to compare it with the intervened one.")
            
             
            st.sidebar.header("Fairness Intervention Settings")
            
            X_test_base = st.session_state.base_model_results["X_test"]
            y_test_base = st.session_state.base_model_results["y_test"]
            y_pred_base = st.session_state.base_model_results["y_pred"]
            feature_names = st.session_state.data_loader["feature_names"]
            selected_state = st.session_state.data_loader["selected_state"]
            selected_year = st.session_state.data_loader["selected_year"]
            model_name = st.session_state.data_loader["model_name"] 
            dataset_name = st.session_state.data_loader["dataset_name"]
             
            if 'intervention_trained' not in st.session_state:
                st.session_state.intervention_trained = False
            if 'int_fairness_calculated' not in st.session_state:
                st.session_state.int_fairness_calculated = False

            if 'saved_intervention_type' not in st.session_state:
                st.session_state.saved_intervention_type = "Please select..."
            
            intervention_opts = ["Please select...", "Remove protected attribute"]

            if model_name == "MLP":
                intervention_opts.append("Adversarial Debiasing")

            try:
                int_type_idx = intervention_opts.index(st.session_state.saved_intervention_type)
            except ValueError:
                int_type_idx = 0

            intervention_type = st.sidebar.selectbox(
                "Select Method", 
                intervention_opts,
                index=int_type_idx,
                key="widget_intervention_type",
                on_change=save_widget_state,
                args=("saved_intervention_type", "widget_intervention_type")
            )
            if st.session_state.saved_intervention_type == "Please select...":
                st.info("Please select a **Fairness Intervention Type** from the sidebar to proceed.")
                st.stop()

             
            if 'saved_protected_attribs' not in st.session_state:
                st.session_state.saved_protected_attribs = []

            cols = feature_names
            
            protected_attributes = st.sidebar.multiselect(
                "Select one or more attributes to remove:",
                options=cols,
                default=st.session_state.saved_protected_attribs, 
                key="widget_protected_attribs_multiselect",
                on_change=save_widget_state,
                args=("saved_protected_attribs", "widget_protected_attribs_multiselect")
            )

            if not protected_attributes:
                st.info("Please select at least one attribute to remove before proceeding with the analysis.")
                st.stop()
            apply_int_btn = st.sidebar.button("Apply Intervention")

            if apply_int_btn:
                st.session_state['intervention_applied'] = True
                with st.spinner(f"Applying {intervention_type}..."):
                    cols_to_drop_on_load = protected_attributes if intervention_type == "Remove protected attribute" else []
                    if selected_state is not None:       
                        data_initial_intervene, data_df_intervene, data_encoded_intervene, target_intervene, class_names_intervene, feature_names_intervene, categorical_features_intervene, categorical_indices_intervene, categorical_names_intervene, label_encoders_intervene = load_dataset(
                            dataset_name, state=selected_state, year=selected_year, protected_to_remove=cols_to_drop_on_load
                        )
                    else:          
                        data_initial_intervene, data_df_intervene, data_encoded_intervene, target_intervene, class_names_intervene, feature_names_intervene, categorical_features_intervene, categorical_indices_intervene, categorical_names_intervene, label_encoders_intervene = load_dataset(
                            dataset_name, protected_to_remove=cols_to_drop_on_load
                        )
                    intervention_results = None
                    if intervention_type == "Remove protected attribute":
                        intervention_results = train_and_predict_model(
                            model_name=model_name,
                            dataset_name=dataset_name,  
                            data_df=data_df_intervene,  
                            data_encoded=data_encoded_intervene, 
                            feature_names=feature_names_intervene, 
                            categorical_features=categorical_features_intervene, 
                            _label_encoders=label_encoders_intervene,
                            removed_attributes=protected_attributes,
                            selected_state=selected_state,
                            selected_year=selected_year
                        )
                    elif intervention_type == "Adversarial Debiasing":
                        
                        st.info("Running Adversarial Debiasing...")
                    
                        intervention_results = train_and_predict_model(
                            model_name=model_name,  
                            dataset_name=dataset_name,
                            data_df=data_df_intervene,
                            data_encoded=data_encoded_intervene,
                            feature_names=feature_names_intervene,
                            categorical_features=categorical_features_intervene,
                            _label_encoders=label_encoders_intervene, 
                            selected_state=selected_state,
                            selected_year=selected_year,
                            debias=True,
                            adversarial_sensitive_attrs=protected_attributes
                        )
                
                    st.session_state['intervention_results'] = intervention_results

                    st.session_state['data_df_intervene'] = data_df_intervene  
                    st.session_state['data_initial_intervene'] = data_initial_intervene
                    st.session_state['data_encoded_intervene'] = data_encoded_intervene
                    st.session_state['target_intervene'] = target_intervene
                    st.session_state['class_names_intervene'] = class_names_intervene
                    st.session_state['feature_names_intervene'] = feature_names_intervene
                    st.session_state['categorical_features_intervene'] = categorical_features_intervene
                    st.session_state['categorical_indices_intervene'] = categorical_indices_intervene
                    st.session_state['categorical_names_intervene'] = categorical_names_intervene
                    st.session_state['label_encoders_intervene'] = label_encoders_intervene

                    st.session_state.intervention_trained = True
                    st.session_state.int_fairness_calculated = False  
                    st.rerun()
            
            if st.session_state.intervention_trained:
                 
                calc_fairness_btn = st.sidebar.button("Calculate Fairness Metrics", key="int_calc_fairness")
                if calc_fairness_btn:
                    st.session_state.int_fairness_calculated = True
                     

            if st.session_state.intervention_trained and 'intervention_results' in st.session_state:
                
                intervention_results = st.session_state['intervention_results']
                data_df = st.session_state['data_df_intervene']
                data_encoded_intervene = st.session_state['data_encoded_intervene']
                feature_names_intervene = st.session_state['feature_names_intervene']
                categorical_indices = st.session_state['categorical_indices_intervene']
                categorical_names = st.session_state['categorical_names_intervene']
                categorical_features = st.session_state['categorical_features_intervene']
                class_names = st.session_state['class_names_intervene']
                continuous_features = [f for f in feature_names_intervene if f not in categorical_features]
                st.subheader(f"📈 Model performance for: {model_name} (Intervened)")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.metric(label="Model Accuracy", value=f"{intervention_results['accuracy']:.2%}")
                    st.subheader("Confusion Matrix")
                    st.pyplot(intervention_results["fig_cm"], use_container_width=True) 
                
                st.divider()

                

                if st.session_state.int_fairness_calculated:
                    X_test_intervene = intervention_results["X_test"]
                    y_test_intervene = intervention_results["y_test"]
                    y_pred_intervene = intervention_results["y_pred"]
                    focus_group_mask = st.session_state.data_loader["focus_group_mask"]
                    focus_group_name = st.session_state.get('focus_group_name', 'Focus Group')
                    rest_group_name = st.session_state.get('rest_group_name', 'Rest of Population')
                    
                    focus_plot_name = shorten_group_name(focus_group_name, "Focus Group")
                    rest_plot_name  = shorten_group_name(rest_group_name, "Rest of Population")
                    fm_base = st.session_state.fm
                    test_indices = X_test_intervene.index 
                    
                    test_focus_mask = focus_group_mask.reindex(test_indices, fill_value=False) 
                    
                    
                    if test_focus_mask.sum() == 0:
                        st.error("⚠️ The Focus Group is empty in the Intervened Test Set.")
                        st.stop()

                    X_test_with_focus = X_test_intervene.copy()
                    X_test_with_focus['focus_group'] = np.where(test_focus_mask, 'Focus Group', 'Rest of Population')

                    try:
                        with st.spinner("Calculating fairness metrics..."):
                            fm_intervene = FairnessMetrics(  
                                X_test=X_test_with_focus,
                                y_test=y_test_intervene,  
                                y_pred=y_pred_intervene,  
                                sensitive_attr='focus_group',
                                attr_values=['Focus Group', 'Rest of Population'],
                                attr_names=[focus_group_name, rest_group_name],  
                                output_dir="fairness_reports"
                            )
                    except Exception as e:
                        st.error(f"Error instantiating Intervened FairnessMetrics: {e}")
                        st.stop()
                    
                    st.subheader(f"Group Fairness metrics after removing protected attribute.")
                    
                    
                    display_fairness_metrics(
                        fm=fm_intervene,
                        focus_group_name=focus_group_name,
                        rest_group_name=rest_group_name
                    )
                    
                    
                    if fm_base is not None:
                        display_fairness_delta(
                            fm_base=fm_base,
                            fm_intervene=fm_intervene,
                            focus_group_name=focus_group_name,
                            rest_group_name=rest_group_name,
                        )
                    st.divider()
                    

                data_df = st.session_state['data_df_intervene']
                data_encoded_intervene = st.session_state['data_encoded_intervene']
                feature_names_intervene = st.session_state['feature_names_intervene']
                categorical_indices = st.session_state['categorical_indices_intervene']
                categorical_names = st.session_state['categorical_names_intervene']
                categorical_features = st.session_state['categorical_features_intervene']
                class_names = st.session_state['class_names_intervene']
                continuous_features = [f for f in feature_names_intervene if f not in categorical_features]
                y_pred_intervene = intervention_results["y_pred"]
                y_test_intervene = intervention_results["y_test"]
                X_test_intervene = intervention_results["X_test"]
                focus_group_mask = st.session_state.data_loader["focus_group_mask"]

                X_test_base = st.session_state.base_model_results["X_test"]


                
                if 'saved_int_exp_type' not in st.session_state:
                    st.session_state.saved_int_exp_type = st.session_state.get('saved_explanation_type', "Please select...")
                    
                if 'saved_int_exp_method' not in st.session_state:
                    st.session_state.saved_int_exp_method = st.session_state.get('saved_explanation_method', "Please select...")
                    
                if 'saved_int_outcome_group' not in st.session_state:
                    st.session_state.saved_int_outcome_group = st.session_state.get('saved_outcome_group', "Please select...")
                    
                if 'saved_int_viz_type' not in st.session_state:
                    st.session_state.saved_int_viz_type = st.session_state.get('saved_viz_type', "Please select...")
                # ---------------------------------------------------
                    
                focus_group_filters = st.session_state.data_loader["focus_group_filters"]
                 
                focus_group_name = st.session_state.get('focus_group_name', 'Focus Group')
                rest_group_name = st.session_state.get('rest_group_name', 'Rest of Population')
                 
                focus_group_name_int = focus_group_name + "_INT"
                rest_group_name_int = rest_group_name + "_INT"
                
                st.sidebar.header("Explanation Settings")
                exp_type_opts = ["Please select...", "Individual", "Group"]
                try:
                    type_idx = exp_type_opts.index(st.session_state.saved_int_exp_type)
                except ValueError:
                    type_idx = 0
                    
                exp_type = st.sidebar.selectbox(
                    "Select Explanation Type", 
                    exp_type_opts,
                    index=type_idx,
                    key="widget_int_exp_type",
                    on_change=save_widget_state,
                    args=("saved_int_exp_type", "widget_int_exp_type")
                )
                
                
                if st.session_state.saved_int_exp_type == "Individual":
                    exp_method_opts = ["Please select...", "LIME", "SHAP", "DiCE", "Best"]
                else:
                    exp_method_opts = ["Please select...", "FACEGroup"]

                try:
                    method_idx = exp_method_opts.index(st.session_state.saved_int_exp_method)
                except ValueError:
                    method_idx = 0
                    st.session_state.saved_int_exp_method = "Please select..."

                exp_method = st.sidebar.selectbox(
                    "Explanation Method", 
                    exp_method_opts,
                    index=method_idx,
                    key="widget_int_exp_method",
                    on_change=save_widget_state,
                    args=("saved_int_exp_method", "widget_int_exp_method")
                )

                if st.session_state.saved_int_exp_type == "Please select...":
                    st.info("Please select an explanation type above to generate explanations.")
                    st.stop()
                
                outcome_opts = [
                        "Please select...", "All Predictions", "All Positives (P)", "All Negatives (N)",
                        "True Positives (TP)", "False Positives (FP)", "True Negatives (TN)", "False Negatives (FN)"
                    ]
                try:
                    out_idx = outcome_opts.index(st.session_state.saved_int_outcome_group)
                except ValueError:
                    out_idx = 0
                    
                outcome_group = st.sidebar.selectbox(
                    "Analyze explanations for which group?", 
                    outcome_opts,
                    index=out_idx,
                    key="widget_int_outcome_group",
                    on_change=save_widget_state,
                    args=("saved_int_outcome_group", "widget_int_outcome_group")
                )

                if st.session_state.saved_int_outcome_group == "Please select...":
                    st.info("Please select an outcome group above to generate explanations.")
                    st.stop()

                
                current_method = st.session_state.saved_int_exp_method
                
                
                
                is_positive_pred = (y_pred_intervene == 1)
                is_negative_pred = (y_pred_intervene == 0)
                is_true_positive = (y_test_intervene == 1) & (y_pred_intervene == 1)
                is_false_positive = (y_test_intervene == 0) & (y_pred_intervene == 1)
                is_true_negative = (y_test_intervene == 0) & (y_pred_intervene == 0)
                is_false_negative = (y_test_intervene == 1) & (y_pred_intervene == 0)

                X_test_outcome_subset = X_test_intervene.copy() 
                
                if outcome_group == "All Positives (P)":
                    X_test_outcome_subset = X_test_intervene[is_positive_pred]
                elif outcome_group == "All Negatives (N)":
                    X_test_outcome_subset = X_test_intervene[is_negative_pred]
                elif outcome_group == "True Positives (TP)":
                    X_test_outcome_subset = X_test_intervene[is_true_positive]
                elif outcome_group == "False Positives (FP)":
                    X_test_outcome_subset = X_test_intervene[is_false_positive]
                elif outcome_group == "True Negatives (TN)":
                    X_test_outcome_subset = X_test_intervene[is_true_negative]
                elif outcome_group == "False Negatives (FN)":
                    X_test_outcome_subset = X_test_intervene[is_false_negative]

                test_indices = X_test_outcome_subset.index
                    
                focus_mask_on_subset = focus_group_mask.reindex(test_indices, fill_value=False) 
                
                X_test_focus_group = X_test_outcome_subset[focus_mask_on_subset]
                X_test_rest_group = X_test_outcome_subset[~focus_mask_on_subset]
                
                n_focus = len(X_test_focus_group)
                n_rest = len(X_test_rest_group)
                max_possible = max(n_focus, n_rest)

                if max_possible == 0:
                    st.warning("No instances found in the selected outcome group.")
                    st.stop()

                st.sidebar.markdown("---")
                st.sidebar.markdown("##### ⏱️ Sampling")
                st.sidebar.write("Some explanation methods can be slow on large datasets. Select a sample to save time.")
                
                default_limit = max_possible
                user_sample_limit = st.sidebar.slider(
                    "Max instances per group:",
                    min_value=5,
                    max_value=max_possible if max_possible > 5 else 10,
                    value=default_limit,
                    step=5,
                    key="user_sample_limit_slider_step3" 
                )
                
                
                if len(X_test_focus_group) > user_sample_limit:                    
                    X_test_focus_group_sampled = X_test_focus_group.sample(user_sample_limit, random_state=42)
                else:
                    X_test_focus_group_sampled = X_test_focus_group

                if len(X_test_rest_group) > user_sample_limit:                   
                    X_test_rest_group_sampled = X_test_rest_group.sample(user_sample_limit, random_state=42)
                else:               
                    X_test_rest_group_sampled = X_test_rest_group

                if X_test_focus_group_sampled.empty and X_test_rest_group_sampled.empty:            
                    st.warning("Selected samples resulted in empty dataframes.")
                    st.stop()
                    
                if "explanations_generated_step3" not in st.session_state:
                    st.session_state.explanations_generated_step3 = False

                if st.sidebar.button("Generate Explanations", key="gen_exp_btn_step3"):
                    st.session_state.explanations_generated_step3 = True
                
                if not st.session_state.explanations_generated_step3:
                    st.stop()

                data_df_no_focus = data_df.drop(columns=["focus_group"], errors="ignore")
                categorical_indices_int = []
                categorical_names_int = {}
                
                if ('categorical_names' in globals() and 'feature_names' in globals() and 
                    isinstance(categorical_names, dict) and isinstance(feature_names_intervene, list)):
                    
                    
                    for old_index, names_list in categorical_names.items():
                        if isinstance(old_index, int) and old_index < len(feature_names):
                            old_ohe_feature_name = feature_names[old_index]
                        else:
                            continue 

                        if old_ohe_feature_name in feature_names_intervene:
                            try:    
                                new_index = feature_names_intervene.index(old_ohe_feature_name)
                                if new_index not in categorical_indices_int:
                                    categorical_indices_int.append(new_index)
                                    categorical_names_int[new_index] = names_list
                                    
                            except ValueError:
                                continue 
                                
                 
                categorical_indices_int.sort()
                
                if not categorical_indices_int and not categorical_names_int:
                        categorical_indices_int = categorical_indices 
                        categorical_names_int = categorical_names    
                continuous_features_intervened = [
                    f for f in continuous_features if f not in protected_attributes
                ]               

                categorical_indices_int = []
                categorical_names_int = {}

                
                for old_index, names_list in categorical_names.items():
                    if not isinstance(old_index, int) or old_index >= len(feature_names):
                        continue

                    old_feature = feature_names[old_index]

                    
                    if old_feature not in feature_names_intervene:
                        continue 

                    
                    try:
                        new_index = feature_names_intervene.index(old_feature)
                    except ValueError:
                        continue  

                    categorical_indices_int.append(new_index)
                    categorical_names_int[new_index] = names_list

                
                categorical_indices_int.sort()

                 

                 
                continuous_features_intervened = [
                    f for f in continuous_features if f in feature_names_intervene  
                ]
               
                try:
                    explainer_obj_int = Explainer(
                        data_df=data_df_no_focus,  
                        data_encoded=data_encoded_intervene, 
                            
                        class_names=class_names, 
                        feature_names=feature_names_intervene, 
                        categorical_names=categorical_names_int, 
                        categorical_indices=categorical_indices_int,
                        continuous_features=continuous_features_intervened 
                    )
                except Exception as e:
                    st.error(f"Error initializing Explainer for intervened model: {e}")
                    st.stop()

                if exp_method == "LIME":
                    model_input = intervention_results["predict_lime"]
                elif exp_method == "SHAP":
                    model_input = intervention_results["predict_shap"]
                elif exp_method == "DiCE":
                    model_input = intervention_results["dice_model"]
                
                st.subheader("Explain")
                st.write("Diagnose *why* the intervened model produces certain outcomes.")
                st.markdown(f"### Generating **{exp_method}** Explanations for Intervened Model")

                if st.session_state.explanations_generated_step3 == True:
                    
                    if exp_method == "LIME" or exp_method == "SHAP":
                        explanations_per_group = {}
                        
                        st.markdown("---")
                        st.subheader(f"Processing: {focus_group_name}")
                        focus_explanations = get_cached_explanations(
                            explainer_obj_int,  
                            exp_method,      
                            model_name,
                            model_input,     
                            X_test_focus_group_sampled,
                            focus_group_name_int,  
                            outcome_group
                        )
                        if focus_explanations:
                            explanations_per_group[focus_group_name] = focus_explanations
                        else:
                            st.info(f"No explanations generated for **{focus_group_name}** (group empty).")

                        st.markdown("---")
                        st.subheader(f"Processing: {rest_group_name}")
                        rest_explanations = get_cached_explanations(
                            explainer_obj_int,  
                            exp_method,      
                            model_name,
                            model_input,     
                            X_test_rest_group_sampled,
                            rest_group_name_int,  
                            outcome_group
                        )
                        if rest_explanations:
                            explanations_per_group[rest_group_name] = rest_explanations
                        else:
                            st.info(f"No explanations generated for **{rest_group_name}** (group empty).")

                        
                        st.divider()
                        
                        if explanations_per_group:
                            st.success(f"✅ {exp_method} Explanations for Intervened Model ready! Choose a visualization below.")
                        
                        
                        if 'intervene_explanations' not in st.session_state:
                            st.session_state['intervene_explanations'] = {}

                        st.session_state['intervene_explanations'][exp_method] = {
                            'explainer_obj': explainer_obj_int, 
                            'explanations': explanations_per_group,
                            'focus_group_name': focus_group_name,
                            'rest_group_name': rest_group_name,
                            'outcome_group': outcome_group,
                            'model_name': 'Intervened Model',
                        }
                        
                        
                        
                        custom_colors = {focus_group_name: "#F07F09", rest_group_name: "#1B587C"}
                        group_names = [focus_group_name, rest_group_name]
                        
                            
                        top_features = explainer_obj_int.get_global_feature_order( 
                            explanations_per_group, group_names, top_n=10
                        )
                        data_subsets = {
                            focus_group_name: X_test_focus_group_sampled,  
                            rest_group_name: X_test_rest_group_sampled
                        }
                        aggregation_choices = ["Please select...", "Distributions (Beeswarm Plot)", "Violin Plot (Distributions)", "Instance Heatmap","Mean Aggregation", "Mean Absolute Aggregation"]
                        
                        try:
                            viz_idx = aggregation_choices.index(st.session_state.saved_int_viz_type)
                        except ValueError:
                            viz_idx = 0
                        
                        viz_type = []
                        if st.session_state.saved_int_exp_type == "Individual":
                            viz_type = st.sidebar.selectbox(
                                "Select Visualizations", 
                                aggregation_choices,
                                index=viz_idx,
                                key="widget_int_viz_type",
                                on_change=save_widget_state,
                                args=("saved_int_viz_type", "widget_int_viz_type")
                            )
                        
                        if st.session_state.saved_int_viz_type == "Please select...":
                            st.info("Please select a visualization to visualize explanations.")
                            st.stop()
                        fig = None
                        if viz_type == "Violin Plot (Distributions)":
                            st.subheader("Feature Contribution Distributions")
                            st.caption(f"Comparing distributions for {model_name} (Intervened).")
                            
                           
                            fig = explainer_obj_int.plot_violin_distributions(
                                explanations_per_group, group_names, custom_colors, feature_order=top_features
                            )
                            
                            if fig:
                                st.pyplot(fig, use_container_width=False)

                                
                                st.markdown(
                                    "#### 📏 Distribution Differences (vs Base Model)", 
                                    help="Distribution distance among groups usings Wasserstein distance and difference from the base model. Larger values indicate a greater disparity between the two groups for that feature.\n\n"
                                            "Red number: Highest disparity.\n"
                                            "🔻 Green arrow: Disparity decreased compared to Base Model.\n"
                                            "🔺 Red arrow: Disparity increased compared to Base Model."
                                )
                                
                               
                                diff_df_int = explainer_obj_int.compute_wasserstein_table(
                                    explanations_per_group, 
                                    group_names, 
                                    feature_order=top_features
                                )
                                
                               
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and exp_method in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations'][exp_method]
                                        base_explainer = base_data['explainer_obj']
                                        base_exps = base_data['explanations']
                                        
                                        
                                        diff_df_base = base_explainer.compute_wasserstein_table(
                                            base_exps, group_names, feature_order=top_features
                                        )
                                        
                                        base_diff_map = diff_df_base["Wasserstein Difference"].to_dict()
                                    except Exception as e:
                                       
                                        pass

                              
                                if not diff_df_int.empty:
                                    max_val = diff_df_int["Wasserstein Difference"].max()
                                    font_size = "18px" 

                                    markdown_lines = []
                                    
                                    for feature_name, row in diff_df_int.iterrows():
                                        current_val = row["Wasserstein Difference"]
                                        
                                        
                                        if current_val == max_val:
                                            
                                            val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{current_val:.3f}</span>'
                                        else:
                                            val_str = f"{current_val:.3f}"
                                        
                                        
                                        delta_str = ""
                                        if feature_name in base_diff_map:
                                            base_val = base_diff_map[feature_name]
                                            delta = current_val - base_val
                                            
                                            
                                            if delta < -0.001: 
                                                delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.3f})</span>'
                                            
                                            elif delta > 0.001:
                                                delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.3f})</span>'
                                            else:
                                                delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                       
                                        line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>'
                                        markdown_lines.append(line)
                                    
                                    st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        
                        elif viz_type == "Distributions (Beeswarm Plot)":
                            st.subheader("Feature Contribution Distributions")
                            fig = explainer_obj_int.plot_beeswarm_comparison(
                                explanations_per_group, group_names, custom_colors, feature_order=top_features, data_per_group=data_subsets
                            )
                            
                            if fig:
                                st.pyplot(fig, use_container_width=False)

                                
                                st.markdown(
                                    "#### 📏 Distribution Differences (vs Base Model)", 
                                    help="Distribution distance among groups usings Wasserstein distance and difference from the base model. Larger values indicate a greater disparity between the two groups for that feature.\n\n"
                                            "Red number: Highest disparity.\n"
                                            "🔻 Green arrow: Disparity decreased compared to Base Model.\n"
                                            "🔺 Red arrow: Disparity increased compared to Base Model."
                                )
                                
                               
                                diff_df_int = explainer_obj_int.compute_wasserstein_table(
                                    explanations_per_group, 
                                    group_names, 
                                    feature_order=top_features
                                )
                                
                              
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and exp_method in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations'][exp_method]
                                        base_explainer = base_data['explainer_obj']
                                        base_exps = base_data['explanations']
                                        
                                        
                                        diff_df_base = base_explainer.compute_wasserstein_table(
                                            base_exps, group_names, feature_order=top_features
                                        )
                                        
                                        base_diff_map = diff_df_base["Wasserstein Difference"].to_dict()
                                    except Exception as e:
                            
                                        pass

                                
                                if not diff_df_int.empty:
                                    max_val = diff_df_int["Wasserstein Difference"].max()
                                    font_size = "18px"  

                                    markdown_lines = []
                                    
                                    for feature_name, row in diff_df_int.iterrows():
                                        current_val = row["Wasserstein Difference"]
                                        
                                        
                                        if current_val == max_val:
                                            
                                            val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{current_val:.3f}</span>'
                                        else:
                                            val_str = f"{current_val:.3f}"
                                        
                                       
                                        delta_str = ""
                                        if feature_name in base_diff_map:
                                            base_val = base_diff_map[feature_name]
                                            delta = current_val - base_val
                                            
                                            
                                            if delta < -0.001: 
                                                delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.3f})</span>'
                                            
                                            elif delta > 0.001:
                                                delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.3f})</span>'
                                            else:
                                                delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                        
                                        line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>'
                                        markdown_lines.append(line)
                                    
                                    st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        elif viz_type == "Instance Heatmap":
                                
                            st.subheader("Feature Contribution Heatmap")
                            n_instances = st.slider("Number of Instances", 5, 50, 10)
                            fig = explainer_obj_int.plot_instance_heatmap(
                                explanations_per_group, 
                                [focus_group_name, rest_group_name], 
                                n_instances=n_instances,
                                feature_order=top_features
                            )
                            
                        
                            if fig:
                                st.pyplot(fig, use_container_width=False)

                                
                                st.markdown(
                                    "#### 📏 Distribution Differences (vs Base Model)", 
                                    help="Distribution distance among groups usings Wasserstein distance and difference from the base model. Larger values indicate a greater disparity between the two groups for that feature.\n\n"
                                            "Red number: Highest disparity.\n"
                                            "🔻 Green arrow: Disparity decreased compared to Base Model.\n"
                                            "🔺 Red arrow: Disparity increased compared to Base Model."
                                )
                                
                                
                                diff_df_int = explainer_obj_int.compute_wasserstein_table(
                                    explanations_per_group, 
                                    group_names, 
                                    feature_order=top_features
                                )
                                
                               
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and exp_method in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations'][exp_method]
                                        base_explainer = base_data['explainer_obj']
                                        base_exps = base_data['explanations']
                                        
                                        
                                        diff_df_base = base_explainer.compute_wasserstein_table(
                                            base_exps, group_names, feature_order=top_features
                                        )
                                        
                                        base_diff_map = diff_df_base["Wasserstein Difference"].to_dict()
                                    except Exception as e:
                                        
                                        pass

                                
                                if not diff_df_int.empty:
                                    max_val = diff_df_int["Wasserstein Difference"].max()
                                    font_size = "18px"  

                                    markdown_lines = []
                                    
                                    for feature_name, row in diff_df_int.iterrows():
                                        current_val = row["Wasserstein Difference"]
                                        
                                        
                                        if current_val == max_val:
                                            
                                            val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{current_val:.3f}</span>'
                                        else:
                                            val_str = f"{current_val:.3f}"
                                        
                                        
                                        delta_str = ""
                                        if feature_name in base_diff_map:
                                            base_val = base_diff_map[feature_name]
                                            delta = current_val - base_val
                                            
                                            
                                            if delta < -0.001: 
                                                delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.3f})</span>'
                                            
                                            elif delta > 0.001:
                                                delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.3f})</span>'
                                            else:
                                                delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                        
                                        line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>'
                                        markdown_lines.append(line)
                                    
                                    st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        elif viz_type == "Mean Aggregation":
                            st.subheader("Mean Feature Contributions")
                            st.write(f"Comparing Mean Contributions for {model_name} (Intervened).")
                            fig = explainer_obj_int.plot_mean_aggregation(
                                explanations_per_group, group_names, custom_colors, feature_order=top_features
                            )
                            if fig: 
                                st.pyplot(fig, use_container_width=False)
                                
                                st.markdown("#### 📏 Differences in Mean Feature Contributions (vs Base Model)", help="Absolute difference between the average contribution of the two groups and difference from the base model."
                                            "\nRed number: Highest disparity."
                                            "\n🔻 Green arrow: Disparity decreased compared to Base Model."
                                            "\n🔺 Red arrow: Disparity increased compared to Base Model.")
                                
                                
                                diff_df_int = explainer_obj_int.compute_mean_diff_table(explanations_per_group, group_names, feature_order=top_features)
                                
                            
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and exp_method in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations'][exp_method]
                                        diff_df_base = base_data['explainer_obj'].compute_mean_diff_table(base_data['explanations'], group_names, feature_order=top_features)
                                        base_diff_map = diff_df_base["Difference"].to_dict()
                                    except: pass

                               
                                if not diff_df_int.empty:
                                    max_val = diff_df_int["Difference"].max()
                                    font_size = "18px"
                                    markdown_lines = []
                                    for feature_name, row in diff_df_int.iterrows():
                                        curr = row["Difference"]
                                        val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{curr:.3f}</span>' if curr == max_val else f"{curr:.3f}"
                                        
                                        
                                        delta_str = ""
                                        if feature_name in base_diff_map:
                                            delta = curr - base_diff_map[feature_name]
                                            if delta < -0.001: delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.3f})</span>'
                                            elif delta > 0.001: delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.3f})</span>'
                                            else: delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                        markdown_lines.append(f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>')
                                    st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        elif viz_type == "Mean Absolute Aggregation":
                            st.subheader("Mean Absolute Feature Importance")
                            st.write(f"Comparing Importance for {model_name} (Intervened).")
                            fig = explainer_obj_int.plot_mean_abs_aggregation(
                                explanations_per_group, group_names, custom_colors, feature_order=top_features
                            )
                            if fig: 
                                st.pyplot(fig, use_container_width=False)

                                st.markdown("#### 📏 Differences in Absolute Feature Importance (vs Base Model)", help="Difference in the average magnitude of the feature.")
                                
                                diff_df_int = explainer_obj_int.compute_mean_abs_diff_table(explanations_per_group, group_names, feature_order=top_features)
                                
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and exp_method in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations'][exp_method]
                                        diff_df_base = base_data['explainer_obj'].compute_mean_abs_diff_table(base_data['explanations'], group_names, feature_order=top_features)
                                        base_diff_map = diff_df_base["Difference"].to_dict()
                                    except: pass

                                if not diff_df_int.empty:
                                    max_val = diff_df_int["Difference"].max()
                                    font_size = "18px"
                                    markdown_lines = []
                                    for feature_name, row in diff_df_int.iterrows():
                                        curr = row["Difference"]
                                        val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{curr:.3f}</span>' if curr == max_val else f"{curr:.3f}"
                                        
                                        delta_str = ""
                                        if feature_name in base_diff_map:
                                            delta = curr - base_diff_map[feature_name]
                                            if delta < -0.001: delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.3f})</span>'
                                            elif delta > 0.001: delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.3f})</span>'
                                            else: delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                        markdown_lines.append(f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>')
                                    st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        
                    elif exp_method == "DiCE":
                        
                        focus_indices = X_test_focus_group_sampled.index
                        rest_indices = X_test_rest_group_sampled.index
                    
                        df_focus_original = data_df.loc[focus_indices].drop(columns=["focus_group"], errors="ignore")
                        df_rest_original = data_df.loc[rest_indices].drop(columns=["focus_group"], errors="ignore")
                    
                        
                         
                        
                        st.markdown("---")
                        st.subheader(f"Processing: {focus_group_name}")
                        focus_explanations = get_cached_explanations(
                            explainer_obj_int, 
                            "DiCE", model_name, model_input, 
                            df_focus_original,  
                            focus_group_name_int, 
                            outcome_group
                        )
                        
                        st.markdown("---")
                        st.subheader(f"Processing: {rest_group_name}")
                        rest_explanations = get_cached_explanations(
                            explainer_obj_int, 
                            "DiCE", model_name, model_input, 
                            df_rest_original,  
                            rest_group_name_int, 
                            outcome_group
                        )

                        st.divider()
                        st.success("✅ DiCE Explanations for Intervened Model ready!")
            
                        viz_dice_type = st.session_state.get("saved_dice_viz_type", "Please select...")
                        viz_choices = [
                            "Please select...",
                            "Beeswarm Plot (Magnitude of Changes)", 
                            "Bar Plot (Percentages of Feature Change)"
                        ]

                        try:
                            viz_dice_index = viz_choices.index(st.session_state.saved_dice_viz_type)
                        except ValueError:
                            viz_dice_index = 0

                        if st.session_state.saved_explanation_type == "Individual":
                            viz_dice_type = st.sidebar.selectbox(
                                "Select Visualizations", 
                                viz_choices,
                                index=viz_dice_index,
                                key="widget_dice_viz_type",
                                on_change=save_widget_state,
                                args=("saved_dice_viz_type", "widget_dice_viz_type")
                            )

                        if st.session_state.saved_dice_viz_type  == "Please select...":
                            st.info("Please select a visualization to visualize explanations.")
                            st.stop()

                        explanations_per_group = {
                                focus_group_name: focus_explanations,
                                rest_group_name: rest_explanations
                            }
                            

                        if 'intervene_explanations' not in st.session_state:
                            st.session_state['intervene_explanations'] = {}

                        st.session_state['intervene_explanations']['DiCE'] = {
                            'explainer_obj': explainer_obj_int, 
                            'explanations': explanations_per_group,
                            'focus_group_name': focus_group_name,
                            'rest_group_name': rest_group_name,
                            'outcome_group': outcome_group,
                            'model_name': 'Intervened Model',
                            'df_focus_original': df_focus_original,
                            'df_rest_original': df_rest_original,
                        }

                        
                        st.subheader("Counterfactual Feature Changes")
                        st.write("This plot shows **how often** has a feature to change to flip the prediction.")
                        
                        custom_colors = {focus_group_name: "#F07F09", rest_group_name: "#1B587C"}
                        group_names = [focus_group_name, rest_group_name]
                        
                        data_subsets = {
                            focus_group_name: df_focus_original, 
                            rest_group_name: df_rest_original
                        }
                        top_features = explainer_obj.get_global_feature_order_dice(explanations_per_group, group_names,data_subsets, top_n=10)

                        df_changes = explainer_obj_int.compute_counterfactual_difference( 
                            explanations_per_group, 
                            group_names,
                            data_subsets
                        )
                        
                        if focus_group_name in df_changes.columns and rest_group_name in df_changes.columns:
                                df_changes['Change_Diff'] = df_changes[focus_group_name] - df_changes[rest_group_name]
                        else:
                            df_changes['Change_Diff'] = 0
                        

                        if viz_dice_type == "Beeswarm Plot (Magnitude of Changes)":
                            st.subheader("Feature Contribution Distributions")
                            fig = explainer_obj.plot_dice_beeswarm(explanations_per_group, group_names, custom_colors, data_subsets, feature_order=top_features)

                        elif viz_dice_type == "Bar Plot (Percentages of Feature Change)":
                            st.subheader("Feature Contribution Distributions")
                            fig = explainer_obj.plot_dice_comparison(
                                df_changes, group_names, custom_colors, feature_order=top_features
                            )
                        
                        if fig:
                            st.pyplot(fig, use_container_width=False)

                            
                            st.markdown(
                                "#### 📏 Frequency Differences (vs Base Model)", 
                                help="Difference in how often a feature changes between groups.\n"
                                        "🔻 Green: Disparity decreased.\n"
                                        "🔺 Red: Disparity increased."
                            )

                            if not df_changes.empty:
                                display_df_int = df_changes.copy()
                                display_df_int['Abs_Diff'] = display_df_int['Change_Diff'].abs()
                                
                                base_diff_map = {}
                                if 'base_explanations' in st.session_state and 'DiCE' in st.session_state['base_explanations']:
                                    try:
                                        base_data = st.session_state['base_explanations']['DiCE']
                                        base_df = base_data['explainer_obj'].compute_counterfactual_difference(
                                            base_data['explanations'],
                                            [base_data['focus_group_name'], base_data['rest_group_name']],
                                            {base_data['focus_group_name']: base_data['df_focus_original'], 
                                                base_data['rest_group_name']: base_data['df_rest_original']}
                                        )
                                        
                                        g1, g2 = base_data['focus_group_name'], base_data['rest_group_name']
                                        if g1 in base_df.columns and g2 in base_df.columns:
                                            base_df['Change_Diff'] = base_df[g1] - base_df[g2]
                                            base_diff_map = base_df['Change_Diff'].abs().to_dict()
                                    except Exception:
                                        pass

                                max_val = display_df_int['Abs_Diff'].max()
                                font_size = "18px"
                                markdown_lines = []
                                display_df_int = display_df_int.sort_values('Abs_Diff', ascending=False)

                                for feature_name, row in display_df_int.iterrows():
                                    curr = row['Abs_Diff']
                                    
                                    if curr == max_val:
                                        val_str = f'<span style="color: #ff4b4b; font-weight: bold;">{curr:.1f}%</span>'
                                    else:
                                        val_str = f"{curr:.1f}%"
                                    
                                    delta_str = ""
                                    if feature_name in base_diff_map:
                                        base_val = base_diff_map[feature_name]
                                        delta = curr - base_val
                                        
                                        if delta < -0.1: 
                                            delta_str = f' <span style="color: green; font-size: 0.8em;">(🔻 {abs(delta):.1f}%)</span>'
                                        elif delta > 0.1: 
                                            delta_str = f' <span style="color: red; font-size: 0.8em;">(🔺 {abs(delta):.1f}%)</span>'
                                        else: 
                                            delta_str = f' <span style="color: gray; font-size: 0.8em;">(=)</span>'

                                    line = f'- <span style="font-size: {font_size};">**{feature_name}**: {val_str}{delta_str}</span>'
                                    markdown_lines.append(line)
                                
                                st.markdown("\n".join(markdown_lines), unsafe_allow_html=True)
                        else:
                            st.warning("No feature changes found.")

if __name__ == "__main__":
    main()