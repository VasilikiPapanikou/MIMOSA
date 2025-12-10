import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from aif360.sklearn.datasets import fetch_compas
from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import streamlit as st
class MappingEncoder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.inverse_mapping = {v: k for k, v in mapping.items()}   

    def transform(self, values):
        return values.map(self.mapping)

    def inverse_transform(self, values):
        return values.map(self.inverse_mapping)
    
class DataLoader:

    def load_dataset(self, dataset_path, year, protected_to_remove, datasetName='Adult', min_max_scale = False, state = None):
        
        if datasetName == "Adult":
            return self.load_adult(dataset_path, protected_to_remove, min_max_scale)
        
        elif datasetName == "Compas":
            return self.load_compas(protected_to_remove, min_max_scale)

        elif datasetName == "GermanCredit":
            return self.load_german_credit(dataset_path, protected_to_remove, min_max_scale)
        
        elif state != None:
           
            return self.load_ACSData(state, protected_to_remove, min_max_scale, year=year)
        

    def encode_categorical(self, df, cat_features):
        """
        Encodes categorical features in the dataframe using LabelEncoder.

        Args:
            df: Input dataframe with categorical features.
            cat_features: List of column names that are categorical.

        """
        df_encoded = df.copy()
        encoders = {}
        cat_names = {}
        for feature in cat_features:
            idx = df_encoded.columns.get_loc(feature)
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature])
            encoders[feature] = le
            cat_names[idx] = le.classes_
        return df_encoded, encoders, cat_names
    
    def load_ACSData(self, state, protected_to_remove, min_max_scale = False, year = None):
        """
        Loads ACS dataset
        
        Args:
            datasetName: The name of the dataset to load.
            protected_to_remove: The protected attribute to remove (e.g., 'sex', 'race').
   
        """

        dir = '../data/acs_data'
        file = f'{dir}/{state}_{str(year)}.parquet'
        
        if os.path.exists(file):
            print(f"Loading cached ACS data for {state} from {file}")
            ca_data = pd.read_parquet(file)
        else:
            print(f"Dowloading ACS data for {state} ...")
            data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
            ca_data = data_source.get_data(states=[state], download=True)
            os.makedirs(dir, exist_ok=True)  
            ca_data.to_parquet(file)
    
        categorical_indices = [1, 2, 3, 4, 5, 7, 8]

        features_to_load = [
            'AGEP',  # Age
            'COW',   # Class of Worker
            'SCHL',  # Educational Attainment
            'MAR',   # Marital Status
            'OCCP',  # Occupation
            'POBP',  # Place of Birth
            'WKHP',  # Hours Worked per Week
            'SEX',   # Sex
            'RAC1P', # Race
            'PINCP'  # The target (Personal Income)
        ]

        categorical_features = ['Class of Worker','Educational Attainment', 'Marital Status', 'Sex','Race', "Occupation", "Place of Birth"]
        continuous_features = ['Age','Hours Worked per Week']

        data = ca_data[features_to_load].copy()
        data = data.dropna()
        ca_labels = (data['PINCP'] > 50000).astype(int)

        data = data.drop(columns=['PINCP'])
        occupation_mapping = {occ: idx + 1 for idx, occ in enumerate(data['OCCP'].unique())}
        place_of_birth_mapping = {pob: idx + 1 for idx, pob in enumerate(data['POBP'].unique())}
        data['OCCP'] = data['OCCP'].map(occupation_mapping)
        data['POBP'] = data['POBP'].map(place_of_birth_mapping)

        occupation_mapping_ = {
            i: f"Occupation {i}" 
            for i in range(0, len(data['OCCP'].unique()))
        }
        
        place_of_birth_mapping_mapping_ = {
            i: f"Place_of_birth {i}" 
            for i in range(0, len(data['POBP'].unique()))
        }
        ACSIncome_categories = {
            "COW": {
                0.0: (
                    "Employee of a private for-profit company or"
                    " business, or of an individual, for wages,"
                    " salary, or commissions"
                ),
                1.0: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
                2.0: "Local government employee (city, county, etc.)",
                3.0: "State government employee",
                4.0: "Federal government employee",
                5.0: (
                    "Self-employed in own not incorporated business,"
                    " professional practice, or farm"
                ),
                6.0: (
                    "Self-employed in own incorporated business,"
                    " professional practice or farm"
                ),
                7.0: "Working without pay in family business or farm",
                8.0: "Unemployed and last worked 5 years ago or earlier or never worked",
            },
            "SCHL": {
                0.0: "No schooling completed",
                1.0: "Nursery school, preschool",
                2.0: "Kindergarten",
                3.0: "Grade 1",
                4.0: "Grade 2",
                5.0: "Grade 3",
                6.0: "Grade 4",
                7.0: "Grade 5",
                8.0: "Grade 6",
                9.0: "Grade 7",
                10.0: "Grade 8",
                11.0: "Grade 9",
                12.0: "Grade 10",
                13.0: "Grade 11",
                14.0: "12th grade - no diploma",
                15.0: "Regular high school diploma",
                16.0: "GED or alternative credential",
                17.0: "Some college, but less than 1 year",
                18.0: "1 or more years of college credit, no degree",
                19.0: "Associate's degree",
                20.0: "Bachelor's degree",
                21.0: "Master's degree",
                22.0: "Professional degree beyond a bachelor's degree",
                23.0: "Doctorate degree",
            },
            "MAR": {
                0.0: "Married",
                1.0: "Widowed",
                2.0: "Divorced",
                3.0: "Separated",
                4.0: "Never married or under 15 years old",
            },
            "SEX": {0.0: "Male", 1.0: "Female"},
            "RAC1P": {
                0.0: "White",
                1.0: "Black",
                2.0: "American Indian alone",
                3.0: "Alaska Native alone",
                4.0: (
                    "American Indian and Alaska Native tribes specified;"
                    " or American Indian or Alaska Native, not specified and no other"
                ),
                5.0: "Asian alone",
                6.0: "Native Hawaiian and Other Pacific Islander alone",
                7.0: "Some Other Race alone",
                8.0: "Two or More Races",
            },
           "OCCP":occupation_mapping_,
           "POBP":place_of_birth_mapping_mapping_
        }

        categorical_names = {
            1: list(ACSIncome_categories["COW"].values()), 
            2: list(ACSIncome_categories["SCHL"].values()),  
            3: list(ACSIncome_categories["MAR"].values()),  
            4: list(ACSIncome_categories["OCCP"].values()),  
            5: list(ACSIncome_categories["POBP"].values()),  
            7: list(ACSIncome_categories["SEX"].values()), 
            8: list(ACSIncome_categories["RAC1P"].values()),  
        }

        data.rename(columns={
            'AGEP': 'Age',
            'COW': 'Class of Worker',
            'SCHL': 'Educational Attainment',
            'MAR': 'Marital Status',
            'OCCP': 'Occupation',
            'POBP': 'Place of Birth',
            'WKHP': 'Hours Worked per Week',
            'SEX': 'Sex',
            'RAC1P': 'Race'
        }, inplace=True)
        data_encoded = data.copy().astype(int) 
        feature_names = data.columns.tolist()
    
        for column in categorical_features:
            data_encoded[column] = data_encoded[column] - 1

        data_df = data_encoded.copy()
        data_df['Class of Worker'] = data_df['Class of Worker'].map(ACSIncome_categories["COW"])
        data_df['Educational Attainment'] = data_df['Educational Attainment'].map(ACSIncome_categories["SCHL"])
        data_df['Marital Status'] = data_df['Marital Status'].map(ACSIncome_categories["MAR"])
        data_df['Sex'] = data_df['Sex'].map(ACSIncome_categories["SEX"])
        data_df['Race'] = data_df['Race'].map(ACSIncome_categories["RAC1P"])
        data_df['Occupation'] = data_df['Occupation'].map(ACSIncome_categories["OCCP"])
        data_df['Place of Birth'] = data_df['Place of Birth'].map(ACSIncome_categories["POBP"])
        data_df["Target"] = ca_labels
        class_names = data_df['Target'].unique()
        labels = data_df["Target"]
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        data_df["Target"] = labels_encoded
        data_encoded["Target"] = labels_encoded
        encoders = {
            "Class of Worker": MappingEncoder({v: k for k, v in ACSIncome_categories["COW"].items()}),
            "Educational Attainment": MappingEncoder({v: k for k, v in ACSIncome_categories["SCHL"].items()}),
            "Marital Status": MappingEncoder({v: k for k, v in ACSIncome_categories["MAR"].items()}),
            "Sex": MappingEncoder({v: k for k, v in ACSIncome_categories["SEX"].items()}),
            "Race": MappingEncoder({v: k for k, v in ACSIncome_categories["RAC1P"].items()}),
            "Occupation": MappingEncoder({v: k for k, v in ACSIncome_categories["OCCP"].items()}),
            "Place of Birth": MappingEncoder({v: k for k, v in ACSIncome_categories["POBP"].items()}),
        }
        data_initial = data_df.copy()
        target_map = {0: '<=50K', 1: '>50K'}
        data_initial["Target"] = data_initial["Target"].map(target_map)
        if min_max_scale:
            scaler = MinMaxScaler()
            data_encoded[continuous_features] = scaler.fit_transform(data_encoded[continuous_features])
            data_df[continuous_features] = scaler.fit_transform(data_encoded[continuous_features])
        if len(protected_to_remove) > 0:
        
            feature_names = [feature for feature in feature_names if feature not in protected_to_remove]
            feature_names_ = feature_names + ["Target"]
            categorical_features = [feature for feature in categorical_features if feature not in protected_to_remove]
            categorical_indices = [feature_names.index(col) for col in categorical_features]
            categorical_names_without_protected = {}
            data_encoded_without_protected = data_df.copy()
            for i, feature in enumerate(categorical_features):
                encoders[feature] = encoders[feature]
            
            continuous_features = [feature for feature in continuous_features if feature not in protected_to_remove]
            data_df_without_protected = data_df.drop(columns=protected_to_remove)
            data_encoded_without_protected = data_encoded.drop(columns=protected_to_remove)
            
            return data_initial, data_df_without_protected, data_encoded_without_protected, class_names, feature_names, categorical_features, categorical_indices, categorical_names, encoders
        
        return data_initial, data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, encoders

    def load_german_credit(self, dataset_path, protected_to_remove, min_max_scale):
        """
        Loads German Credit dataset 
        
        Args:
            datasetName: The name of the dataset to load.
            protected_to_remove: The protected attribute to remove (e.g., 'sex', 'race').
       
        """
        
        column_names = ["Existing-Account-Status", "Month-Duration",
                              "Credit-History", "Purpose", "Credit-Amount",
                              "Savings-Account", "Present-Employment", "Instalment-Rate",
                              "Sex", "Guarantors", "Residence","Property", "Age",
                              "Installment", "Housing", "Existing-Credits", "Job",
                              "Num-People", "Telephone", "Foreign-Worker", "Status"]
        
        status_sex_mapping = {
        'A91': ('male', 'divorced/separated'),
        'A92': ('female', 'divorced/separated/married'),
        'A93': ('male', 'single'),
        'A94': ('male', 'married/widowed'),
        'A95': ('female', 'single')}
        data_df = pd.read_csv(f"{dataset_path}", header=None, delim_whitespace = True)
        data_df.columns = column_names
        data_df[data_df.columns[-1]] = 2 - data_df[data_df.columns[-1]]
        target = data_df[["Status"]]
        data_df = data_df.drop('Status', axis=1)

        data_df['Sex'], data_df['Marital-Status'] = zip(*data_df['Sex'].map(status_sex_mapping))
        
        data_df['Existing-Account-Status'] = data_df['Existing-Account-Status'].apply(lambda x: 'A10' if x == 'A14' else x)
        data_df['Savings-Account'] = data_df['Savings-Account'].apply(lambda x: 'A60' if x == 'A65' else x)
        
        data_df = pd.concat([data_df, target], axis=1)
        data_df.rename(columns={'Status': 'Target'}, inplace=True)
        class_names = data_df['Target'].unique()
         
        feature_names_ = list(data_df.columns)
         
        feature_names = feature_names_[:-1]
        categorical_features = ["Existing-Account-Status", "Credit-History","Purpose", "Savings-Account","Present-Employment", "Sex", "Guarantors","Property","Installment","Housing","Job","Telephone","Foreign-Worker","Marital-Status"]
        continuous_features = ["Month-Duration","Credit-Amount", "Instalment-Rate", "Residence", "Age","Existing-Credits","Num-People"]
        categorical_indices = [feature_names.index(col) for col in categorical_features]

        data_initial = data_df.copy()

        for feature in continuous_features:
            data_df[feature] = pd.to_numeric(data_df[feature], errors='coerce')
        categorical_names = {}
        data_encoded = data_df.copy()
        if min_max_scale:
            scaler = MinMaxScaler()
            data_encoded[continuous_features] = scaler.fit_transform(data_encoded[continuous_features])

        encoders = {}
        for i, feature in enumerate(categorical_features):
            
            enc = sklearn.preprocessing.LabelEncoder()
            data_encoded[feature] = enc.fit_transform(data_encoded[feature])
            categorical_names[categorical_indices[i]] = enc.classes_
            encoders[feature] = enc

        if len(protected_to_remove) > 0:
            feature_names = [feature for feature in feature_names if feature not in protected_to_remove]
            feature_names_ = feature_names + ["Target"]
            categorical_features = [feature for feature in categorical_features if feature not in protected_to_remove]
            categorical_indices = [feature_names.index(col) for col in categorical_features]
            categorical_names_without_protected = {}
            data_encoded_without_protected = data_df.copy()
            encoders = {}
            for i, feature in enumerate(categorical_features):
                enc = sklearn.preprocessing.LabelEncoder()
                data_encoded_without_protected[feature] = enc.fit_transform(data_encoded_without_protected[feature])
                categorical_names_without_protected[categorical_indices[i]] = enc.classes_
                encoders[feature] = enc
            
            continuous_features = [feature for feature in continuous_features if feature not in protected_to_remove]
            data_df_without_protected = data_df.drop(columns=protected_to_remove)
            data_encoded_without_protected = data_encoded.drop(columns=protected_to_remove)
            return data_initial, data_df_without_protected, data_encoded_without_protected, class_names, feature_names, categorical_features, categorical_indices, categorical_names, encoders
        
        return data_initial, data_df, data_encoded, class_names, feature_names, categorical_features, categorical_indices, categorical_names, encoders
    
    def load_compas(self, protected_to_remove=[], min_max_scale=False):
        """
        Loads and processes the COMPAS dataset.
        
        Args:
            protected_to_remove: List of protected attributes to remove (e.g., ['Sex', 'Race']).
            min_max_scal: Whether to apply Min-Max scaling to continuous features.
        
        """
        X, y = fetch_compas()
         
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        data = X.drop(['c_charge_desc', 'age_cat'], axis=1)
        data.columns = [col.capitalize() for col in data.columns]
        data['Target'] = y

        class_names = data['Target'].unique()
        feature_names = data.columns.drop('Target').tolist()
        categorical_features = ["Sex", "Race", "C_charge_degree"]
        continuous_features = ["Age", "Juv_fel_count", "Juv_misd_count", "Juv_other_count", "Priors_count"]

        for feature in continuous_features:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        data_initial = data.copy()
        scaler = None
        if min_max_scale:
            scaler = MinMaxScaler()
            data[continuous_features] = scaler.fit_transform(data[continuous_features])

        label_encoder = LabelEncoder()
        data["Target"] = label_encoder.fit_transform(data["Target"])
        class_names = label_encoder.classes_

        data_encoded, encoders, categorical_names = self.encode_categorical(data, categorical_features)
        categorical_indices = [feature_names.index(f) for f in categorical_features]

        if protected_to_remove:
            # Remove protected attributes
            feature_names_wo = [f for f in feature_names if f not in protected_to_remove]
            cat_features_wo = [f for f in categorical_features if f not in protected_to_remove]

            data_df_wo = data.drop(columns=protected_to_remove)
            data_encoded_wo = data_encoded.drop(columns=protected_to_remove)

            data_encoded_wo, encoders_wo, categorical_names_wo = self.encode_categorical(data_df_wo, cat_features_wo)
            cat_indices_wo = [feature_names_wo.index(f) for f in cat_features_wo]
            
            return (
                data_initial, data_df_wo, data_encoded_wo,
                class_names, feature_names_wo,
                cat_features_wo, cat_indices_wo,
                categorical_names,
                encoders
            )
 
        return (
            data_initial, data, data_encoded,
            class_names, feature_names,
            categorical_features, categorical_indices,
            categorical_names, encoders
        )

    def summarize_group_proportions(self, df, name, sex_col, target_col):
        
        df[target_col] = df[target_col].str.strip().str.replace('.', '', regex=False)

        total = len(df)
        male_count = (df[sex_col] == 'Male').sum()
        female_count = (df[sex_col] == 'Female').sum()

        positive_mask = df[target_col] == '>50K'
        negative_mask = df[target_col] == '<=50K'

        positive_count = positive_mask.sum()
        negative_count = negative_mask.sum()

        male_positive = ((df[sex_col] == 'Male') & positive_mask).sum()
        male_negative = ((df[sex_col] == 'Male') & negative_mask).sum()
        female_positive = ((df[sex_col] == 'Female') & positive_mask).sum()
        female_negative = ((df[sex_col] == 'Female') & negative_mask).sum()

        print(f"\n{name} Set:")
        print(f"  Total: {total}")
        print(f"  Sex:")
        print(f"    Male: {male_count} ({male_count / total:.2%})")
        print(f"    Female: {female_count} ({female_count / total:.2%})")
        print(f"  Income:")
        print(f"    >50K: {positive_count} ({positive_count / total:.2%})")
        print(f"    <=50K: {negative_count} ({negative_count / total:.2%})")
        print(f"  Breakdown by Sex and Income:")
        print(f"    Male >50K: {male_positive} ({male_positive / male_count:.2%} of males)")
        print(f"    Male <=50K: {male_negative} ({male_negative / male_count:.2%} of males)")
        print(f"    Female >50K: {female_positive} ({female_positive / female_count:.2%} of females)")
        print(f"    Female <=50K: {female_negative} ({female_negative / female_count:.2%} of females)")
    
    def load_adult(self, dataset_path, protected_to_remove = [], min_max_scale = False):
        """
        Loads and processes the UCI Adult dataset.
        
        Args:
            dataset_path: Path to the dataset folder.
            protected_to_remove:  List of protected attributes to remove (e.g., ['Sex', 'Race']).
        
        """
        column_names = ['Age', 'Workclass', 'fnlwgt', 'Education',
            'Education-Num', 'Marital Status', 'Occupation', 'Relationship',
            'Race', 'Sex', 'Capital gain', 'Capital loss', 'Hours per week',
            'Country', 'Target']
        
        feature_names = ["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Hours per week", "Country"]        
        
        categorical_features = ["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
        
        continuous_features = ["Age", "Hours per week"]

        train_file = dataset_path+ "/adult.data"
        test_file = dataset_path + "/adult.test"

        na_values=['?']
        train = pd.read_csv(train_file, header=None, names=column_names,
                skipinitialspace=True, na_values=na_values)
        test = pd.read_csv(test_file, header=0, names=column_names,
            skipinitialspace=True, na_values=na_values)
        
        self.summarize_group_proportions(train, "Train", "Sex", "Target")

        
        def clean_target(s: pd.Series) -> pd.Series:
            return (s.astype(str)
                    .str.replace(r'\.', '', regex=True)        # remove trailing periods
                    .str.replace(r'\s+', ' ', regex=True)       # collapse any whitespace
                    .str.replace('\u00A0', ' ', regex=False)    # non-breaking space
                    .str.strip()
                    .str.normalize('NFKC')
                )

        train['Target'] = clean_target(train['Target'])
        test['Target']  = clean_target(test['Target'])

        # force exact two values
        train['Target'] = train['Target'].map(
            {">50K": ">50K", "<=50K": "<=50K"}
        )
        test['Target'] = test['Target'].map(
            {">50K": ">50K", "<=50K": "<=50K"}
        )
        data = pd.concat([train, test], ignore_index=True)
        
        data = data.dropna()
        columns_to_drop = ['fnlwgt', 'Education-Num', 'Capital gain', 'Capital loss']
        data = data.drop(columns=columns_to_drop)
        data_initial = data.copy()
        
     
        data['Race'] = data['Race'].replace({
            'Asian-Pac-Islander': 'API',
            'Amer-Indian-Eskimo': 'AIE'
        })

        label_encoder = LabelEncoder()
        data['Target'] = data['Target'].astype(str).str.replace('.', '', regex=False).str.strip()
        
        data['Target'] = label_encoder.fit_transform(data['Target'])
        class_names = label_encoder.classes_

        for feature in continuous_features:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')

        scaler = None
        if min_max_scale:
            scaler = MinMaxScaler()
            data[continuous_features] = scaler.fit_transform(data[continuous_features])

        categorical_indices = [feature_names.index(f) for f in categorical_features]
        
        
        data_encoded, encoders, categorical_name_map = self.encode_categorical(data, categorical_features)
        
        if len(protected_to_remove)>0:
           
            features_wo = [f for f in feature_names if f not in protected_to_remove]
            cat_features_wo = [f for f in categorical_features if f not in protected_to_remove]

            data_df_wo = data.drop(columns=protected_to_remove)
            data_encoded_wo = data_encoded.drop(columns=protected_to_remove)

            data_encoded_wo, encoders_wo, categorical_name_map_wo = self.encode_categorical(data_df_wo, cat_features_wo)
            cat_indices_wo = [features_wo.index(f) for f in cat_features_wo]

            return (
                data_initial, data_df_wo, data_encoded_wo,
                class_names, features_wo,
                cat_features_wo, cat_indices_wo,
                categorical_name_map,
                encoders
            )

        return (
            data_initial, data, data_encoded, class_names,
            feature_names, categorical_features,
            categorical_indices, categorical_name_map,
            encoders
        )