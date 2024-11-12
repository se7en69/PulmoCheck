import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import io
from sklearn.preprocessing import StandardScaler

# Configure the page with title and emoji icon
st.set_page_config(
    page_title="PulmoCheck",  # Your app's title
    page_icon="🫁",  # Lung emoji as the page icon
)

# Define symptom features
symptom_features = ['Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty']

# Function to generate patient report
def generate_patient_report(patient_data):
    """Generate a patient report as a string."""
    patient_buffer = io.StringIO()
    patient_buffer.write("Patient Profile Report\n")
    patient_buffer.write(patient_data.to_string(index=False))
    patient_buffer.write("\n\nSymptom Severity Radar Chart and Risk Factor Analysis included.\n")
    report_content = patient_buffer.getvalue()
    patient_buffer.close()
    return report_content

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cancer_patient_data_sets.csv')

data = load_data()

# Prepare data for training models
X = data.drop(['Level', 'Patient Id'], axis=1)
y = data['Level'].apply(lambda x: {'Low': 0, 'Medium': 1, 'High': 2}[x])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the symptom checker model
@st.cache_resource
def train_symptom_model():
    model = RandomForestClassifier()
    model.fit(X_train[symptom_features], y_train)
    return model

symptom_model = train_symptom_model()

# Train the main prediction models
@st.cache_resource
def train_model(model_name, hyperparameters=None):
    # Initialize the selected model
    if model_name == 'Random Forest':
        model = RandomForestClassifier(**hyperparameters)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(**hyperparameters)
    elif model_name == 'Support Vector Machine (SVM)':
        model = SVC(**hyperparameters, probability=True)
    else:
        st.error("Invalid model name selected.")
        return None, None, None, None

    # Scale the features (for all models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the selected model using scaled data
    model.fit(X_train_scaled, y_train)

    # Return the model, scaler, and scaled datasets
    return model, scaler, X_train_scaled, X_test_scaled

# Initialize session state variables
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar navigation
# Add the logo at the top of the sidebar
st.sidebar.image('logo.png', width=300)  # Adjust the file path and width as needed

# Improved Sidebar for navigation
st.sidebar.title("PulmoCheck 🫁")
st.sidebar.write("**Breathe easier by understanding your lung health.** 🌿")
st.sidebar.markdown("---")  # Divider

# App introduction
st.sidebar.header("Introduction")
st.sidebar.write(
    """
    Welcome to **PulmoCheck**!
    This app provides a comprehensive analysis of lung cancer risk factors, 
    predictive modeling based on symptoms, and detailed visualizations to help 
    you better understand your lung health.
    """
)

st.sidebar.markdown("---")  # Divider


# Navigation options
app_mode = st.sidebar.radio(
    "Navigate the App",
    ["Data Exploration", "Predictive Modeling", "Risk Factor Analysis", "Patient Profile Viewer", "Symptom Checker"]
)

st.sidebar.markdown("---")  # Divider

# Additional resources or information
st.sidebar.header("About")
st.sidebar.write(
    """
    This tool is designed for educational purposes and should not replace professional medical advice. 
    Always consult your healthcare provider for concerns about your health.
    """
)

st.sidebar.markdown("---")  # Divider

# Contact or feedback section
st.sidebar.write("Created by Abdul. For feedback, contact: [hanzo7n@gmail.com](mailto:hanzo7n@gmail.com)")

# Store user inputs for predictions
def store_user_input():
    for col in X.columns:
        st.session_state.user_input[col] = st.number_input(
            f"Enter {col}:", 
            min_value=int(data[col].min()), 
            max_value=int(data[col].max()), 
            value=st.session_state.user_input.get(col, int(data[col].min()))
        )

if app_mode == "Data Exploration":
    st.title("Exploratory Data Analysis")
    st.write("In this section, you can explore the dataset attributes related to lung cancer. Visualize the data through histograms, bar plots, and correlation heatmaps to identify patterns, trends, and relationships among various features. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link), specifically from the 'Lung Cancer Prediction: Air Pollution, Alcohol, Smoking & Risk of Lung Cancer' dataset.")
    
    # Display dataset
    with st.expander("Show Raw Data"):
        st.write(data)

    # Exclude certain columns from the histogram selection
    exclude_columns = ['Patient Id', 'Level', 'Age', 'Gender', 'index']
    available_columns = [col for col in data.columns if col not in exclude_columns]

    # Display histograms for selected columns
    st.write("Histogram of Various Attributes")
    selected_columns = st.multiselect("Select columns to display", available_columns)
    if selected_columns:
        data[selected_columns].hist(bins=15, figsize=(15, 10))
        st.pyplot(plt)

    # Display bar plots for categorical features
    st.write("Bar Plots for Categorical Features")
    categorical_features = ['Gender', 'Genetic Risk', 'chronic Lung Disease']
    selected_cat_feature = st.selectbox("Select a categorical feature to visualize", categorical_features)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=selected_cat_feature, data=data, hue='Level')
    st.pyplot(plt)

if app_mode == "Predictive Modeling":
    st.title("Predictive Modeling for Lung Cancer Risk")
    st.write(
        "This section allows you to train machine learning models to predict lung cancer risk levels. "
        "Choose from different models, fine-tune hyperparameters, and evaluate their performance based on "
        "accuracy, cross-validation scores, and other metrics."
    )

    model_name = st.selectbox(
        "Select Model", ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)"]
    )

    # Hyperparameter selection based on the chosen model
    if model_name == 'Random Forest':
        n_estimators = st.slider('Number of Trees', 10, 200, 100)
        max_depth = st.slider('Max Depth', 1, 20, 10)
        hyperparameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
    elif model_name == 'Logistic Regression':
        C = st.slider('Regularization (C)', 0.01, 10.0, 1.0)
        max_iter = st.slider('Max Iterations', 100, 1000, 300)
        hyperparameters = {'C': C, 'max_iter': max_iter}
    else:  # SVM
        C = st.slider('Regularization Parameter (C)', 0.01, 10.0, 1.0)
        kernel = st.selectbox('Kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        hyperparameters = {'C': C, 'kernel': kernel}

    # Train the model and get the scaler and scaled datasets
    model, scaler, X_train_scaled, X_test_scaled = train_model(model_name, hyperparameters)

    if model:
        st.write("### Cross-Validation Scores:")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        st.write(f"Mean Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

        # Store user inputs
        store_user_input()

        if st.button("Predict"):
            # Use the scaler to transform user input data
            user_input_df = pd.DataFrame([st.session_state.user_input])
            user_input_scaled = scaler.transform(user_input_df)  # Scale the input data
            prediction = model.predict(user_input_scaled)
            st.session_state.prediction_result = ['Low', 'Medium', 'High'][prediction[0]]

        # Display prediction result
        if st.session_state.prediction_result:
            st.write(f"### Predicted Risk Level: {st.session_state.prediction_result}")

        # Model performance evaluation
        y_pred = model.predict(X_test_scaled)
        test_accuracy = model.score(X_test_scaled, y_test)
        st.write(f"Model Accuracy on Test Data: **{test_accuracy * 100:.2f}%**")

        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues')
        st.pyplot(plt)

        # ROC Curve
        st.write("### ROC Curve:")
        y_prob = (
            model.predict_proba(X_test_scaled)[:, 1]
            if hasattr(model, 'predict_proba')
            else model.decision_function(X_test_scaled)
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=2)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        st.pyplot(plt)

elif app_mode == "Risk Factor Analysis":
    st.title("Risk Factor Analysis")
    st.write("Here, you can analyze the impact of different risk factors on lung cancer severity. Visualize how each factor contributes to the risk level and identify the most significant predictors of lung health.")

    selected_factor = st.selectbox("Select a factor to analyze", X.columns)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Level', y=selected_factor, data=data)
    st.pyplot(plt)

elif app_mode == "Patient Profile Viewer":
    st.title("Patient Profile Viewer")
    st.write("View detailed profiles of individual patients, including their symptoms, risk factors, and health status. Use visualizations like radar charts to gain deeper insights into each patient's condition.")

    patient_id = st.selectbox("Select Patient ID", data['Patient Id'].unique())

    if patient_id:
        patient_data = data[data['Patient Id'] == patient_id]
        st.write(patient_data)
        
        # Radar Chart for Symptom Severity
        symptom_features = ['Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty']
        patient_symptoms = patient_data[symptom_features].iloc[0]
            
        # Create radar chart
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        categories = list(patient_symptoms.index)
        values = patient_symptoms.values.flatten().tolist()
        values += values[:1]  # To close the radar chart
            
        ax.plot(range(len(categories) + 1), values, marker='o')
        ax.fill(range(len(categories) + 1), values, alpha=0.3)
        ax.set_yticklabels([])
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title("Symptom Severity Radar Chart", fontsize=14)
        st.pyplot(fig)
        
        # Generate and download patient report
        report_content = generate_patient_report(patient_data)
        st.download_button(
            label="Download Patient Report",
            data=report_content,
            file_name="Patient_Profile_Report.txt",
            mime="text/plain"
        )
elif app_mode == "Symptom Checker":
    st.title("Symptom Checker")
    st.write("Input symptoms to check potential lung cancer risk levels. Use this interactive tool to understand the severity of your symptoms and get a predicted risk assessment.")

    # User input for symptoms
    user_symptoms = {col: st.slider(f"Select severity for {col}:", 0, 10, 0) for col in symptom_features}
    symptoms_df = pd.DataFrame(list(user_symptoms.items()), columns=['Symptom', 'Severity'])
    st.bar_chart(symptoms_df.set_index('Symptom'))

    # Predict risk level
    symptoms_input_df = pd.DataFrame([user_symptoms])
    symptom_risk = symptom_model.predict(symptoms_input_df)
    predicted_level = ['Low', 'Medium', 'High'][symptom_risk[0]]

    st.metric(label="Risk Level", value=predicted_level)

    # Visualization: Correlation heatmap for selected symptoms
    st.write("Correlation Heatmap of Selected Symptoms with Lung Cancer Levels")

    # Encode 'Level' column as numeric
    encoded_data = data.copy()
    encoded_data['Level'] = encoded_data['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Compute and display heatmap for correlation
    symptom_data = encoded_data[symptom_features + ['Level']]
    plt.figure(figsize=(8, 6))
    sns.heatmap(symptom_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Visualization: Distribution of risk levels for selected symptoms
    st.write("Distribution of Risk Levels Based on Selected Symptoms")
    plt.figure(figsize=(10, 6))
    for symptom in symptom_features:
        sns.kdeplot(data=encoded_data, x=symptom, hue='Level', fill=True, common_norm=False)
    plt.title("Risk Level Distributions for Selected Symptoms")
    plt.xlabel("Symptom Severity")
    plt.ylabel("Density")
    st.pyplot(plt)

    # Generate and download symptom report
    def generate_symptom_report(user_symptoms, predicted_level):
        # Create a buffer for the symptom checker report
        symptom_buffer = io.StringIO()
        
        # Write user inputs and predictions
        symptom_buffer.write("Symptom Checker Report\n")
        symptom_buffer.write(f"Predicted Risk Level: {predicted_level}\n")
        symptom_buffer.write("\nUser-Entered Symptoms Severity:\n")
        for symptom, severity in user_symptoms.items():
            symptom_buffer.write(f"{symptom}: {severity}\n")
        
        # Get the report content
        symptom_report_content = symptom_buffer.getvalue()
        symptom_buffer.close()
        return symptom_report_content

    # Download button for symptom checker report
    symptom_report_content = generate_symptom_report(user_symptoms, predicted_level)
    st.download_button(label="Download Symptom Checker Report", data=symptom_report_content, file_name="Symptom_Checker_Report.txt", mime="text/plain")