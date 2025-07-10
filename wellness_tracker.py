import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Page Setup ---
st.set_page_config(page_title="Finance Buddy", layout="wide")
st.title("💼 Welcome to Finance Buddy")

# --- Enhanced Model Training ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("cleaned_finance_dataset - cleaned_finance_dataset.csv")
        
        # Feature Engineering
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.weekday
        
        # Handle categorical features
        df['Merchant'] = df['Merchant'].astype(str)
        df['Category'] = df['Category'].astype(str)
        
        # Add unknown category for new merchants/categories
        df = pd.concat([df, pd.DataFrame([{
            'Merchant': 'Unknown', 
            'Category': 'Unknown', 
            'Amount': 0, 
            'Is_Fraud': 0,
            'Day': 1,
            'Month': 1,
            'Weekday': 0
        }])], ignore_index=True)
        
        # Encoding
        le_merchant = LabelEncoder()
        le_category = LabelEncoder()
        df['Merchant'] = le_merchant.fit_transform(df['Merchant'])
        df['Category'] = le_category.fit_transform(df['Category'])
        
        # Feature selection
        features = ['Merchant', 'Amount', 'Category', 'Day', 'Month', 'Weekday']
        X = df[features]
        y = df['Is_Fraud']
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        
        # Model training - Using Random Forest for better performance
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_scaled, y_res)
        
        return model, le_merchant, le_category, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# --- Load Model and Encoders ---
if 'model' not in st.session_state:
    model, le_m, le_c, scaler = load_and_train_model()
    st.session_state.model = model
    st.session_state.encoders = {'Merchant': le_m, 'Category': le_c}
    st.session_state.scaler = scaler

# --- Feature Selection ---
mode = st.radio("Choose a feature", ["🔐 Check for Fraudulent Transactions", "💰 Financial Wellness Tracker"])

# --- Fraud Detection Mode ---
if mode == "🔐 Check for Fraudulent Transactions":
    st.header("📂 Upload Your Transactions CSV")
    
    # Example file download
    example_data = {
        'Date': ['01-01-2025', '02-01-2025'],
        'Merchant': ['Amazon', 'Starbucks'],
        'Amount': [100.50, 45.75],
        'Category': ['Shopping', 'Food & Dining']
    }
    example_df = pd.DataFrame(example_data)
    st.download_button(
        label="⬇️ Download Example CSV",
        data=example_df.to_csv(index=False),
        file_name="example_transactions.csv",
        mime="text/csv"
    )
    
    predict_file = st.file_uploader("Upload your transactions CSV", type=['csv'])
    
    if predict_file:
        try:
            df_new = pd.read_csv(predict_file)
            
            # Validate required columns
            required_cols = ['Date', 'Merchant', 'Amount', 'Category']
            if not all(col in df_new.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_new.columns]
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.stop()
            
            st.subheader("📄 Uploaded Data Sample")
            st.dataframe(df_new.head())
            
            # Feature engineering for new data
            df_new['Date'] = pd.to_datetime(df_new['Date'], dayfirst=True, errors='coerce')
            df_new['Day'] = df_new['Date'].dt.day.fillna(1)
            df_new['Month'] = df_new['Date'].dt.month.fillna(1)
            df_new['Weekday'] = df_new['Date'].dt.weekday.fillna(0)
            
            df_new['Merchant'] = df_new['Merchant'].astype(str)
            df_new['Category'] = df_new['Category'].astype(str)
            
            # Handle unseen categories
            for col in ['Merchant', 'Category']:
                df_new[col] = df_new[col].apply(
                    lambda x: x if x in st.session_state.encoders[col].classes_ else 'Unknown'
                )
                if 'Unknown' not in st.session_state.encoders[col].classes_:
                    st.session_state.encoders[col].classes_ = np.append(
                        st.session_state.encoders[col].classes_, 'Unknown'
                    )
            
            df_new['Merchant'] = st.session_state.encoders['Merchant'].transform(df_new['Merchant'])
            df_new['Category'] = st.session_state.encoders['Category'].transform(df_new['Category'])
            
            # Prepare features
            features = ['Merchant', 'Amount', 'Category', 'Day', 'Month', 'Weekday']
            X_new = df_new[features]
            
            # Scale features
            X_scaled = st.session_state.scaler.transform(X_new)
            
            # Make predictions
            y_pred = st.session_state.model.predict(X_scaled)
            y_prob = st.session_state.model.predict_proba(X_scaled)[:, 1]  # Probability of fraud
            
            df_new['Predicted_Is_Fraud'] = y_pred
            df_new['Fraud_Probability'] = y_prob
            fraud_count = int(np.sum(y_pred))
            
            # Display results
            st.subheader("🔎 Prediction Results")
            st.write(f"🔴 Fraudulent Transactions Detected: **{fraud_count}**")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Fraud Probability Distribution")
                fig1, ax1 = plt.subplots()
                sns.histplot(df_new['Fraud_Probability'], bins=20, kde=True, ax=ax1)
                ax1.set_xlabel('Fraud Probability')
                ax1.set_ylabel('Count')
                st.pyplot(fig1)
            
            with col2:
                st.markdown("### Fraud by Category")
                fraud_by_cat = df_new[df_new['Predicted_Is_Fraud'] == 1]['Category'].value_counts()
                fig2, ax2 = plt.subplots()
                fraud_by_cat.plot(kind='bar', ax=ax2)
                ax2.set_xlabel('Category')
                ax2.set_ylabel('Fraud Count')
                st.pyplot(fig2)
            
            # Detailed results
            st.subheader("Detailed Predictions")
            st.dataframe(df_new.sort_values('Fraud_Probability', ascending=False))
            
            # Download results
            st.download_button(
                "⬇️ Download Results", 
                data=df_new.to_csv(index=False), 
                file_name="fraud_predictions.csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# --- Financial Wellness Mode ---
elif mode == "💰 Financial Wellness Tracker":
    st.header("🔢 Enter Your Monthly Financial Info")

    salary = st.number_input("Monthly Salary (INR)", min_value=0.0, step=500.0, format="%.2f")
    expenses = st.number_input("Monthly Expenses (INR)", min_value=0.0, step=100.0, format="%.2f")
    savings = st.number_input("Monthly Savings (INR)", min_value=0.0, step=100.0, format="%.2f")
    debt = st.number_input("Monthly Debt Payments (INR)", min_value=0.0, step=100.0, format="%.2f")

    def calculate_wellscore(salary, expenses, savings, debt):
        if salary <= 0:
            return "Invalid", "Please enter a valid salary greater than zero.", "⚠️"

        savings_ratio = savings / salary
        expense_ratio = expenses / salary
        debt_ratio = debt / salary

        with st.expander("📊 View Financial Ratios"):
            st.write(f"Savings Ratio: `{savings_ratio:.2f}`")
            st.write(f"Expense Ratio: `{expense_ratio:.2f}`")
            st.write(f"Debt Ratio: `{debt_ratio:.2f}`")

        if savings_ratio >= 0.2 and expense_ratio <= 0.5 and debt_ratio <= 0.3:
           return (
    "Good",
    "✅ Excellent financial health!\nYou're saving wisely and spending within limits.\nYour debt is under control — keep it that way!\nStay consistent and plan ahead for future goals.",
    "🟢"
)

        elif savings_ratio >= 0.1 and expense_ratio <= 0.7 and debt_ratio <= 0.5:
            return (
    "Average",
    "🟡 You're on the right track, but there's room for improvement.\nTry to increase your monthly savings bit by bit.\nKeep a close watch on expenses and cut down non-essentials.\nReducing your debt gradually will boost your financial health.",
    "🟡"
)

        else:
           return (
    "Poor",
    "🔴 Your financial health needs urgent attention.\nYour expenses or debt may be too high compared to your income.\nStart by tracking where your money goes and cut unnecessary costs.\nFocus on building an emergency fund and creating a savings habit.",
    "🔴"
)


    if st.button("📈 Calculate WellScore"):
        score, tip, icon = calculate_wellscore(salary, expenses, savings, debt)
        st.subheader(f"Your WellScore: {icon} **{score}**")
        if score == "Good":
            st.success(tip)
        elif score == "Average":
            st.warning(tip)
        else:
            st.error(tip)

        st.markdown("### 🥧 Financial Breakdown Pie Chart")
        labels = ['Expenses', 'Savings', 'Debt', 'Remaining']
        remaining = max(salary - (expenses + savings + debt), 0)
        values = [expenses, savings, debt, remaining]
        colors = ['#ff9999', '#8fd9b6', '#ffc966', '#c2c2f0']

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 6},
            pctdistance=0.8,
            labeldistance=1.2
        )
        ax.axis('equal')
        st.pyplot(fig, use_container_width=True)