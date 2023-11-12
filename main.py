# Security
import hashlib

# Image Processing
import base64

# Date and Time Calculations
import datetime

# Reading Files
import os

# Data Analysis
import pandas as pd

# Frontend Development
import streamlit as st

# Customer Transactions Profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# Prediction
from sklearn.model_selection import train_test_split
import sklearn.linear_model

# DB Management
import sqlite3


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text

    return False


conn = sqlite3.connect("data.db")
c = conn.cursor()


# DB  Functions
def create_usertable():
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")


def add_userdata(username, password):
    c.execute("INSERT INTO userstable(username,password) VALUES (?,?)", (username, password))
    conn.commit()


def login_user(username, password):
    c.execute("SELECT * FROM userstable WHERE username =? AND password = ?", (username, password))
    data = c.fetchall()
    return data


def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if BEGIN_DATE + '.pkl' <= f <= END_DATE + '.pkl']
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)

    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    #  Note: -1 are missing values for real world data
    df_final = df_final.replace([-1], 0)

    return df_final


def generate_transaction(start_date="2018-04-01"):
    customer_transaction = []
    customer_id = st.number_input("Customer ID", step=1, min_value=0, max_value=4999)
    terminal_id = st.number_input("Terminal ID", step=1, min_value=0, max_value=9999)
    d = st.date_input("Date", min_value=datetime.date(2018, 10, 1))
    t = st.time_input("Time", step=60)
    sd = datetime.date(2018, 4, 1)
    day = (d - sd).days
    time_tx = (t.hour * 60 + t.minute) * 60 + t.second
    amount = st.number_input("Amount", min_value=1)
    customer_transaction.append([time_tx + day * 86400, day, customer_id, terminal_id, amount])
    customer_transaction = pd.DataFrame(customer_transaction,
                                        columns=["TX_TIME_SECONDS", "TX_TIME_DAYS", "CUSTOMER_ID", "TERMINAL_ID",
                                                 "TX_AMOUNT"])
    transactions_df = pd.read_csv("transactions.csv")
    customer_transaction["TRANSACTION_ID"] = max(transactions_df["TRANSACTION_ID"]) + 1
    customer_transaction["TX_DATETIME"] = pd.to_datetime(customer_transaction["TX_TIME_SECONDS"], unit="s",
                                                         origin=start_date)
    customer_transaction = customer_transaction[
        ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]]
    return customer_transaction


def is_weekend(tx_datetime):
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5

    return int(is_weekend)


def is_night(tx_datetime):
    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour <= 6

    return int(is_night)


def get_customer_spending_behaviour_features(transaction_df, customer_transaction, windows_size_in_days=[1, 7, 30]):
    # Let us first order transactions chronologically
    new_transaction_df = pd.concat([transaction_df, customer_transaction], axis=0, ignore_index=True)
    new_transaction_df = new_transaction_df.sort_values('TX_DATETIME')

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    new_transaction_df.index = new_transaction_df.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = new_transaction_df['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = new_transaction_df['TX_AMOUNT'].rolling(str(window_size) + 'd').count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        customer_transaction['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)[
            len(list(NB_TX_WINDOW)) - 1]
        customer_transaction['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(AVG_AMOUNT_TX_WINDOW)[
            len(list(AVG_AMOUNT_TX_WINDOW)) - 1]

    # Reindex according to transaction IDs
    new_transaction_df.index = new_transaction_df.TRANSACTION_ID

    # And return the dataframe with the new features
    return customer_transaction


def get_count_risk_rolling_window(terminal_transactions, customer_transaction, delay_period=7,
                                  windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"):
    new_terminal_transactions = pd.concat([terminal_transactions, customer_transaction], axis=0, ignore_index=True)
    new_terminal_transactions = new_terminal_transactions.sort_values('TX_DATETIME')

    new_terminal_transactions.index = new_terminal_transactions.TX_DATETIME

    NB_FRAUD_DELAY = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = new_terminal_transactions['TX_FRAUD'].rolling(
            str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = new_terminal_transactions['TX_FRAUD'].rolling(
            str(delay_period + window_size) + 'd').count()

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        customer_transaction[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)[
            len(list(NB_TX_WINDOW)) - 1]
        customer_transaction[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)[
            len(list(RISK_WINDOW)) - 1]

    new_terminal_transactions.index = new_terminal_transactions.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    customer_transaction.fillna(0, inplace=True)

    return customer_transaction


def set_bg_hack(main_bg):
    """
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    """
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def show_customer_profiles():
    data = pd.read_csv("customerProfiles.csv")
    customer_id = st.number_input("Customer ID", value=None, step=1, min_value=0, max_value=4999)
    st.dataframe(data[data.CUSTOMER_ID == customer_id])


def show_terminal_profiles():
    data = pd.read_csv("terminalProfiles.csv")
    terminal_id = st.number_input("Terminal ID", value=None, step=1, min_value=0, max_value=9999)
    st.dataframe(data[data.TERMINAL_ID == terminal_id])


def show_transactions():
    data = pd.read_csv("transactions.csv")
    customer_id = st.number_input("Customer ID", step=1, min_value=0, max_value=4999)
    df = data[data.CUSTOMER_ID == customer_id]
    pr = df.profile_report()
    st_profile_report(pr)
    st.write("All Transactions")
    st.dataframe(df)


def get_customer_transactions(customer_id):
    data = pd.read_csv("transactions.csv")
    return data[data.CUSTOMER_ID == customer_id]


def store_transaction(customer_transaction):
    data = pd.read_csv("transactions.csv")
    new_transaction_df = pd.concat([data, customer_transaction], axis=0, join='inner')
    new_transaction_df.to_csv('transactions.csv', index=False)


def get_blacklist(customer_id):
    customer_df = pd.read_csv('customerProfiles.csv')
    return eval(customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'BLACKLIST'].iloc[0])


def get_frequency(customer_id):
    customer_df = pd.read_csv('customerProfiles.csv')
    return customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'FREQUENCY'].iloc[0]


def add_to_customer_transaction1(customer_transaction):
    transactions_df = read_from_files('tx1', "2018-04-01", "2018-09-30")
    return get_customer_spending_behaviour_features(
        transactions_df[transactions_df.CUSTOMER_ID == customer_transaction.at[0, 'CUSTOMER_ID']],
        customer_transaction)


def add_to_customer_transaction2(customer_transaction):
    transactions_df = read_from_files('tx2', "2018-04-01", "2018-09-30")
    return get_count_risk_rolling_window(
        transactions_df[transactions_df.TERMINAL_ID == customer_transaction.at[0, 'TERMINAL_ID']],
        customer_transaction, delay_period=7, windows_size_in_days=[1, 7, 30])


def make_transaction_and_predict(customer_transaction):
    customer_id = customer_transaction.loc[0, 'CUSTOMER_ID']
    terminal_id = customer_transaction.loc[0, 'TERMINAL_ID']
    blacklist = get_blacklist(customer_id)
    frequency = get_frequency(customer_id)
    df = get_customer_transactions(customer_id)
    pd.options.mode.chained_assignment = None
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    dat = str(customer_transaction.at[0, 'TX_DATETIME'].date())
    count = len(df[df['TX_DATETIME'].dt.date == pd.to_datetime(dat).date()])
    if terminal_id in blacklist:
        st.warning(
            "Terminal {} blacklisted for customer {}, transaction blocked".format(terminal_id, customer_id))

    elif count >= frequency:
        st.warning(
            "Threshold frequency of {} has been reached for customer {} for date {}, transaction blocked".format(
                frequency, customer_id, dat))

    else:
        customer_transaction['TX_FRAUD_SCENARIO'] = 0
        customer_transaction['TX_DURING_WEEKEND'] = customer_transaction.TX_DATETIME.apply(is_weekend)
        customer_transaction['TX_DURING_NIGHT'] = customer_transaction.TX_DATETIME.apply(is_night)
        customer_transaction = add_to_customer_transaction1(customer_transaction)
        customer_transaction = add_to_customer_transaction2(customer_transaction)

        # loading the dataset to a Pandas DataFrame
        credit_card_data = pd.read_csv('creditCard.csv')

        # separating the data for analysis
        legit = credit_card_data[credit_card_data.TX_FRAUD == 0]
        fraud = credit_card_data[credit_card_data.TX_FRAUD == 1]

        # undersample legitimate transactions to balance the classes
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        new_dataset = pd.concat([legit_sample, fraud], axis=0)

        # split data into training and testing sets
        X = new_dataset[['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                         'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                         'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                         'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                         'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                         'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                         'TERMINAL_ID_RISK_30DAY_WINDOW']]
        Y = new_dataset['TX_FRAUD']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # train logistic regression model
        model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
        model.fit(X_train, Y_train)

        # get input feature values
        X_input = customer_transaction[
            ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
             'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
             'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
             'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
             'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
             'TERMINAL_ID_RISK_30DAY_WINDOW']]

        # make prediction
        prediction = model.predict(X_input)

        # display result
        if prediction[0] == 0:
            st.success("Legitimate Transaction")
        else:
            st.warning("Fraudulent Transaction")

        customer_transaction['TX_FRAUD'] = int(prediction[0])
        st.dataframe(customer_transaction)
        store_transaction(customer_transaction)


def blacklist_terminals(customer_id, terminal_id):
    customer_df = pd.read_csv("customerProfiles.csv")
    blacklist_row = eval(customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'BLACKLIST'].iloc[0])
    blacklist_row.extend([int(terminal_id)])
    customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'BLACKLIST'] = customer_df.loc[
        customer_df['CUSTOMER_ID'] == customer_id, 'BLACKLIST'].apply(lambda x: blacklist_row)
    customer_df.to_csv('customerProfiles.csv', index=False)
    st.write("Terminal {} is blacklisted for customer {}".format(terminal_id, customer_id))


def velocity_checks(customer_id, frequency):
    customer_df = pd.read_csv("customerProfiles.csv")
    customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'FREQUENCY'] = customer_df.loc[
        customer_df['CUSTOMER_ID'] == customer_id, 'FREQUENCY'].apply(lambda x: frequency)
    customer_df.to_csv('customerProfiles.csv', index=False)
    st.write("Threshold frequency of customer {} has been updated to {}".format(customer_id, frequency))


def about():
    credit_card_data = pd.read_csv("creditCard.csv")
    csv = convert_df(credit_card_data)
    st.download_button(label="Download", data=csv, file_name='creditCard.csv', mime='text/csv', )


def main():
    set_bg_hack("background.png")
    st.title(":blue[Cred Secure]")
    st.header("A Credit Card Fraud Detection System")
    menu = ["Login", "SignUp", "About", "Contacts"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login Section")

        # Take input for username and password
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type="password")
        st.sidebar.caption("Tick the checkbox to login and untick to logout")

        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                task = st.selectbox("Task", ["Show Customer Profiles", "Show Terminal Profiles", "Show Transactions",
                                             "Make Transaction and Predict", "Blacklist Terminals", "Velocity Checks"])

                if task == "Show Customer Profiles":
                    st.subheader("Customer Profiles")
                    show_customer_profiles()

                elif task == "Show Terminal Profiles":
                    st.subheader("Terminal Profiles")
                    show_terminal_profiles()

                elif task == "Show Transactions":
                    st.subheader("Transactions")
                    show_transactions()

                elif task == "Make Transaction and Predict":
                    st.subheader("Make Transaction and Predict")
                    st.caption("Enter new transaction details")
                    customer_transaction = generate_transaction(start_date="2018-04-01")

                    # create a button to submit input and get prediction
                    submit = st.button("Submit")
                    if submit:
                        make_transaction_and_predict(customer_transaction)

                elif task == "Blacklist Terminals":
                    st.subheader("Blacklist Terminals")
                    customer_id = st.number_input("Customer ID", step=1, min_value=0, max_value=4999)
                    terminal_id = st.number_input("Terminal ID", step=1, min_value=0, max_value=9999)
                    submit = st.button("Submit")

                    if submit:
                        blacklist_terminals(customer_id, terminal_id)

                elif task == "Velocity Checks":
                    st.subheader("Velocity Checks")
                    customer_id = st.number_input("Customer ID", step=1, min_value=0, max_value=4999)
                    frequency = st.number_input("Threshold Frequency", step=1, min_value=0)
                    submit = st.button("Submit")

                    if submit:
                        velocity_checks(customer_id, frequency)

            else:
                st.warning("Incorrect Username/Password")
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type="password")

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

    elif choice == "About":
        st.subheader("About")
        st.write(
            "Credit Card Fraud Detection (CCFD) is like looking for needles in a haystack. It requires finding, out of millions of daily transactions, which ones are fraudulent. Due to the ever-increasing amount of data, it is now almost impossible for a human specialist to detect meaningful patterns from transaction data. For this reason, the use of machine learning techniques is now widespread in the field of fraud detection, where information extraction from large datasets is required.")
        st.write(
            "Machine Learning (ML) is the study of algorithms that improve automatically through experience. ML is closely related to the fields of Statistics, Pattern Recognition, and Data Mining. At the same time, it emerges as a subfield of computer science and artificial intelligence and gives special attention to the algorithmic part of the knowledge extraction process. In this case Logistic Regression Model is used for developing the prediction system for detecting credit card fraud.")
        st.write("Dataset used(Download from below):")
        about()

    elif choice == "Contacts":
        st.subheader("Contacts")
        st.write("Ph. 9932793573")
        st.write("Email: miracle10117@gmail.com")


if __name__ == '__main__':
    main()
