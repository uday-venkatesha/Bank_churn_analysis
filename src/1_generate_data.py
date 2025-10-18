"""
PNC Banking Data Generation and Database Loading Script
This script generates synthetic banking data and loads it into PostgreSQL
"""

import pandas as pd
import numpy as np
from faker import Faker
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Constants
NUM_CUSTOMERS = 5000
DATA_DIR = 'data'

# Database connection parameters (UPDATE THESE WITH YOUR CREDENTIALS)
DB_PARAMS = {
    'host': os.getenv("host"),
    'database': os.getenv("database"),
    'user': os.getenv("user"),
    'password': os.getenv("password")
}

def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}/ directory")

def generate_customers(num_customers):
    """Generate synthetic customer data"""
    print(f"Generating {num_customers} customers...")
    
    customers = []
    for i in range(1, num_customers + 1):
        customer = {
            'customer_id': i,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'age': np.random.randint(18, 80),
            'gender': np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04]),
            'city': fake.city(),
            'join_date': fake.date_between(start_date='-10y', end_date='today')
        }
        customers.append(customer)
    
    df_customers = pd.DataFrame(customers)
    print(f"✓ Generated {len(df_customers)} customers")
    return df_customers

def generate_accounts(df_customers):
    """Generate synthetic account data"""
    print("Generating accounts...")
    
    accounts = []
    account_types = ['Checking', 'Savings', 'Money Market', 'CD']
    
    for idx, customer in df_customers.iterrows():
        # Each customer has 1-3 accounts
        num_accounts = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        
        for _ in range(num_accounts):
            account = {
                'customer_id': customer['customer_id'],
                'account_type': np.random.choice(account_types),
                'balance': round(np.random.lognormal(8, 2), 2),  # Log-normal distribution
                'open_date': fake.date_between(
                    start_date=customer['join_date'],
                    end_date='today'
                )
            }
            accounts.append(account)
    
    df_accounts = pd.DataFrame(accounts)
    df_accounts['account_id'] = range(1, len(df_accounts) + 1)
    df_accounts = df_accounts[['account_id', 'customer_id', 'account_type', 'balance', 'open_date']]
    
    print(f"✓ Generated {len(df_accounts)} accounts")
    return df_accounts

def generate_loans(df_customers, df_accounts):
    """Generate synthetic loan data with realistic correlations"""
    print("Generating loans...")
    
    loans = []
    
    # Calculate average balance per customer for correlation
    avg_balance = df_accounts.groupby('customer_id')['balance'].mean()
    
    # About 40% of customers have loans
    customers_with_loans = np.random.choice(
        df_customers['customer_id'].values,
        size=int(NUM_CUSTOMERS * 0.4),
        replace=False
    )
    
    for customer_id in customers_with_loans:
        balance = avg_balance.get(customer_id, 5000)
        
        # Customers with lower balances have slightly higher default probability
        if balance < 2000:
            default_prob = 0.25
        elif balance < 5000:
            default_prob = 0.15
        else:
            default_prob = 0.08
        
        loan = {
            'customer_id': customer_id,
            'loan_amount': round(np.random.uniform(5000, 100000), 2),
            'interest_rate': round(np.random.uniform(3.5, 12.5), 2),
            'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 84, 120]),
            'loan_status': np.random.choice(
                ['Active', 'Paid Off', 'Defaulted'],
                p=[0.5, 0.5 - default_prob, default_prob]
            )
        }
        loans.append(loan)
    
    df_loans = pd.DataFrame(loans)
    df_loans['loan_id'] = range(1, len(df_loans) + 1)
    df_loans = df_loans[['loan_id', 'customer_id', 'loan_amount', 'interest_rate', 
                          'loan_term_months', 'loan_status']]
    
    print(f"✓ Generated {len(df_loans)} loans")
    return df_loans

def generate_churn(df_customers, df_accounts):
    """Generate churn data with realistic correlations"""
    print("Generating churn data...")
    
    churn_data = []
    
    # Calculate metrics for each customer
    avg_balance = df_accounts.groupby('customer_id')['balance'].mean()
    
    for idx, customer in df_customers.iterrows():
        customer_id = customer['customer_id']
        balance = avg_balance.get(customer_id, 5000)
        
        # Calculate tenure in days
        tenure_days = (datetime.now().date() - customer['join_date']).days
        
        # Churn probability based on balance and tenure
        base_churn_prob = 0.15
        
        if balance < 1000:
            base_churn_prob += 0.15
        elif balance < 3000:
            base_churn_prob += 0.08
        
        if tenure_days < 180:  # Less than 6 months
            base_churn_prob += 0.20
        elif tenure_days < 365:  # Less than 1 year
            base_churn_prob += 0.10
        
        churn_status = np.random.binomial(1, min(base_churn_prob, 0.6))
        
        churn = {
            'customer_id': customer_id,
            'churn_status': churn_status,
            'churn_date': fake.date_between(
                start_date=customer['join_date'],
                end_date='today'
            ) if churn_status == 1 else None
        }
        churn_data.append(churn)
    
    df_churn = pd.DataFrame(churn_data)
    df_churn['churn_id'] = range(1, len(df_churn) + 1)
    df_churn = df_churn[['churn_id', 'customer_id', 'churn_status', 'churn_date']]
    
    print(f"✓ Generated churn data for {len(df_churn)} customers")
    print(f"  - Churned customers: {df_churn['churn_status'].sum()}")
    return df_churn

def save_dataframes_to_csv(df_customers, df_accounts, df_loans, df_churn):
    """Save all dataframes to CSV files"""
    print("\nSaving data to CSV files...")
    
    df_customers.to_csv(f'{DATA_DIR}/customers.csv', index=False)
    print(f"✓ Saved customers.csv")
    
    df_accounts.to_csv(f'{DATA_DIR}/accounts.csv', index=False)
    print(f"✓ Saved accounts.csv")
    
    df_loans.to_csv(f'{DATA_DIR}/loans.csv', index=False)
    print(f"✓ Saved loans.csv")
    
    df_churn.to_csv(f'{DATA_DIR}/churn.csv', index=False)
    print(f"✓ Saved churn.csv")

def load_data_to_postgres():
    """Load CSV data into PostgreSQL database"""
    print("\n" + "="*60)
    print("LOADING DATA INTO POSTGRESQL DATABASE")
    print("="*60)
    
    try:
        # Connect to database
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        print("✓ Connected successfully")
        
        # Clear existing data (in correct order due to foreign keys)
        print("\nClearing existing data...")
        cur.execute("TRUNCATE TABLE churn, loans, accounts, customers RESTART IDENTITY CASCADE;")
        conn.commit()
        print("✓ Cleared existing data")
        
        # Load customers
        print("\nLoading customers...")
        with open(f'{DATA_DIR}/customers.csv', 'r') as f:
            next(f)  # Skip header
            cur.copy_from(f, 'customers', sep=',', 
                         columns=['customer_id', 'first_name', 'last_name', 
                                 'age', 'gender', 'city', 'join_date'])
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM customers;")
        count = cur.fetchone()[0]
        print(f"✓ Successfully loaded {count} customers into database")
        
        # Load accounts
        print("\nLoading accounts...")
        with open(f'{DATA_DIR}/accounts.csv', 'r') as f:
            next(f)  # Skip header
            cur.copy_from(f, 'accounts', sep=',',
                         columns=['account_id', 'customer_id', 'account_type', 
                                 'balance', 'open_date'])
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM accounts;")
        count = cur.fetchone()[0]
        print(f"✓ Successfully loaded {count} accounts into database")
        
        # Load loans
        print("\nLoading loans...")
        with open(f'{DATA_DIR}/loans.csv', 'r') as f:
            next(f)  # Skip header
            cur.copy_from(f, 'loans', sep=',',
                         columns=['loan_id', 'customer_id', 'loan_amount', 
                                 'interest_rate', 'loan_term_months', 'loan_status'])
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM loans;")
        count = cur.fetchone()[0]
        print(f"✓ Successfully loaded {count} loans into database")
        
        # Load churn
        print("\nLoading churn data...")
        # Handle NULL values in churn_date
        df_churn_load = pd.read_csv(f'{DATA_DIR}/churn.csv')
        for _, row in df_churn_load.iterrows():
            cur.execute("""
                INSERT INTO churn (churn_id, customer_id, churn_status, churn_date)
                VALUES (%s, %s, %s, %s)
            """, (row['churn_id'], row['customer_id'], row['churn_status'], 
                  row['churn_date'] if pd.notna(row['churn_date']) else None))
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM churn;")
        count = cur.fetchone()[0]
        print(f"✓ Successfully loaded {count} churn records into database")
        
        # Close connection
        cur.close()
        conn.close()
        print("\n" + "="*60)
        print("DATABASE LOADING COMPLETE!")
        print("="*60)
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print("  2. Database 'pnc_banking' exists")
        print("  3. Database credentials in DB_PARAMS are correct")
        print("  4. Schema has been created (run sql/schema.sql)")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main execution function"""
    print("="*60)
    print("PNC BANKING DATA GENERATION")
    print("="*60 + "\n")
    
    # Create data directory
    create_data_directory()
    
    # Generate data
    df_customers = generate_customers(NUM_CUSTOMERS)
    df_accounts = generate_accounts(df_customers)
    df_loans = generate_loans(df_customers, df_accounts)
    df_churn = generate_churn(df_customers, df_accounts)
    
    # Save to CSV
    save_dataframes_to_csv(df_customers, df_accounts, df_loans, df_churn)
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    
    # Load to database
    load_data_to_postgres()

if __name__ == "__main__":
    main()