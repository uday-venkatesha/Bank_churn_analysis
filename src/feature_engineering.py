"""
Advanced Feature Engineering Module for Banking Analytics
This module creates enterprise-grade features used in real banking ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from scipy import stats

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'database': 'pnc_banking',
    'user': 'postgres',
    'password': 'your_password_here'
}


class BankingFeatureEngineer:
    """
    Feature engineering class for creating banking-specific features
    """
    
    def __init__(self, df):
        """
        Initialize with base dataframe from database
        
        Parameters:
        -----------
        df : pandas DataFrame
            Base customer/account data
        """
        self.df = df.copy()
        self.reference_date = datetime.now().date()
        
    def create_all_features(self):
        """
        Master function to create all feature categories
        """
        print("="*70)
        print("BANKING FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        print("\n1. Creating temporal features...")
        self._create_temporal_features()
        
        print("2. Creating account health features...")
        self._create_account_health_features()
        
        print("3. Creating relationship depth features...")
        self._create_relationship_features()
        
        print("4. Creating behavioral features...")
        self._create_behavioral_features()
        
        print("5. Creating risk indicators...")
        self._create_risk_features()
        
        print("6. Creating RFM features...")
        self._create_rfm_features()
        
        print("7. Creating derived banking metrics...")
        self._create_banking_metrics()
        
        print("\n" + "="*70)
        print(f"✓ Feature engineering complete!")
        print(f"✓ Original features: {len(self.df.columns) - self._count_new_features()}")
        print(f"✓ New features created: {self._count_new_features()}")
        print(f"✓ Total features: {len(self.df.columns)}")
        print("="*70)
        
        return self.df
    
    def _create_temporal_features(self):
        """
        Create time-based features critical for banking analysis
        """
        # Ensure date columns are datetime
        if 'join_date' in self.df.columns:
            self.df['join_date'] = pd.to_datetime(self.df['join_date'])
        if 'open_date' in self.df.columns:
            self.df['open_date'] = pd.to_datetime(self.df['open_date'])
        if 'churn_date' in self.df.columns:
            self.df['churn_date'] = pd.to_datetime(self.df['churn_date'])
        
        # Customer tenure (critical feature)
        self.df['tenure_days'] = (pd.Timestamp(self.reference_date) - self.df['join_date']).dt.days
        self.df['tenure_months'] = self.df['tenure_days'] / 30.44
        self.df['tenure_years'] = self.df['tenure_days'] / 365.25
        
        # Tenure buckets (helps with non-linear relationships)
        self.df['tenure_bucket'] = pd.cut(
            self.df['tenure_months'],
            bins=[0, 3, 6, 12, 24, 36, 60, float('inf')],
            labels=['0-3m', '3-6m', '6-12m', '1-2y', '2-3y', '3-5y', '5y+']
        )
        
        # Is new customer? (first 90 days)
        self.df['is_new_customer'] = (self.df['tenure_days'] <= 90).astype(int)
        
        # Seasonality features
        self.df['join_month'] = self.df['join_date'].dt.month
        self.df['join_quarter'] = self.df['join_date'].dt.quarter
        self.df['join_day_of_week'] = self.df['join_date'].dt.dayofweek
        self.df['joined_in_q4'] = (self.df['join_quarter'] == 4).astype(int)  # Holiday season
        
        # Account age vs customer tenure
        if 'open_date' in self.df.columns:
            self.df['account_age_days'] = (pd.Timestamp(self.reference_date) - self.df['open_date']).dt.days
            self.df['days_between_join_and_account'] = (self.df['open_date'] - self.df['join_date']).dt.days
        
        print(f"   ✓ Created {8} temporal features")
    
    def _create_account_health_features(self):
        """
        Features indicating account health and activity level
        """
        # Balance-based features
        if 'balance' in self.df.columns:
            # Balance statistics by customer
            balance_stats = self.df.groupby('customer_id')['balance'].agg([
                ('avg_balance', 'mean'),
                ('total_balance', 'sum'),
                ('min_balance', 'min'),
                ('max_balance', 'max'),
                ('balance_std', 'std')
            ]).reset_index()
            
            self.df = self.df.merge(balance_stats, on='customer_id', how='left')
            
            # Balance health indicators
            self.df['balance_volatility'] = self.df['balance_std'] / (self.df['avg_balance'] + 1)
            self.df['balance_range'] = self.df['max_balance'] - self.df['min_balance']
            
            # Low balance flag (risk indicator)
            self.df['has_low_balance'] = (self.df['balance'] < 1000).astype(int)
            self.df['has_very_low_balance'] = (self.df['balance'] < 500).astype(int)
            
            # Balance categories
            self.df['balance_category'] = pd.cut(
                self.df['balance'],
                bins=[0, 500, 1000, 5000, 10000, 25000, float('inf')],
                labels=['very_low', 'low', 'medium', 'medium_high', 'high', 'very_high']
            )
            
            # Balance percentile (relative standing)
            self.df['balance_percentile'] = self.df['balance'].rank(pct=True)
            
            # Balance growth proxy (current vs average)
            self.df['balance_vs_avg_ratio'] = self.df['balance'] / (self.df['avg_balance'] + 1)
        
        print(f"   ✓ Created {11} account health features")
    
    def _create_relationship_features(self):
        """
        Features measuring depth of customer relationship with bank
        """
        # Number of accounts per customer
        accounts_per_customer = self.df.groupby('customer_id').size().reset_index(name='num_accounts')
        self.df = self.df.merge(accounts_per_customer, on='customer_id', how='left')
        
        # Account type diversity
        if 'account_type' in self.df.columns:
            account_types = self.df.groupby('customer_id')['account_type'].agg([
                ('unique_account_types', 'nunique'),
                ('primary_account_type', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            ]).reset_index()
            
            self.df = self.df.merge(account_types, on='customer_id', how='left')
            
            # Has multiple product types (cross-sell indicator)
            self.df['has_multiple_products'] = (self.df['unique_account_types'] > 1).astype(int)
            
            # Specific account type flags
            account_dummies = pd.get_dummies(self.df['account_type'], prefix='has_account')
            customer_account_types = account_dummies.groupby(self.df['customer_id']).max()
            self.df = self.df.merge(customer_account_types, on='customer_id', how='left')
        
        # Loan relationship
        if 'loan_id' in self.df.columns:
            self.df['has_loan'] = self.df['loan_id'].notna().astype(int)
            
            # Loan details for customers with loans
            if 'loan_amount' in self.df.columns:
                loan_stats = self.df[self.df['has_loan'] == 1].groupby('customer_id').agg({
                    'loan_amount': ['sum', 'mean', 'count']
                })
                loan_stats.columns = ['total_loan_amount', 'avg_loan_amount', 'num_loans']
                self.df = self.df.merge(loan_stats, on='customer_id', how='left')
                self.df[['total_loan_amount', 'avg_loan_amount', 'num_loans']] = \
                    self.df[['total_loan_amount', 'avg_loan_amount', 'num_loans']].fillna(0)
        
        # Relationship depth score (composite)
        relationship_components = []
        if 'num_accounts' in self.df.columns:
            relationship_components.append(np.log1p(self.df['num_accounts']))
        if 'total_balance' in self.df.columns:
            relationship_components.append(np.log1p(self.df['total_balance']) / 10)
        if 'has_loan' in self.df.columns:
            relationship_components.append(self.df['has_loan'] * 2)
        
        if relationship_components:
            self.df['relationship_depth_score'] = sum(relationship_components)
        
        print(f"   ✓ Created {10+} relationship features")
    
    def _create_behavioral_features(self):
        """
        Features capturing customer behavior patterns
        """
        # Age-based behavioral segments
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            
            self.df['is_millennial'] = self.df['age'].between(27, 42).astype(int)
            self.df['is_gen_z'] = (self.df['age'] < 27).astype(int)
            self.df['is_boomer'] = (self.df['age'] >= 59).astype(int)
            self.df['is_senior'] = (self.df['age'] >= 65).astype(int)
        
        # Geographic features
        if 'city' in self.df.columns:
            # City frequency (urban vs rural proxy)
            city_counts = self.df['city'].value_counts()
            self.df['city_customer_count'] = self.df['city'].map(city_counts)
            self.df['is_major_market'] = (self.df['city_customer_count'] > self.df['city_customer_count'].median()).astype(int)
        
        # Gender-based features
        if 'gender' in self.df.columns:
            gender_dummies = pd.get_dummies(self.df['gender'], prefix='gender')
            self.df = pd.concat([self.df, gender_dummies], axis=1)
        
        print(f"   ✓ Created {8+} behavioral features")
    
    def _create_risk_features(self):
        """
        Features indicating financial risk
        """
        # Loan-related risk
        if 'loan_status' in self.df.columns:
            self.df['has_active_loan'] = (self.df['loan_status'] == 'Active').astype(int)
            self.df['has_defaulted_loan'] = (self.df['loan_status'] == 'Defaulted').astype(int)
            self.df['has_paid_off_loan'] = (self.df['loan_status'] == 'Paid Off').astype(int)
        
        # Loan to balance ratio (key risk metric)
        if 'loan_amount' in self.df.columns and 'balance' in self.df.columns:
            self.df['loan_to_balance_ratio'] = self.df['loan_amount'] / (self.df['balance'] + 1)
            self.df['high_loan_to_balance'] = (self.df['loan_to_balance_ratio'] > 5).astype(int)
        
        # Interest rate risk
        if 'interest_rate' in self.df.columns:
            self.df['high_interest_rate'] = (self.df['interest_rate'] > 10).astype(int)
            self.df['interest_rate_percentile'] = self.df['interest_rate'].rank(pct=True)
        
        # Loan term features
        if 'loan_term_months' in self.df.columns:
            self.df['long_term_loan'] = (self.df['loan_term_months'] >= 60).astype(int)
            self.df['short_term_loan'] = (self.df['loan_term_months'] <= 24).astype(int)
        
        # Composite risk score
        risk_components = []
        if 'has_low_balance' in self.df.columns:
            risk_components.append(self.df['has_low_balance'] * 2)
        if 'has_defaulted_loan' in self.df.columns:
            risk_components.append(self.df['has_defaulted_loan'] * 3)
        if 'is_new_customer' in self.df.columns:
            risk_components.append(self.df['is_new_customer'] * 1)
        if 'high_loan_to_balance' in self.df.columns:
            risk_components.append(self.df['high_loan_to_balance'] * 2)
        
        if risk_components:
            self.df['composite_risk_score'] = sum(risk_components)
            self.df['risk_category'] = pd.cut(
                self.df['composite_risk_score'],
                bins=[-1, 0, 2, 4, 100],
                labels=['low_risk', 'medium_risk', 'high_risk', 'very_high_risk']
            )
        
        print(f"   ✓ Created {10+} risk features")
    
    def _create_rfm_features(self):
        """
        Recency, Frequency, Monetary features (key for segmentation)
        """
        # Recency: Days since customer joined (already have tenure)
        self.df['recency_score'] = pd.qcut(
            self.df['tenure_days'],
            q=5,
            labels=[5, 4, 3, 2, 1],  # More recent = higher score
            duplicates='drop'
        ).astype(int)
        
        # Frequency: Number of accounts (proxy for engagement)
        if 'num_accounts' in self.df.columns:
            self.df['frequency_score'] = pd.qcut(
                self.df['num_accounts'],
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        
        # Monetary: Total balance
        if 'total_balance' in self.df.columns:
            self.df['monetary_score'] = pd.qcut(
                self.df['total_balance'],
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        
        # RFM composite score
        if all(col in self.df.columns for col in ['recency_score', 'frequency_score', 'monetary_score']):
            self.df['rfm_score'] = (
                self.df['recency_score'] * 100 +
                self.df['frequency_score'] * 10 +
                self.df['monetary_score']
            )
            
            # RFM segment labels
            self.df['rfm_segment'] = self.df['rfm_score'].apply(self._assign_rfm_segment)
        
        print(f"   ✓ Created {5} RFM features")
    
    def _assign_rfm_segment(self, score):
        """Assign customer segment based on RFM score"""
        if score >= 444:
            return 'Champions'
        elif score >= 434:
            return 'Loyal'
        elif score >= 344:
            return 'Potential_Loyalist'
        elif score >= 334:
            return 'Recent_Customers'
        elif score >= 313:
            return 'Promising'
        elif score >= 244:
            return 'Needs_Attention'
        elif score >= 233:
            return 'About_To_Sleep'
        elif score >= 144:
            return 'At_Risk'
        elif score >= 133:
            return 'Cant_Lose'
        else:
            return 'Hibernating'
    
    def _create_banking_metrics(self):
        """
        Create industry-standard banking metrics
        """
        # Customer Lifetime Value (CLV) proxy
        if 'total_balance' in self.df.columns and 'tenure_months' in self.df.columns:
            # Simple CLV = (Monthly Value × Average Lifespan)
            avg_monthly_value = self.df['total_balance'] * 0.02  # Assume 2% monthly revenue
            expected_lifespan = 36  # months
            self.df['estimated_clv'] = avg_monthly_value * expected_lifespan
        
        # Share of Wallet proxy
        if 'total_balance' in self.df.columns:
            # Assuming total market wallet is 3x what they have with us
            self.df['share_of_wallet_estimate'] = np.minimum(
                self.df['total_balance'] / (self.df['total_balance'] * 3),
                1.0
            )
        
        # Primary bank indicator
        if 'total_balance' in self.df.columns:
            median_balance = self.df['total_balance'].median()
            self.df['likely_primary_bank'] = (self.df['total_balance'] > median_balance * 1.5).astype(int)
        
        # Account concentration (diversification)
        if 'unique_account_types' in self.df.columns and 'num_accounts' in self.df.columns:
            self.df['account_concentration'] = self.df['unique_account_types'] / (self.df['num_accounts'] + 1)
        
        # Value tier (for segmentation)
        if 'total_balance' in self.df.columns:
            self.df['value_tier'] = pd.qcut(
                self.df['total_balance'],
                q=4,
                labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
                duplicates='drop'
            )
        
        print(f"   ✓ Created {5} banking metric features")
    
    def _count_new_features(self):
        """Count features created by this pipeline"""
        original_features = ['customer_id', 'first_name', 'last_name', 'age', 'gender', 
                           'city', 'join_date', 'account_id', 'account_type', 'balance',
                           'open_date', 'loan_id', 'loan_amount', 'interest_rate',
                           'loan_term_months', 'loan_status', 'churn_status', 'churn_date']
        return len(self.df.columns) - len([col for col in original_features if col in self.df.columns])
    
    def get_feature_importance_ready_df(self):
        """
        Return dataframe ready for machine learning
        (numeric only, no missing values)
        """
        # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        
        # Fill missing values
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        print(f"\n✓ ML-ready dataframe created with {len(numeric_df.columns)} features")
        return numeric_df
    
    def save_features(self, output_path='data/customer_features.csv'):
        """Save engineered features to CSV"""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to {output_path}")


def load_data_from_db():
    """
    Load data from PostgreSQL database
    """
    print("Loading data from PostgreSQL...")
    
    query = """
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.age,
        c.gender,
        c.city,
        c.join_date,
        a.account_id,
        a.account_type,
        a.balance,
        a.open_date,
        l.loan_id,
        l.loan_amount,
        l.interest_rate,
        l.loan_term_months,
        l.loan_status,
        ch.churn_status,
        ch.churn_date
    FROM customers c
    LEFT JOIN accounts a ON c.customer_id = a.customer_id
    LEFT JOIN loans l ON c.customer_id = l.customer_id
    LEFT JOIN churn ch ON c.customer_id = ch.customer_id
    ORDER BY c.customer_id;
    """
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"✓ Loaded {len(df)} records from database\n")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def generate_feature_report(df_original, df_engineered):
    """
    Generate a report comparing original vs engineered features
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING REPORT")
    print("="*70)
    
    print(f"\nOriginal Dataset:")
    print(f"  - Records: {len(df_original):,}")
    print(f"  - Features: {len(df_original.columns)}")
    print(f"  - Memory: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nEngineered Dataset:")
    print(f"  - Records: {len(df_engineered):,}")
    print(f"  - Features: {len(df_engineered.columns)}")
    print(f"  - Memory: {df_engineered.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nFeature Categories Created:")
    print(f"  ✓ Temporal Features (tenure, seasonality)")
    print(f"  ✓ Account Health Features (balance metrics, volatility)")
    print(f"  ✓ Relationship Features (product holdings, cross-sell)")
    print(f"  ✓ Behavioral Features (demographics, location)")
    print(f"  ✓ Risk Indicators (default risk, credit metrics)")
    print(f"  ✓ RFM Scores (customer segmentation)")
    print(f"  ✓ Banking Metrics (CLV, share of wallet)")
    
    print(f"\nTop 10 Most Important Feature Types:")
    feature_types = {
        'Balance-based': len([c for c in df_engineered.columns if 'balance' in c.lower()]),
        'Tenure-based': len([c for c in df_engineered.columns if 'tenure' in c.lower()]),
        'Risk-based': len([c for c in df_engineered.columns if 'risk' in c.lower()]),
        'Loan-based': len([c for c in df_engineered.columns if 'loan' in c.lower()]),
        'Account-based': len([c for c in df_engineered.columns if 'account' in c.lower()]),
        'RFM-based': len([c for c in df_engineered.columns if 'rfm' in c.lower() or any(x in c.lower() for x in ['recency', 'frequency', 'monetary'])]),
        'Behavioral': len([c for c in df_engineered.columns if 'age' in c.lower() or 'gender' in c.lower()]),
        'Flags': len([c for c in df_engineered.columns if c.lower().startswith('is_') or c.lower().startswith('has_')]),
    }
    
    for feature_type, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {feature_type}: {count} features")
    
    print("\n" + "="*70)


def main():
    """
    Main execution function
    """
    print("\n" + "🏦" * 35)
    print("PNC BANKING - ADVANCED FEATURE ENGINEERING")
    print("🏦" * 35 + "\n")
    
    # Load data from database
    df = load_data_from_db()
    
    if df is None:
        print("❌ Cannot proceed without data. Please check database connection.")
        return
    
    # Create feature engineer instance
    engineer = BankingFeatureEngineer(df)
    
    # Generate all features
    df_engineered = engineer.create_all_features()
    
    # Generate report
    generate_feature_report(df, df_engineered)
    
    # Save features
    engineer.save_features()
    
    # Get ML-ready dataframe
    df_ml_ready = engineer.get_feature_importance_ready_df()
    
    print("\n✅ Feature engineering complete!")
    print("📊 Next steps:")
    print("   1. Use df_engineered for exploratory analysis")
    print("   2. Use df_ml_ready for machine learning models")
    print("   3. Review feature_importance in your models")
    print("   4. Update your Jupyter notebook to use these features\n")
    
    return df_engineered


if __name__ == "__main__":
    df_features = main()
