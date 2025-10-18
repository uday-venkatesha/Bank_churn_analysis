# PNC Customer Churn & Loan Default Prediction

## Project Overview
This project demonstrates end-to-end data analytics skills by analyzing customer churn patterns and predicting loan defaults for a fictional banking scenario. The project showcases data generation, database management, exploratory data analysis, machine learning, and business insights extraction.

## Objective
As a data analyst, this project aims to:
- Generate realistic synthetic financial data for 5,000 customers
- Build and populate a PostgreSQL database with customer, account, loan, and churn data
- Perform exploratory data analysis to identify key drivers of customer churn
- Develop a machine learning model to predict loan defaults
- Prepare actionable insights for visualization in a Tableau dashboard

## Project Architecture

### Workflow Phases
1. **Data Simulation**: Generate synthetic banking data using Python and Faker
2. **Database Ingestion**: Load data into PostgreSQL using psycopg2
3. **Analysis & Modeling**: Conduct EDA and build predictive models in Jupyter
4. **Visualization**: Create interactive Tableau dashboard (link below)

### Technology Stack
- **Languages**: Python, SQL
- **Database**: PostgreSQL
- **Libraries**: pandas, numpy, Faker, psycopg2, scikit-learn, matplotlib, seaborn
- **Tools**: Jupyter Notebook, VS Code, Tableau

## Directory Structure
```
pnc_churn_analysis/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── sql/
│   └── schema.sql
│
└── src/
    ├── 1_generate_data.py
    └── 2_churn_analysis_and_modeling.ipynb
```

## How to Run

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Jupyter Notebook

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pnc_churn_analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**
   ```bash
   # Create a new database
   createdb pnc_banking
   
   # Run the schema script
   psql -d pnc_banking -f sql/schema.sql
   ```

5. **Update database credentials**
   - Edit `src/1_generate_data.py` and `src/2_churn_analysis_and_modeling.ipynb`
   - Update the connection parameters with your PostgreSQL credentials

6. **Generate and load data**
   ```bash
   python src/1_generate_data.py
   ```

7. **Run analysis notebook**
   ```bash
   jupyter notebook src/2_churn_analysis_and_modeling.ipynb
   ```

## Key Findings

### Churn Analysis
- Customer tenure and account balance are significant indicators of churn risk
- Specific account types show higher churn rates
- Age demographics reveal distinct churn patterns

### Loan Default Prediction
- Machine learning model achieves strong predictive performance
- Key risk factors identified for loan default
- Model can be used for proactive risk management

## Tableau Dashboard
**[View Interactive Dashboard](#)** *(Link to be added after dashboard creation)*

## Future Enhancements
- Implement additional machine learning algorithms (Random Forest, XGBoost)
- Add time-series analysis for transaction patterns
- Develop customer segmentation using clustering techniques
- Create automated reporting pipeline

## Author
Data Analytics Portfolio Project

## License
MIT License