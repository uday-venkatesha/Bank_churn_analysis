-- -- PNC Banking Database Schema
-- -- Drop tables if they exist (for clean setup)
-- DROP TABLE IF EXISTS churn CASCADE;
-- DROP TABLE IF EXISTS loans CASCADE;
-- DROP TABLE IF EXISTS accounts CASCADE;
-- DROP TABLE IF EXISTS customers CASCADE;

-- -- Create customers table
-- CREATE TABLE customers (
--     customer_id SERIAL PRIMARY KEY,
--     first_name VARCHAR(50) NOT NULL,
--     last_name VARCHAR(50) NOT NULL,
--     age INTEGER NOT NULL CHECK (age >= 18 AND age <= 100),
--     gender VARCHAR(10),
--     city VARCHAR(100),
--     join_date DATE NOT NULL
-- );

-- -- Create accounts table
-- CREATE TABLE accounts (
--     account_id SERIAL PRIMARY KEY,
--     customer_id INTEGER NOT NULL,
--     account_type VARCHAR(50) NOT NULL,
--     balance DECIMAL(12, 2) NOT NULL CHECK (balance >= 0),
--     open_date DATE NOT NULL,
--     FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
-- );

-- -- Create loans table
-- CREATE TABLE loans (
--     loan_id SERIAL PRIMARY KEY,
--     customer_id INTEGER NOT NULL,
--     loan_amount DECIMAL(12, 2) NOT NULL CHECK (loan_amount > 0),
--     interest_rate DECIMAL(5, 2) NOT NULL CHECK (interest_rate >= 0),
--     loan_term_months INTEGER NOT NULL CHECK (loan_term_months > 0),
--     loan_status VARCHAR(20) NOT NULL CHECK (loan_status IN ('Active', 'Paid Off', 'Defaulted')),
--     FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
-- );

-- -- Create churn table
-- CREATE TABLE churn (
--     churn_id SERIAL PRIMARY KEY,
--     customer_id INTEGER NOT NULL UNIQUE,
--     churn_status INTEGER NOT NULL CHECK (churn_status IN (0, 1)),
--     churn_date DATE,
--     FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
-- );

-- -- Create indexes for better query performance
-- CREATE INDEX idx_customer_join_date ON customers(join_date);
-- CREATE INDEX idx_accounts_customer ON accounts(customer_id);
-- CREATE INDEX idx_loans_customer ON loans(customer_id);
-- CREATE INDEX idx_churn_customer ON churn(customer_id);
-- CREATE INDEX idx_churn_status ON churn(churn_status);

-- -- Display success message
-- SELECT 'Database schema created successfully!' AS status;

