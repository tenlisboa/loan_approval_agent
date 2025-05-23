# Loan Approval Agent

A machine learning-powered loan approval system with an AI agent interface for loan eligibility assessment.

## Overview

This project combines machine learning and AI agents to create a loan approval system that can:

1. Predict loan approval likelihood using an XGBoost model
2. Provide an interactive AI agent interface for loan eligibility assessment
3. Analyze loan applications based on various factors like income, credit history, and demographics

## Project Structure

- `loan_analyzer.pkl`: Trained XGBoost model for loan approval prediction
- `loan_profiler_agent.ipynb`: Implementation of the AI agent interface using LangGraph
- `xgboost.ipynb`: Notebook containing the model training and feature analysis
- `data/loan_data.csv`: Dataset used for training the loan approval model

## Features

### Machine Learning Model

The system uses an XGBoost classifier to predict loan approval based on the following features:

- Personal information (age, gender, education, income, employment experience)
- Loan details (amount, intent, interest rate, percent of income)
- Credit history (credit score, previous defaults, credit history length)
- Calculated metrics (monthly payment amount, monthly payment as percent of income)

### AI Agent Interface

The system includes an interactive AI agent built with:

- LangGraph for conversation flow management
- LangChain for LLM integration
- OpenAI's GPT-4.1 model for natural language understanding

The agent can:
- Engage in natural conversation with users
- Collect relevant information for loan assessment
- Use tools to verify loan eligibility
- Provide explanations for loan decisions

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - xgboost
  - langchain
  - langgraph
  - openai

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

#### Running the AI Agent

1. Open the `loan_profiler_agent.ipynb` notebook
2. Run all cells to initialize the agent
3. Interact with the agent through the provided interface

#### Using the ML Model Directly

The trained model is saved as `loan_analyzer.pkl` and can be loaded for direct predictions:

```python
import pickle

# Load the model
with open('loan_analyzer.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(input_data)
```

## Model Performance

The XGBoost model achieves high accuracy in predicting loan approvals. Key features influencing the model's decisions include:

- Previous loan defaults (most important)
- Monthly loan payment as percent of income
- Home ownership status
- Credit score
- Loan amount

## Ethics and Bias Considerations

This project includes a simplified loan approval model for demonstration purposes. The model in `loan_profiler_agent.ipynb` contains a deliberately biased example that approves loans based on gender and income thresholds. This is included only to demonstrate how bias can exist in decision systems and should NOT be used in real applications.

In production systems, care must be taken to:
- Identify and mitigate algorithmic bias
- Ensure fair lending practices
- Comply with relevant regulations
- Provide transparency in decision-making

## License

This project is available for educational and research purposes.

## Acknowledgments

- Dataset used for training the model
- LangChain and LangGraph for agent development
- OpenAI for language model capabilities
