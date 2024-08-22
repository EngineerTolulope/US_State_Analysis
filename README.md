# State Comparison Analysis

## Overview
This repository contains a Python script for analyzing various factors across U.S. states to aid in decision-making about relocation. The script evaluates states based on criteria like cost of living, tax burden, diversity, crime rate, climate, and more.

## Data
Data is rated from 1 (least desirable) to 10 (most desirable).

### Cost of Living
- **Description**: The cost of living refers to the average expenses required to maintain a certain standard of living, including housing, groceries, transportation, and healthcare. It is essential for budgeting and evaluating financial feasibility in different locations.

### Overall Tax Burden
- **Description**: The overall tax burden includes all taxes paid by individuals and businesses, including income tax, property tax, sales tax, and other local taxes. It reflects the total tax impact on residents and can affect disposable income.

### Income Tax
- **Description**: Income tax is the tax imposed on individual or corporate earnings. It varies by state and can significantly influence net income and overall financial planning.

### Diversity
- **Description**: Diversity refers to the representation of different racial, ethnic, and cultural groups within a community. It impacts social integration and inclusivity, contributing to a richer cultural environment.

### Quality of Life
- **Description**: Quality of life encompasses various factors such as health, education, environment, and overall well-being. It reflects how enjoyable and satisfying living in a particular area can be.

### Career Opportunities
- **Description**: Career opportunities include the availability and variety of job options in different fields and industries. It affects professional growth, job satisfaction, and career development.

### Crime Rate
- **Description**: The crime rate measures the frequency of criminal activities in a specific area. It is a critical factor in assessing safety and security for residents.

### Racism
- **Description**: Racism includes systemic and individual discrimination based on race. It affects social cohesion and the well-being of minority groups within a community.

### Natural Disasters
- **Description**: Natural disasters encompass events such as earthquakes, hurricanes, floods, and wildfires. The risk and frequency of such events impact safety and property values.

### Healthcare Quality
- **Description**: Healthcare quality involves the standard of medical services, access to healthcare facilities, and overall health outcomes. It is crucial for maintaining health and managing medical needs.

### Climate
- **Description**: Climate refers to the long-term weather patterns of a region, including temperature, precipitation, and seasonal variations. It affects daily life and outdoor activities.

### Buying Power
- **Description**: Buying power measures the ability to purchase goods and services based on income and cost of living. It indicates the relative value of money in different locations.

### School Ratings
- **Description**: School ratings reflect the performance and quality of educational institutions. They are important for families with children and influence educational outcomes and opportunities.


## Requirements

- `pandas`
- `openpyxl`
- `numpy`
- `scikit-learn`

Install the necessary packages using pip:

```bash
pip install pandas openpyxl numpy scikit-learn
```

## Usage

The script performs a linear regression analysis to predict a composite score for each state based on the provided criteria. To run the script:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/EngineerTolulope/US_States_Living_Ranking_Analysis.git
    ```

2. **Navigate to the directory:**

    ```bash
    cd US_States_Living_Ranking_Analysis
    ```

3. **Run the script:**

    ```bash
    python us_state_comparison.py
    ```

## Script

The script includes:

- **Data Preparation**: Cleaning and organizing data.
- **Feature Scaling**: Normalizing data for analysis.
- **Linear Regression Analysis**: Predicts composite scores for states.
- **Output**: Displays predicted scores for each state.

## `results.xlsx`

After running the script, a file named `results.xlsx` is generated in the project directory. This Excel file contains the following sheets:

- **State Scores**: Contains the predicted composite scores for each state based on the criteria analyzed. Each state is listed with its corresponding score and detailed breakdown by category.
- **Criteria Details**: Provides detailed ratings for each criterion used in the analysis, including cost of living, tax burden, diversity, and more.

This file allows you to review and compare the results in a structured format, facilitating better decision-making regarding relocation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## Acknowledgements

This analysis uses publicly available data to provide an overview of state conditions for prospective movers. Special thanks to the contributors and data providers who made this analysis possible.

