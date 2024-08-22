import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define the states and their respective data for each category
data = {
    "State": [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
    ],
    "Cost of Living": [
        9,  # Alabama (low cost of living)
        5,  # Alaska (high cost due to remoteness)
        7,  # Arizona (moderate cost of living)
        9,  # Arkansas (low cost of living)
        1,  # California (very high cost of living)
        4,  # Colorado (high cost of living)
        3,  # Connecticut (very high cost of living)
        6,  # Delaware (above average)
        5,  # Florida (above average)
        7,  # Georgia (moderate cost of living)
        1,  # Hawaii (highest cost of living in the US)
        7,  # Idaho (moderate cost of living)
        5,  # Illinois (above average)
        8,  # Indiana (below average)
        8,  # Iowa (below average)
        8,  # Kansas (below average)
        8,  # Kentucky (below average)
        8,  # Louisiana (below average)
        6,  # Maine (above average)
        3,  # Maryland (very high cost of living)
        2,  # Massachusetts (very high cost of living)
        6,  # Michigan (above average)
        5,  # Minnesota (above average)
        10,  # Mississippi (very low cost of living)
        8,  # Missouri (below average)
        6,  # Montana (above average)
        7,  # Nebraska (moderate cost of living)
        6,  # Nevada (above average)
        4,  # New Hampshire (high cost of living)
        3,  # New Jersey (very high cost of living)
        8,  # New Mexico (below average)
        2,  # New York (very high cost of living)
        7,  # North Carolina (moderate cost of living)
        8,  # North Dakota (below average)
        7,  # Ohio (moderate cost of living)
        8,  # Oklahoma (below average)
        4,  # Oregon (high cost of living)
        6,  # Pennsylvania (above average)
        5,  # Rhode Island (above average)
        7,  # South Carolina (moderate cost of living)
        8,  # South Dakota (below average)
        8,  # Tennessee (below average)
        7,  # Texas (moderate cost of living)
        7,  # Utah (moderate cost of living)
        4,  # Vermont (high cost of living)
        6,  # Virginia (above average)
        5,  # Washington (above average)
        9,  # West Virginia (low cost of living)
        8,  # Wisconsin (below average)
        8,  # Wyoming
    ],
    "Overall Tax Burden": [
        6,  # Alabama
        10,  # Alaska (no state income tax, low overall burden)
        7,  # Arizona
        6,  # Arkansas
        2,  # California (high taxes)
        5,  # Colorado
        4,  # Connecticut (high taxes)
        5,  # Delaware
        10,  # Florida (no state income tax)
        6,  # Georgia
        3,  # Hawaii (high cost and taxes)
        7,  # Idaho
        5,  # Illinois (high property taxes)
        6,  # Indiana
        7,  # Iowa
        7,  # Kansas
        6,  # Kentucky
        5,  # Louisiana
        8,  # Maine
        4,  # Maryland (high taxes)
        3,  # Massachusetts (high taxes)
        5,  # Michigan
        5,  # Minnesota
        7,  # Mississippi
        6,  # Missouri
        7,  # Montana
        6,  # Nebraska
        10,  # Nevada (no state income tax)
        8,  # New Hampshire (no state income tax but high property taxes)
        2,  # New Jersey (high taxes)
        6,  # New Mexico
        2,  # New York (high taxes)
        6,  # North Carolina
        9,  # North Dakota
        6,  # Ohio
        7,  # Oklahoma
        4,  # Oregon
        5,  # Pennsylvania
        4,  # Rhode Island (high taxes)
        6,  # South Carolina
        9,  # South Dakota (no state income tax, low overall burden)
        6,  # Tennessee (no state income tax but high sales tax)
        10,  # Texas (no state income tax)
        7,  # Utah
        8,  # Vermont
        6,  # Virginia
        10,  # Washington (no state income tax)
        8,  # West Virginia
        6,  # Wisconsin
        10,  # Wyoming (no state income tax)
    ],
    "Income Tax": [
        5,  # Alabama (moderate income tax rates)
        10,  # Alaska (no state income tax)
        6,  # Arizona (moderate income tax rates)
        6,  # Arkansas (moderate income tax rates)
        2,  # California (high income tax rates)
        7,  # Colorado (moderate income tax rates)
        3,  # Connecticut (high income tax rates)
        4,  # Delaware (moderate income tax rates)
        10,  # Florida (no state income tax)
        7,  # Georgia (moderate income tax rates)
        3,  # Hawaii (high income tax rates)
        6,  # Idaho (moderate income tax rates)
        6,  # Illinois (high income tax rates)
        6,  # Indiana (moderate income tax rates)
        6,  # Iowa (moderate income tax rates)
        7,  # Kansas (moderate income tax rates)
        6,  # Kentucky (moderate income tax rates)
        5,  # Louisiana (moderate income tax rates)
        8,  # Maine (high income tax rates)
        4,  # Maryland (high income tax rates)
        3,  # Massachusetts (high income tax rates)
        6,  # Michigan (moderate income tax rates)
        6,  # Minnesota (high income tax rates)
        7,  # Mississippi (moderate income tax rates)
        6,  # Missouri (moderate income tax rates)
        6,  # Montana (moderate income tax rates)
        6,  # Nebraska (moderate income tax rates)
        10,  # Nevada (no state income tax)
        10,  # New Hampshire (no state income tax)
        2,  # New Jersey (high income tax rates)
        6,  # New Mexico (moderate income tax rates)
        2,  # New York (high income tax rates)
        7,  # North Carolina (moderate income tax rates)
        10,  # North Dakota (no state income tax)
        6,  # Ohio (moderate income tax rates)
        6,  # Oklahoma (moderate income tax rates)
        4,  # Oregon (high income tax rates)
        6,  # Pennsylvania (moderate income tax rates)
        4,  # Rhode Island (high income tax rates)
        7,  # South Carolina (moderate income tax rates)
        10,  # South Dakota (no state income tax)
        10,  # Tennessee (no state income tax)
        10,  # Texas (no state income tax)
        7,  # Utah (moderate income tax rates)
        8,  # Vermont (high income tax rates)
        6,  # Virginia (moderate income tax rates)
        10,  # Washington (no state income tax)
        8,  # West Virginia (high income tax rates)
        6,  # Wisconsin (moderate income tax rates)
        10,  # Wyoming (no state income tax)
    ],
    "Diversity": [
        6,  # Alabama
        4,  # Alaska
        7,  # Arizona
        5,  # Arkansas
        10,  # California (highly diverse)
        7,  # Colorado
        6,  # Connecticut
        6,  # Delaware
        9,  # Florida (highly diverse)
        8,  # Georgia
        10,  # Hawaii (very diverse)
        4,  # Idaho
        8,  # Illinois
        5,  # Indiana
        4,  # Iowa
        5,  # Kansas
        6,  # Kentucky
        6,  # Louisiana
        4,  # Maine
        7,  # Maryland
        8,  # Massachusetts
        7,  # Michigan
        6,  # Minnesota
        5,  # Mississippi
        5,  # Missouri
        4,  # Montana
        5,  # Nebraska
        8,  # Nevada
        4,  # New Hampshire
        8,  # New Jersey
        6,  # New Mexico
        10,  # New York (highly diverse)
        7,  # North Carolina
        3,  # North Dakota
        6,  # Ohio
        5,  # Oklahoma
        6,  # Oregon
        7,  # Pennsylvania
        6,  # Rhode Island
        7,  # South Carolina
        3,  # South Dakota
        6,  # Tennessee
        8,  # Texas
        5,  # Utah
        4,  # Vermont
        7,  # Virginia
        9,  # Washington
        3,  # West Virginia
        6,  # Wisconsin
        4,  # Wyoming
    ],
    "Crime Rate": [
        5,  # Alabama
        6,  # Alaska
        5,  # Arizona
        6,  # Arkansas
        3,  # California (higher crime rate)
        5,  # Colorado
        7,  # Connecticut
        6,  # Delaware
        6,  # Florida
        6,  # Georgia
        4,  # Hawaii
        5,  # Idaho
        3,  # Illinois (higher crime rate)
        5,  # Indiana
        6,  # Iowa
        5,  # Kansas
        6,  # Kentucky
        4,  # Louisiana (higher crime rate)
        10,  # Maine (lowest crime rate)
        5,  # Maryland
        4,  # Massachusetts
        6,  # Michigan
        7,  # Minnesota
        6,  # Mississippi
        5,  # Missouri
        8,  # Montana
        6,  # Nebraska
        6,  # Nevada
        10,  # New Hampshire (lowest crime rate)
        6,  # New Jersey
        6,  # New Mexico
        3,  # New York (higher crime rate)
        5,  # North Carolina
        10,  # North Dakota (lowest crime rate)
        6,  # Ohio
        6,  # Oklahoma
        6,  # Oregon
        5,  # Pennsylvania
        6,  # Rhode Island
        6,  # South Carolina
        10,  # South Dakota (lowest crime rate)
        5,  # Tennessee
        6,  # Texas
        7,  # Utah
        10,  # Vermont (lowest crime rate)
        6,  # Virginia
        6,  # Washington
        8,  # West Virginia
        6,  # Wisconsin
        10,  # Wyoming (lowest crime rate)
    ],
    "Climate": [
        6,  # Alabama
        2,  # Alaska
        8,  # Arizona
        6,  # Arkansas
        7,  # California
        6,  # Colorado
        7,  # Connecticut
        6,  # Delaware
        8,  # Florida
        7,  # Georgia
        10,  # Hawaii
        6,  # Idaho
        5,  # Illinois
        5,  # Indiana
        6,  # Iowa
        6,  # Kansas
        6,  # Kentucky
        5,  # Louisiana
        8,  # Maine
        6,  # Maryland
        5,  # Massachusetts
        5,  # Michigan
        4,  # Minnesota
        6,  # Mississippi
        6,  # Missouri
        5,  # Montana
        6,  # Nebraska
        7,  # Nevada
        8,  # New Hampshire
        6,  # New Jersey
        6,  # New Mexico
        5,  # New York
        7,  # North Carolina
        3,  # North Dakota
        6,  # Ohio
        6,  # Oklahoma
        5,  # Oregon
        6,  # Pennsylvania
        6,  # Rhode Island
        7,  # South Carolina
        3,  # South Dakota
        6,  # Tennessee
        6,  # Texas
        6,  # Utah
        8,  # Vermont
        6,  # Virginia
        7,  # Washington
        5,  # West Virginia
        6,  # Wisconsin
        5,  # Wyoming
    ],
    "Racism": [
        4,  # Alabama (some issues with racial tension and discrimination)
        7,  # Alaska (less racial diversity, some issues with discrimination)
        6,  # Arizona (issues with racial discrimination, but some protections)
        5,  # Arkansas (moderate issues with racial tension)
        8,  # California (strong legal protections, generally positive climate)
        6,  # Colorado (good legal framework, some racial tensions)
        8,  # Connecticut (strong legal protections, generally inclusive)
        8,  # Delaware (good legal protections, moderate social climate)
        6,  # Florida (some issues with racial tensions, but diverse)
        5,  # Georgia (issues with racial discrimination, but improving)
        10,  # Hawaii (generally low levels of racial discrimination, high inclusiveness)
        6,  # Idaho (less diversity, some racial tension)
        5,  # Illinois (good protections but occasional issues)
        5,  # Indiana (moderate issues with racial tension and discrimination)
        5,  # Iowa (some issues with racial discrimination)
        6,  # Kansas (moderate issues with racial tension)
        5,  # Kentucky (issues with racial discrimination and tension)
        6,  # Louisiana (moderate issues, improving protections)
        8,  # Maine (generally low levels of racial tension)
        7,  # Maryland (good legal protections, some racial issues)
        8,  # Massachusetts (strong legal protections, generally positive)
        6,  # Michigan (moderate issues with racial tension)
        4,  # Minnesota (some racial tensions, but generally strong protections)
        5,  # Mississippi (high issues with racial discrimination and tension)
        5,  # Missouri (moderate issues with racial tension)
        7,  # Montana (less diversity, some racial tensions)
        6,  # Nebraska (moderate issues with racial discrimination)
        6,  # Nevada (moderate issues, improving protections)
        8,  # New Hampshire (generally low levels of racial tension)
        8,  # New Jersey (good legal protections, generally inclusive)
        6,  # New Mexico (moderate issues, but diverse)
        7,  # New York (generally good protections, but some issues)
        6,  # North Carolina (moderate issues with racial tensions)
        3,  # North Dakota (less diversity, issues with racial discrimination)
        6,  # Ohio (moderate issues with racial tension)
        5,  # Oklahoma (some issues with racial discrimination)
        6,  # Oregon (moderate issues, generally positive)
        6,  # Pennsylvania (moderate issues with racial tension)
        8,  # Rhode Island (generally low levels of racial tension)
        5,  # South Carolina (moderate issues with racial tension)
        3,  # South Dakota (less diversity, issues with racial discrimination)
        5,  # Tennessee (moderate issues with racial tension)
        6,  # Texas (moderate issues, improving legal protections)
        7,  # Utah (less diversity, some racial tensions)
        8,  # Vermont (generally low levels of racial tension)
        6,  # Virginia (moderate issues with racial tension)
        8,  # Washington (generally positive, strong protections)
        6,  # West Virginia (moderate issues with racial tension)
        6,  # Wisconsin (moderate issues with racial discrimination and tension)
        5,  # Wyoming (less diversity, some racial tensions)
    ],
    "Quality of Life": [
        5,  # Alabama
        7,  # Alaska
        6,  # Arizona
        5,  # Arkansas
        8,  # California
        8,  # Colorado
        9,  # Connecticut
        7,  # Delaware
        7,  # Florida
        6,  # Georgia
        9,  # Hawaii
        7,  # Idaho
        7,  # Illinois
        6,  # Indiana
        6,  # Iowa
        6,  # Kansas
        5,  # Kentucky
        8,  # Louisiana
        8,  # Maine
        9,  # Maryland
        9,  # Massachusetts
        6,  # Michigan
        6,  # Minnesota
        4,  # Mississippi
        6,  # Missouri
        7,  # Montana
        6,  # Nebraska
        7,  # Nevada
        7,  # New Hampshire
        8,  # New Jersey
        6,  # New Mexico
        9,  # New York
        7,  # North Carolina
        4,  # North Dakota
        6,  # Ohio
        6,  # Oklahoma
        7,  # Oregon
        7,  # Pennsylvania
        8,  # Rhode Island
        6,  # South Carolina
        4,  # South Dakota
        6,  # Tennessee
        7,  # Texas
        7,  # Utah
        8,  # Vermont
        7,  # Virginia
        9,  # Washington
        5,  # West Virginia
        6,  # Wisconsin
        6,  # Wyoming
    ],
    "Career Opportunities": [
        4,  # Alabama
        3,  # Alaska
        6,  # Arizona
        4,  # Arkansas
        10,  # California
        9,  # Colorado
        7,  # Connecticut
        6,  # Delaware
        8,  # Florida
        7,  # Georgia
        4,  # Hawaii
        5,  # Idaho
        9,  # Illinois
        6,  # Indiana
        5,  # Iowa
        5,  # Kansas
        4,  # Kentucky
        5,  # Louisiana
        6,  # Maine
        9,  # Maryland
        10,  # Massachusetts
        7,  # Michigan
        8,  # Minnesota
        3,  # Mississippi
        6,  # Missouri
        4,  # Montana
        5,  # Nebraska
        8,  # Nevada
        6,  # New Hampshire
        9,  # New Jersey
        5,  # New Mexico
        10,  # New York
        8,  # North Carolina
        3,  # North Dakota
        7,  # Ohio
        6,  # Oklahoma
        8,  # Oregon
        7,  # Pennsylvania
        6,  # Rhode Island
        5,  # South Carolina
        3,  # South Dakota
        6,  # Tennessee
        9,  # Texas
        7,  # Utah
        5,  # Vermont
        8,  # Virginia
        10,  # Washington
        4,  # West Virginia
        6,  # Wisconsin
        3,  # Wyoming
    ],
    "Buying Power": [
        6,  # Alabama
        5,  # Alaska
        6,  # Arizona
        7,  # Arkansas
        3,  # California
        7,  # Colorado
        4,  # Connecticut
        6,  # Delaware
        6,  # Florida
        6,  # Georgia
        2,  # Hawaii
        7,  # Idaho
        5,  # Illinois
        7,  # Indiana
        7,  # Iowa
        7,  # Kansas
        6,  # Kentucky
        6,  # Louisiana
        6,  # Maine
        5,  # Maryland
        3,  # Massachusetts
        6,  # Michigan
        6,  # Minnesota
        8,  # Mississippi
        6,  # Missouri
        7,  # Montana
        7,  # Nebraska
        6,  # Nevada
        6,  # New Hampshire
        5,  # New Jersey
        6,  # New Mexico
        2,  # New York
        6,  # North Carolina
        8,  # North Dakota
        6,  # Ohio
        6,  # Oklahoma
        6,  # Oregon
        6,  # Pennsylvania
        5,  # Rhode Island
        6,  # South Carolina
        8,  # South Dakota
        6,  # Tennessee
        6,  # Texas
        6,  # Utah
        5,  # Vermont
        6,  # Virginia
        4,  # Washington
        5,  # West Virginia
        7,  # Wisconsin
        8,  # Wyoming
    ],
    "Healthcare Quality": [
        4,  # Alabama
        6,  # Alaska
        5,  # Arizona
        4,  # Arkansas
        8,  # California
        8,  # Colorado
        8,  # Connecticut
        7,  # Delaware
        6,  # Florida
        5,  # Georgia
        9,  # Hawaii
        6,  # Idaho
        7,  # Illinois
        6,  # Indiana
        8,  # Iowa
        7,  # Kansas
        5,  # Kentucky
        4,  # Louisiana
        7,  # Maine
        8,  # Maryland
        9,  # Massachusetts
        7,  # Michigan
        9,  # Minnesota
        4,  # Mississippi
        6,  # Missouri
        6,  # Montana
        7,  # Nebraska
        5,  # Nevada
        8,  # New Hampshire
        8,  # New Jersey
        5,  # New Mexico
        8,  # New York
        6,  # North Carolina
        7,  # North Dakota
        7,  # Ohio
        5,  # Oklahoma
        7,  # Oregon
        7,  # Pennsylvania
        8,  # Rhode Island
        5,  # South Carolina
        7,  # South Dakota
        5,  # Tennessee
        6,  # Texas
        7,  # Utah
        8,  # Vermont
        7,  # Virginia
        8,  # Washington
        4,  # West Virginia
        7,  # Wisconsin
        5,  # Wyoming
    ],
    "Natural Disasters": [
        3,  # Alabama - High hurricane, tornado, and flood risk
        7,  # Alaska - Earthquakes and some extreme weather, but lower tornado/hurricane risk
        5,  # Arizona - Risk of wildfires, but minimal hurricane and tornado risk
        4,  # Arkansas - High tornado risk, some flooding
        3,  # California - High earthquake and wildfire risk
        6,  # Colorado - Wildfires, some floods, but no hurricanes or major earthquakes
        7,  # Connecticut - Moderate hurricane risk, low other risks
        6,  # Delaware - Moderate hurricane and flood risk
        3,  # Florida - High hurricane risk, flooding, some tornadoes
        4,  # Georgia - Hurricane, tornado, and flood risk
        6,  # Hawaii - Volcanic activity, some hurricanes, but rare
        6,  # Idaho - Low earthquake and flood risk
        4,  # Illinois - Tornadoes and flooding, low earthquake risk
        5,  # Indiana - Tornadoes and some flooding
        6,  # Iowa - Tornadoes and some flooding
        4,  # Kansas - High tornado risk, low hurricane risk
        5,  # Kentucky - Tornadoes and some flooding
        4,  # Louisiana - High hurricane and flood risk
        8,  # Maine - Low hurricane, earthquake, and flood risk
        5,  # Maryland - Moderate hurricane and flood risk
        7,  # Massachusetts - Some hurricane risk, low earthquake and flood risk
        7,  # Michigan - Low earthquake, tornado, and hurricane risk
        7,  # Minnesota - Low earthquake and hurricane risk, some flooding
        3,  # Mississippi - High hurricane and flood risk
        4,  # Missouri - High tornado and some earthquake risk
        6,  # Montana - Low earthquake and flood risk
        4,  # Nebraska - High tornado risk, some flooding
        7,  # Nevada - Earthquake risk but low hurricane/tornado risk
        7,  # New Hampshire - Low hurricane and earthquake risk
        6,  # New Jersey - Moderate hurricane and flood risk
        6,  # New Mexico - Wildfires and some flooding, but low hurricane and earthquake risk
        6,  # New York - Moderate hurricane and flood risk
        4,  # North Carolina - High hurricane risk, tornadoes
        7,  # North Dakota - Low hurricane and earthquake risk, some flooding
        6,  # Ohio - Tornadoes and some flooding
        3,  # Oklahoma - High tornado risk, some earthquakes
        7,  # Oregon - Earthquake and wildfire risk, low hurricane risk
        6,  # Pennsylvania - Some flooding and moderate tornado risk
        7,  # Rhode Island - Low hurricane and earthquake risk
        4,  # South Carolina - High hurricane risk, some flooding
        7,  # South Dakota - Low earthquake and hurricane risk, some flooding
        4,  # Tennessee - Tornadoes and some flooding
        4,  # Texas - High hurricane, tornado, and flood risk
        7,  # Utah - Earthquake risk but low hurricane/tornado risk
        8,  # Vermont - Low hurricane, tornado, and earthquake risk
        5,  # Virginia - Moderate hurricane and flood risk
        7,  # Washington - Earthquake and wildfire risk, low hurricane risk
        4,  # West Virginia - Flooding and some tornado risk
        7,  # Wisconsin - Low hurricane and earthquake risk, some flooding
        6,  # Wyoming - Low hurricane and earthquake risk, some wildfire risk
    ],
    "School Ratings": [
        4,  # Alabama - Challenges with funding and performance, particularly in rural areas
        7,  # Alaska - Challenges with remote and rural schools, though some urban schools perform well
        6,  # Arizona - Struggles with funding, but some strong schools in specific areas
        4,  # Arkansas - Faces challenges in rural areas, with some strong suburban schools
        6,  # California - Wide disparity between top-performing and struggling schools
        8,  # Colorado - High-quality education, particularly in certain districts
        10,  # Connecticut - Consistently ranks high in education quality and student outcomes
        6,  # Delaware - Mixed performance with some strong and some struggling districts
        6,  # Florida - Decent education system, but variability across districts
        6,  # Georgia - Mixed performance with strong suburban schools and weaker rural areas
        7,  # Hawaii - Mixed performance, with challenges in rural areas
        6,  # Idaho - Struggles with funding and performance, particularly in rural areas
        7,  # Illinois - Mixed performance, with some strong suburban schools and weaker urban schools
        5,  # Indiana - Mixed performance with some strong and some struggling districts
        6,  # Iowa - Adequate education system, but not top-tier
        4,  # Kansas - Struggles with funding and performance, particularly in rural areas
        5,  # Kentucky - Struggles with funding and performance in rural areas
        4,  # Louisiana - Struggles with funding and performance, particularly in urban areas
        8,  # Maine - Good education system, though not as high as neighboring states
        8,  # Maryland - High-performing schools, particularly in suburban areas
        10,  # Massachusetts - Known for top-tier education with high standardized test scores and well-funded schools
        7,  # Michigan - Struggles in urban areas but strong suburban schools
        7,  # Minnesota - Excellent public school system with high graduation rates
        3,  # Mississippi - Struggles with funding and performance, particularly in rural areas
        4,  # Missouri - Significant disparities between urban, suburban, and rural schools
        6,  # Montana - Struggles with funding and performance, particularly in rural areas
        7,  # Nebraska - Decent education, but not exceptional
        5,  # Nevada - Faces significant challenges in funding and performance
        8,  # New Hampshire - Good education system, though not as high as neighboring states
        6,  # New Jersey - Strong public education system with high graduation rates and funding
        6,  # New Mexico - Faces significant challenges in funding and student performance
        8,  # New York - Strong education, particularly in suburban areas and top-ranked public schools
        6,  # North Carolina - Adequate education system with variability between districts
        3,  # North Dakota - Struggles with funding and performance in rural areas
        6,  # Ohio - Tornadoes and some flooding
        3,  # Oklahoma - High tornado risk, some earthquakes
        7,  # Oregon - Adequate school system, but faces funding challenges
        7,  # Pennsylvania - Strong suburban schools, but significant disparities in urban areas
        6,  # Rhode Island - Decent school system, though it has some disparities
        4,  # South Carolina - Struggles in rural areas, but some good suburban schools
        3,  # South Dakota - Challenges with funding and performance in rural areas
        5,  # Tennessee - Struggles in rural and urban areas, but some good suburban schools
        6,  # Texas - Strong in certain districts, but significant disparities in rural and urban areas
        8,  # Utah - Good education, but challenges in funding
        8,  # Vermont - Strong education system with small class sizes and high funding
        5,  # Virginia - High standards for education and good student performance
        7,  # Washington - Decent school system, though varies by district
        4,  # West Virginia - Struggles with funding and performance, particularly in rural areas
        7,  # Wisconsin - Good education quality, but some disparities in rural areas
        6,  # Wyoming - Adequate education system with variability between districts
    ],
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Define the priority weights for each category
weights = {
    "Cost of Living": 0.15,
    "Overall Tax Burden": 0.12,
    "Income Tax": 0.1,
    "Diversity": 0.1,
    "Crime Rate": 0.09,
    "Climate": 0.08,
    "Racism": 0.08,
    "Quality of Life": 0.09,
    "Career Opportunities": 0.08,
    "School Ratings": 0.04,
    "Buying Power": 0.05,
    "Healthcare Quality": 0.05,
    "Natural Disasters": 0.05,
}

# Ensure weights sum to 1.0
def adjust_weights(weights):
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        adjustment_factor = 1.0 / total_weight
        return {key: value * adjustment_factor for key, value in weights.items()}
    return weights


weights = adjust_weights(weights)

# Store original min and max values for each column
original_min_max = {}


def normalize(df, columns):
    global original_min_max
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        original_min_max[col] = (min_val, max_val)
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


# Normalize the relevant columns
feature_columns = [col for col in df.columns if col != "State"]
df = normalize(df, feature_columns)

# Calculate initial scores using given weights
df["Overall Score"] = df[feature_columns].apply(
    lambda row: sum(row[col] * weights[col] for col in feature_columns), axis=1
)

# Prepare data for machine learning model
X = df[feature_columns].values
y = df["Overall Score"].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Retrieve the model coefficients as new weights
new_weights = dict(zip(feature_columns, model.coef_))
new_weights = adjust_weights(new_weights)

# Sort the weights in descending order
sorted_weights = sorted(new_weights.items(), key=lambda x: x[1], reverse=True)

# Print the sorted weights
print("Final Weights Used (sorted):")
for category, weight in sorted_weights:
    print(f"{category}: {weight}")

# Calculate the overall score with new weights
df["Overall Score"] = df[feature_columns].apply(
    lambda row: sum(row[col] * new_weights[col] for col in feature_columns), axis=1
)

# Normalize the 'Overall Score' to be between 1 and 10
overall_score_min = df['Overall Score'].min()
overall_score_max = df['Overall Score'].max()
df['Overall Score'] = 1 + 9 * (df['Overall Score'] - overall_score_min) / (overall_score_max - overall_score_min)


# Denormalize the feature columns for final output
def denormalize(score, min_val, max_val):
    return score * (max_val - min_val) + min_val


# Apply denormalization to feature columns
for col in feature_columns:
    min_val, max_val = original_min_max[col]
    df[col] = denormalize(df[col], min_val, max_val)

# Sort by the overall score
df_sorted = df.sort_values(by="Overall Score", ascending=False)

# Convert sorted DataFrame to CSV format
csv_output = df_sorted.to_csv(index=False, sep="\t")

# Print the CSV output
print(csv_output)
