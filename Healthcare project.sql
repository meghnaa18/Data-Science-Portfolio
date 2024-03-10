create table details (resident_status varchar,
					  month_of_death int,
					  sex varchar,
					  detail_age int,
					  marital_status varchar,
					  icd_code_10th_revision varchar,
					  race varchar,
					  hispanic_origin varchar
);
select * from details;
create table deathstats(resident_status varchar,
						current_data_year int,
						injury_at_work varchar,
						manner_of_death varchar,
						method_of_disposition varchar,
						autopsy varchar
);
---EDA
SELECT 
    COUNT(*) AS total_records,
    AVG(detail_age) AS average_age,
    MIN(detail_age) AS min_age,
    MAX(detail_age) AS max_age
FROM 
    details;
---Visualization
SELECT 
    FLOOR(detail_age / 10) * 10 AS age_group,
    COUNT(*) AS count
FROM 
    details
GROUP BY 
    FLOOR(detail_age / 10)
ORDER BY 
    age_group;
	
---Joining both the tables
SELECT *
FROM details AS d
INNER JOIN deathstats AS ds ON d.resident_status = ds.resident_status;
---Demographic Analysis
-- Analyze age distribution
SELECT AVG(detail_age) AS average_age, MIN(detail_age) AS min_age, MAX(detail_age) AS max_age
FROM details;

-- Analyze gender distribution
SELECT sex, COUNT(*) AS count
FROM details
GROUP BY sex;

-- Analyze race distribution
SELECT race, COUNT(*) AS count
FROM details
GROUP BY race;

-- Analyze marital status distribution
SELECT marital_status, COUNT(*) AS count
FROM details
GROUP BY marital_status;

---Correlation Analysis:
-- Correlation between injury at work and demographic factors
SELECT d.sex, d.race, d.marital_status, COUNT(*) AS count
FROM details d
INNER JOIN deathstats ds ON d.resident_status = ds.resident_status
WHERE ds.injury_at_work = 'U'
GROUP BY d.sex, d.race, d.marital_status;

-- Correlation between manner of death and demographic factors
SELECT d.sex, d.race, d.marital_status, ds.manner_of_death, COUNT(*) AS count
FROM details d
INNER JOIN deathstats ds ON d.resident_status = ds.resident_status
GROUP BY d.sex, d.race, d.marital_status, ds.manner_of_death;

---Health outcome analysis
-- Count occurrences of different causes of death
SELECT icd_code_10th_revision, COUNT(*) AS death_count
FROM details
GROUP BY icd_code_10th_revision
ORDER BY death_count DESC;

-- Explore associations between health outcomes and other variables
SELECT ds.manner_of_death, ds.autopsy, COUNT(*) AS death_count
FROM details d
INNER JOIN deathstats ds ON d.resident_status = ds.resident_status
GROUP BY ds.manner_of_death, ds.autopsy;

---Temporal Analysis
-- Count occurrences of different causes of death by year
SELECT current_data_year, icd_code_10th_revision, COUNT(*) AS death_count
FROM details
INNER JOIN deathstats ON details.resident_status = deathstats.resident_status
GROUP BY current_data_year, icd_code_10th_revision
ORDER BY current_data_year, death_count DESC;

---Analysis of Patterns of Injury at Work:
-- Count occurrences of injury at work by year
SELECT current_data_year, injury_at_work, COUNT(*) AS count
FROM deathstats
GROUP BY current_data_year, injury_at_work
ORDER BY current_data_year;

---Predictive modelling
-- Data preprocessing and feature engineering
SELECT
    COALESCE(detail_age, 0) AS age,
    CASE
        WHEN sex = 'Male' THEN 1
        WHEN sex = 'Female' THEN 0
        ELSE -1
    END AS gender,
    ...
    target_variable
FROM
    details
JOIN
    deathstats ON details.resident_status = deathstats.resident_status;

-- Model training
CREATE TABLE training_data AS
SELECT
    features,
    target_variable
FROM
    preprocessed_data
WHERE
    condition_for_training;

-- Model evaluation
SELECT
    model_evaluation_metrics
FROM
    trained_model
JOIN
    testing_data ON condition_for_testing;

-- Prediction
SELECT
    predicted_outcomes
FROM
    trained_model
JOIN
    new_data ON condition_for_prediction;








