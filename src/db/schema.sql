DROP TABLE IF EXISTS daily_metrics CASCADE;
DROP TABLE IF EXISTS activities CASCADE;
DROP TABLE IF EXISTS athletes CASCADE;

CREATE TABLE athletes (
    athlete_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    weight_kg NUMERIC(5, 2),
    resting_hr INT,
    max_hr INT,
    ftp_watts INT
);

CREATE TABLE activities (
    activity_id VARCHAR(50) PRIMARY KEY,
    athlete_id VARCHAR(50) REFERENCES athletes(athlete_id),
    name VARCHAR(255) NOT NULL,
    sport_type VARCHAR(50) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    distance_km NUMERIC(6, 2),
    duration_minutes NUMERIC(6, 2),
    elevation_gain_m NUMERIC(6, 1),
    avg_heart_rate INT,
    max_heart_rate INT,
    avg_pace_min_km NUMERIC(4, 2),
    training_stress_score NUMERIC(6, 1),
    perceived_exertion INT
);

CREATE TABLE daily_metrics (
    metric_id SERIAL PRIMARY KEY,
    athlete_id VARCHAR(50) REFERENCES athletes(athlete_id),
    date DATE NOT NULL,
    ctl NUMERIC(5, 2),
    atl NUMERIC(5, 2),
    tsb NUMERIC(5, 2),
    acwr NUMERIC(4, 2)
);