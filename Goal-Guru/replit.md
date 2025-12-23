# Football Prediction Hub

## Overview
The Football Prediction Hub is a comprehensive system designed to provide accurate football match predictions using advanced machine learning, statistical analysis, and real-time data. It specializes in various prediction types, including exact scores, half-time/full-time outcomes, betting predictions, and performance analytics. The project aims to deliver highly reliable predictions for the football market.

## How to Run
```bash
python main.py
```
The app runs on port 5000.

## User Preferences
Preferred communication style: Simple, everyday language.
Interface language: Turkish
Priority: Automatic fixture data refresh when API keys are updated
UI Style: Normal font weight for progress bars (not bold) for modern appearance
API Plan: FREE (no access to standings, weather, injuries - uses match history for stats)

## System Architecture
The application is built on a modular Flask-based architecture, incorporating multiple specialized prediction models and robust data processing components.

### Core Architecture
- **Backend Framework**: Flask web application with RESTful API endpoints.
- **Data Processing**: Real-time football data fetching from multiple APIs.
- **Prediction Engine**: Multi-model ensemble approach with specialized algorithms and dynamic weight adjustment.
- **Caching Layer**: Two-tier (memory + disk) JSON-based prediction caching for performance optimization.
- **Model Validation**: Cross-validation and backtesting capabilities ensure continuous improvement.
- **Database**: PostgreSQL for persistent storage of match data, predictions, and model performance.
- **Performance Optimization**: Parallel processing, batch prediction, and asynchronous data fetching.
- **Explainable AI (XAI)**: Provides natural language explanations and key factor analysis for predictions.
- **Dynamic Analysis**: Includes dynamic league strength analysis, team performance analysis, and HT/FT surprise detection.
- **Rating System**: Enhanced hybrid ML rating system with xG integration (Updated 02/08/2025):
  - Glicko-2: 15% (reduced)
  - TrueSkill: 10% (reduced) 
  - Recent Form: 25% (focusing on last 5 matches)
  - xG Rating: 40% (INCREASED - using Soccer Prediction approach for better draw detection)
  - ML Factors: 10% (reduced)
- **Feature Extraction Pipeline** (Added 11/08/2025):
  - Advanced ML-based feature extraction system with 65% venue-specific, 35% general performance weighting
  - Automated team characteristic profiling (attack style, defense approach, game tempo, risk appetite)
  - Pattern recognition using Random Forest, XGBoost, and KNN algorithms
  - Dynamic weight adjustment based on team away/home strength patterns
  - Feature quality scoring for data reliability assessment
- **Venue-Specific Performance** (Updated 10/08/2025):
  - Lambda calculations now use 65% venue-specific performance weight, 35% general form weight
  - Separate analysis for last 5-10 home matches (for home team) and away matches (for away team)
  - Venue bonus applied: 10% for home teams with >60% win rate in last 5 home matches
  - Venue bonus applied: 5% for away teams with >40% win rate in last 5 away matches
- **Team Statistics Popup** (Added 10/08/2025):
  - Modern tab-based interface with team names as tab headers
  - Shows last 5 match performance for each team separately
  - Includes comparison tab showing head-to-head form statistics
  - API endpoints stored in TEAM_STATS_API object for easy reference
  - Uses `/api/predict-match/{teamId}/{teamId}` with home_name and away_name parameters
- **1X2 Prediction Enhancement** (Updated 02/08/2025):
  - xG-based predictions now have 50% weight in final 1X2 calculations
  - Lambda factor calculation prioritizes xG ratings with 60% weight
  - Special draw correction when team strengths are similar (within 10 points)
  - Direct Poisson calculation using xG attack/defense ratings
- **Lambda Calculation Enhancement** (Updated 11/08/2025):
  - Now uses WEIGHTED AVERAGE approach instead of multiplication for more balanced results
  - Base Lambda = xG × xGA (core prediction)
  - Combined Factor = 40% log_adjustment + 30% venue_bonus + 30% league_factor
  - Final Lambda = Base Lambda × Combined Factor (typically 0.8-1.3 range)
  - League factor only applies when teams are from DIFFERENT leagues (cup matches, cross-league)
  - Same league matches use neutral factor (1.0) - no adjustment needed
  - Dynamic league profiling for high/medium/low scoring leagues
- **xG Integration**: Implements Soccer Prediction methodology:
  - Dynamic team strength ratings based on xG and actual goals
  - Formula: g = (xG × 0.876) + (goals × 0.124)
  - Separate home/away attack and defense ratings
- **PSO Optimization**: Particle Swarm Optimization for parameter tuning
- **Draw Correction Factor**: Automatically increases draw probability based on rating differences and match context.

### Key Components
- **Main Application (`main.py`)**: Manages Flask web server, API endpoints, and template rendering.
- **Match Prediction Engine (`match_prediction.py`)**: Houses core prediction logic, including Poisson distribution, Monte Carlo simulations, and specialized betting predictions.
- **Advanced ML Models**: Integrates XGBoost, LSTM neural networks, Bayesian networks, CRF, and self-learning models.
- **Data Management**: Handles API integration, dynamic team analysis, and prediction caching.
- **Model Validation and Learning**: Features a model validator, self-learning predictor, and continuous performance monitoring.
- **Enhanced Analysis Features**: Includes Goal Trend Analyzer, Enhanced Prediction Factors, and Match Insights Generator.

### UI/UX Decisions
- Uses Bootstrap for UI framework with dark theme support.
- Implements mobile-responsive design, particularly for explanation and H2H sections.
- Incorporates dynamic UI elements like animated confidence meters and modern stat cards.
- Supports Turkish language interface with UTF-8 character support.

## External Dependencies

### APIs
- **Football-Data.org API**: Primary source for match fixtures and results.
- **API-Football**: Secondary data source for comprehensive coverage.
- **Grok AI**: For advanced analysis (optional XAI API integration).

### Python Libraries
- **Flask**: Web framework and API development.
- **TensorFlow/Keras**: Deep learning models.
- **XGBoost**: Gradient boosting algorithms.
- **scikit-learn**: Traditional ML algorithms and validation.
- **NumPy/Pandas**: Data manipulation and analysis.
- **SciPy**: Statistical distributions and analysis.
- **aiohttp**: For asynchronous HTTP requests.
- **glicko2**: For rating system.

### Databases
- **PostgreSQL**: Primary production database.
- **SQLite**: Used for local development.

### Frontend Technologies
- **Bootstrap**: UI framework.
- **jQuery**: DOM manipulation and AJAX requests.
- **Chart.js**: Data visualization.