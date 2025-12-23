"""
Fixture Congestion Analyzer Module for Football Prediction System
Analyzes match density, fatigue effects, and team recovery patterns.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from statistics import mean, median
from collections import defaultdict

logger = logging.getLogger(__name__)

class FixtureCongestionAnalyzer:
    """
    Advanced fixture congestion and fatigue analysis for football predictions.
    Analyzes match density, travel fatigue, recovery patterns, and optimal performance windows.
    """
    
    def __init__(self):
        """Initialize the fixture congestion analyzer with calibration parameters"""
        
        # Analysis time windows (days)
        self.time_windows = {
            'short': 7,     # Last 7 days
            'medium': 14,   # Last 14 days  
            'long': 21      # Last 21 days
        }
        
        # League-specific calibration factors
        self.league_calibration = {
            # Top 5 European leagues - higher intensity
            '39': {'fatigue_multiplier': 1.2, 'recovery_base': 2.5, 'travel_impact': 1.1},  # Premier League
            '140': {'fatigue_multiplier': 1.15, 'recovery_base': 2.3, 'travel_impact': 1.0}, # La Liga
            '135': {'fatigue_multiplier': 1.1, 'recovery_base': 2.2, 'travel_impact': 0.9},  # Serie A
            '78': {'fatigue_multiplier': 1.05, 'recovery_base': 2.0, 'travel_impact': 0.8},  # Bundesliga
            '61': {'fatigue_multiplier': 1.0, 'recovery_base': 1.8, 'travel_impact': 0.7},   # Ligue 1
            
            # European competitions - very high intensity
            '2': {'fatigue_multiplier': 1.4, 'recovery_base': 3.0, 'travel_impact': 1.5},    # Champions League
            '3': {'fatigue_multiplier': 1.3, 'recovery_base': 2.8, 'travel_impact': 1.4},    # Europa League
            '848': {'fatigue_multiplier': 1.2, 'recovery_base': 2.6, 'travel_impact': 1.3},  # Conference League
            
            # Other leagues - standard intensity
            'default': {'fatigue_multiplier': 1.0, 'recovery_base': 2.0, 'travel_impact': 1.0}
        }
        
        # Travel distance impact zones (approximate km)
        self.travel_zones = {
            'local': {'max_distance': 100, 'fatigue_factor': 1.0, 'recovery_penalty': 0},
            'domestic': {'max_distance': 800, 'fatigue_factor': 1.1, 'recovery_penalty': 0.5},
            'european': {'max_distance': 3000, 'fatigue_factor': 1.3, 'recovery_penalty': 1.0},
            'intercontinental': {'max_distance': float('inf'), 'fatigue_factor': 1.6, 'recovery_penalty': 2.0}
        }
        
        # Performance correlation factors
        self.performance_factors = {
            'optimal_rest_days': [3, 4, 5],  # Optimal recovery period
            'fatigue_threshold': 70,          # Above this = significant fatigue
            'overload_threshold': 85,         # Above this = dangerous overload
            'peak_performance_window': [2, 6] # Days after rest for peak performance
        }
        
        # City coordinates for travel distance calculation (major football cities)
        self.city_coordinates = {
            # England
            'London': (51.5074, -0.1278),
            'Manchester': (53.4808, -2.2426),
            'Liverpool': (53.4084, -2.9916),
            'Birmingham': (52.4862, -1.8904),
            'Newcastle': (54.9783, -1.6178),
            'Leeds': (53.8008, -1.5491),
            
            # Spain
            'Madrid': (40.4168, -3.7038),
            'Barcelona': (41.3851, 2.1734),
            'Seville': (37.3891, -5.9845),
            'Valencia': (39.4699, -0.3763),
            'Bilbao': (43.2627, -2.9253),
            
            # Italy
            'Milan': (45.4642, 9.1900),
            'Rome': (41.9028, 12.4964),
            'Naples': (40.8518, 14.2681),
            'Turin': (45.0703, 7.6869),
            'Florence': (43.7696, 11.2558),
            
            # Germany
            'Berlin': (52.5200, 13.4050),
            'Munich': (48.1351, 11.5820),
            'Hamburg': (53.5511, 9.9937),
            'Cologne': (50.9375, 6.9603),
            'Frankfurt': (50.1109, 8.6821),
            'Dortmund': (51.5136, 7.4653),
            
            # France
            'Paris': (48.8566, 2.3522),
            'Lyon': (45.7640, 4.8357),
            'Marseille': (43.2965, 5.3698),
            'Nice': (43.7102, 7.2620),
            'Bordeaux': (44.8378, -0.5792),
            
            # Turkey
            'Istanbul': (41.0082, 28.9784),
            'Ankara': (39.9334, 32.8597),
            'Izmir': (38.4192, 27.1287),
            'Bursa': (40.1826, 29.0669),
            'Antalya': (36.8969, 30.7133)
        }
        
        logger.info("FixtureCongestionAnalyzer initialized with league calibration and travel zones")
    
    def analyze_fixture_congestion(self, team_id: int, matches: List[Dict], 
                                 upcoming_match_date: Optional[datetime] = None,
                                 league_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive fixture congestion analysis for a team
        
        Args:
            team_id: ID of the team to analyze
            matches: List of recent match data
            upcoming_match_date: Date of the upcoming match for analysis
            league_id: League ID for calibration
            
        Returns:
            Dict containing comprehensive congestion analysis
        """
        if not matches:
            return self._get_default_analysis()
            
        try:
            # Reference date for analysis
            reference_date = upcoming_match_date or datetime.now()
            
            # Get league calibration
            league_cal = self._get_league_calibration(league_id)
            
            # Sort matches by date (most recent first)
            sorted_matches = self._sort_matches_by_date(matches)
            
            # 1. Fixture Congestion Calculator
            try:
                congestion_metrics = self._calculate_fixture_congestion(
                    sorted_matches, team_id, reference_date
                )
            except ZeroDivisionError:
                congestion_metrics = self._get_default_congestion_metrics()
            
            # 2. Rest Day Analysis
            rest_analysis = self._analyze_rest_days(
                sorted_matches, team_id, reference_date
            )
            
            # 3. Travel Fatigue Simulation
            travel_analysis = self._simulate_travel_fatigue(
                sorted_matches, team_id, reference_date
            )
            
            # 4. Match Frequency Impact
            frequency_impact = self._analyze_match_frequency_impact(
                sorted_matches, team_id, reference_date
            )
            
            # 5. Recovery Time Modeling
            recovery_analysis = self._model_recovery_time(
                sorted_matches, team_id, reference_date, rest_analysis
            )
            
            # 6. Calculate overall fatigue score (0-100)
            fatigue_score = self._calculate_fatigue_score(
                congestion_metrics, rest_analysis, travel_analysis, 
                frequency_impact, league_cal
            )
            
            # 7. Determine optimal performance windows
            performance_windows = self._identify_performance_windows(
                rest_analysis, recovery_analysis, reference_date
            )
            
            # 8. Generate recommendations
            recommendations = self._generate_recommendations(
                fatigue_score, congestion_metrics, rest_analysis
            )
            
            return {
                'team_id': team_id,
                'analysis_date': reference_date.isoformat() if reference_date else None,
                'league_calibration': league_cal,
                'congestion_metrics': congestion_metrics,
                'rest_analysis': rest_analysis,
                'travel_analysis': travel_analysis,
                'frequency_impact': frequency_impact,
                'recovery_analysis': recovery_analysis,
                'fatigue_score': fatigue_score,
                'performance_windows': performance_windows,
                'recommendations': recommendations,
                'risk_level': self._assess_risk_level(fatigue_score.get('overall_fatigue_score', 50)),
                'comparison_baseline': self._get_league_baseline(league_id)
            }
            
        except Exception as e:
            logger.error(f"Error in fixture congestion analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _calculate_fixture_congestion(self, matches: List[Dict], team_id: int, 
                                    reference_date: datetime) -> Dict[str, Any]:
        """Calculate fixture congestion for different time windows"""
        congestion = {}
        
        for window_name, days in self.time_windows.items():
            cutoff_date = reference_date - timedelta(days=days)
            
            # Count matches in this window
            window_matches = [
                match for match in matches 
                if self._get_match_date(match) and self._get_match_date(match) >= cutoff_date
            ]
            
            # Calculate metrics
            match_count = len(window_matches)
            matches_per_week = (match_count / days) * 7
            
            # Analyze match distribution
            distribution = self._analyze_match_distribution(window_matches, days)
            
            # Calculate congestion intensity
            intensity = self._calculate_congestion_intensity(
                match_count, days, distribution
            )
            
            congestion[window_name] = {
                'days_analyzed': days,
                'match_count': match_count,
                'matches_per_week': round(matches_per_week, 2),
                'distribution': distribution,
                'intensity_score': intensity,
                'congestion_level': self._classify_congestion_level(intensity)
            }
        
        # Calculate overall congestion trend
        congestion['overall_trend'] = self._calculate_congestion_trend(congestion)
        
        return congestion
    
    def _analyze_rest_days(self, matches: List[Dict], team_id: int, 
                         reference_date: datetime) -> Dict[str, Any]:
        """Analyze rest days between matches and their impact on performance"""
        if len(matches) < 2:
            return {'insufficient_data': True}
        
        rest_periods = []
        performance_by_rest = defaultdict(list)
        
        # Calculate rest periods and associated performance
        for i in range(len(matches) - 1):
            current_match = matches[i]
            next_match = matches[i + 1]
            
            current_date = self._get_match_date(current_match)
            next_date = self._get_match_date(next_match)
            
            if current_date and next_date:
                rest_days = (current_date - next_date).days
                rest_periods.append(rest_days)
                
                # Get performance metrics for the current match
                performance = self._extract_match_performance(current_match, team_id)
                if performance:
                    rest_category = self._categorize_rest_period(rest_days)
                    performance_by_rest[rest_category].append(performance)
        
        # Calculate statistics
        avg_rest_days = mean(rest_periods) if rest_periods else 0
        min_rest = min(rest_periods) if rest_periods else 0
        max_rest = max(rest_periods) if rest_periods else 0
        
        # Analyze performance correlation with rest
        rest_correlation = self._calculate_rest_performance_correlation(
            performance_by_rest
        )
        
        # Recent rest pattern analysis
        recent_pattern = self._analyze_recent_rest_pattern(
            matches[:10], team_id  # Last 10 matches
        )
        
        return {
            'rest_statistics': {
                'average_rest_days': round(avg_rest_days, 1),
                'minimum_rest': min_rest,
                'maximum_rest': max_rest,
                'rest_periods': rest_periods
            },
            'performance_correlation': rest_correlation,
            'recent_pattern': recent_pattern,
            'optimal_rest_assessment': self._assess_optimal_rest(rest_periods, performance_by_rest),
            'next_match_rest_prediction': self._predict_next_match_rest(matches, reference_date)
        }
    
    def _simulate_travel_fatigue(self, matches: List[Dict], team_id: int, 
                               reference_date: datetime) -> Dict[str, Any]:
        """Simulate travel fatigue effects based on distances and frequency"""
        travel_data = []
        cumulative_fatigue = 0
        
        # Analyze recent travel pattern (last 30 days)
        cutoff_date = reference_date - timedelta(days=30)
        recent_matches = [
            match for match in matches
            if self._get_match_date(match) and self._get_match_date(match) >= cutoff_date
        ]
        
        for match in recent_matches:
            travel_info = self._calculate_match_travel(match, team_id)
            if travel_info:
                travel_data.append(travel_info)
                cumulative_fatigue += travel_info['fatigue_impact']
        
        # Calculate travel metrics
        total_distance = sum(t['distance'] for t in travel_data)
        avg_distance = mean([t['distance'] for t in travel_data]) if travel_data else 0
        
        # Analyze travel zones
        zone_analysis = self._analyze_travel_zones(travel_data)
        
        # Calculate travel fatigue coefficient
        travel_coefficient = self._calculate_travel_coefficient(
            cumulative_fatigue, total_distance, len(travel_data)
        )
        
        return {
            'travel_statistics': {
                'total_distance_km': round(total_distance, 0),
                'average_distance_km': round(avg_distance, 0),
                'travel_count': len(travel_data),
                'cumulative_fatigue': round(cumulative_fatigue, 2)
            },
            'zone_analysis': zone_analysis,
            'travel_coefficient': travel_coefficient,
            'fatigue_impact_level': self._classify_travel_fatigue(travel_coefficient),
            'recovery_penalty': self._calculate_travel_recovery_penalty(travel_data)
        }
    
    def _analyze_match_frequency_impact(self, matches: List[Dict], team_id: int,
                                      reference_date: datetime) -> Dict[str, Any]:
        """Analyze the impact of match frequency on scores and performance"""
        if len(matches) < 5:
            return {'insufficient_data': True}
        
        frequency_periods = []
        performance_by_frequency = defaultdict(list)
        
        # Analyze different frequency periods
        for window_size in [7, 14, 21]:
            cutoff_date = reference_date - timedelta(days=window_size)
            period_matches = [
                match for match in matches
                if self._get_match_date(match) and self._get_match_date(match) >= cutoff_date
            ]
            
            if period_matches:
                match_frequency = len(period_matches) / (window_size / 7)  # matches per week
                avg_performance = self._calculate_period_performance(period_matches, team_id)
                
                frequency_periods.append({
                    'window_days': window_size,
                    'match_count': len(period_matches),
                    'frequency_per_week': round(match_frequency, 2),
                    'average_performance': avg_performance
                })
                
                # Categorize frequency level
                freq_category = self._categorize_frequency(match_frequency)
                performance_by_frequency[freq_category].append(avg_performance)
        
        # Calculate frequency impact score
        impact_score = self._calculate_frequency_impact_score(frequency_periods)
        
        # Determine optimal frequency range
        optimal_range = self._determine_optimal_frequency(performance_by_frequency)
        
        return {
            'frequency_analysis': frequency_periods,
            'performance_by_frequency': dict(performance_by_frequency),
            'impact_score': impact_score,
            'optimal_frequency_range': optimal_range,
            'current_frequency_assessment': self._assess_current_frequency(frequency_periods)
        }
    
    def _model_recovery_time(self, matches: List[Dict], team_id: int,
                           reference_date: datetime, rest_analysis: Dict) -> Dict[str, Any]:
        """Model recovery time requirements and patterns"""
        if 'rest_statistics' not in rest_analysis:
            return {'insufficient_data': True}
        
        # Get recent match intensities and recovery patterns
        recovery_data = []
        
        for i, match in enumerate(matches[:15]):  # Analyze last 15 matches
            match_intensity = self._calculate_match_intensity(match, team_id)
            
            if i < len(matches) - 1:
                next_match = matches[i + 1]
                rest_days = self._calculate_rest_between_matches(match, next_match)
                recovery_quality = self._assess_recovery_quality(
                    match, next_match, team_id, rest_days or 0
                )
                
                recovery_data.append({
                    'match_intensity': match_intensity,
                    'rest_days': rest_days,
                    'recovery_quality': recovery_quality,
                    'next_match_performance': self._extract_match_performance(next_match, team_id)
                })
        
        # Calculate ideal recovery time
        ideal_recovery = self._calculate_ideal_recovery_time(recovery_data)
        
        # Model recovery curve
        recovery_curve = self._model_recovery_curve(recovery_data)
        
        # Predict recovery status for upcoming matches
        upcoming_recovery = self._predict_upcoming_recovery(
            matches, reference_date, ideal_recovery
        )
        
        return {
            'recovery_data': recovery_data,
            'ideal_recovery_days': ideal_recovery,
            'recovery_curve': recovery_curve,
            'upcoming_recovery_status': upcoming_recovery,
            'recovery_efficiency_score': self._calculate_recovery_efficiency(recovery_data)
        }
    
    def _calculate_fatigue_score(self, congestion_metrics: Dict, rest_analysis: Dict,
                               travel_analysis: Dict, frequency_impact: Dict,
                               league_calibration: Dict) -> Dict[str, Any]:
        """Calculate comprehensive fatigue score (0-100)"""
        
        # Base fatigue components (0-100 each)
        congestion_fatigue = self._calculate_congestion_fatigue_score(congestion_metrics)
        rest_fatigue = self._calculate_rest_fatigue_score(rest_analysis)
        travel_fatigue = self._calculate_travel_fatigue_score(travel_analysis)
        frequency_fatigue = self._calculate_frequency_fatigue_score(frequency_impact)
        
        # Weighted combination
        weights = {
            'congestion': 0.3,
            'rest': 0.3,
            'travel': 0.2,
            'frequency': 0.2
        }
        
        base_score = (
            congestion_fatigue * weights['congestion'] +
            rest_fatigue * weights['rest'] +
            travel_fatigue * weights['travel'] +
            frequency_fatigue * weights['frequency']
        )
        
        # Apply league calibration
        calibrated_score = base_score * league_calibration['fatigue_multiplier']
        
        # Ensure score is within 0-100 range
        final_score = max(0, min(100, calibrated_score))
        
        return {
            'overall_fatigue_score': round(final_score, 1),
            'component_scores': {
                'congestion_fatigue': round(congestion_fatigue, 1),
                'rest_fatigue': round(rest_fatigue, 1),
                'travel_fatigue': round(travel_fatigue, 1),
                'frequency_fatigue': round(frequency_fatigue, 1)
            },
            'base_score': round(base_score, 1),
            'league_multiplier': league_calibration['fatigue_multiplier'],
            'fatigue_level': self._classify_fatigue_level(final_score)
        }
    
    def compare_team_fatigue(self, home_analysis: Dict, away_analysis: Dict) -> Dict[str, Any]:
        """Compare fatigue levels between two teams"""
        
        home_fatigue = home_analysis.get('fatigue_score', {}).get('overall_fatigue_score', 0)
        away_fatigue = away_analysis.get('fatigue_score', {}).get('overall_fatigue_score', 0)
        
        fatigue_difference = abs(home_fatigue - away_fatigue)
        
        # Determine advantage
        if fatigue_difference < 5:
            advantage = 'balanced'
            advantage_team = None
        elif home_fatigue < away_fatigue:
            advantage = 'home'
            advantage_team = 'home'
        else:
            advantage = 'away'
            advantage_team = 'away'
        
        # Calculate fatigue impact on match prediction
        match_impact = self._calculate_match_fatigue_impact(
            home_fatigue, away_fatigue, fatigue_difference
        )
        
        return {
            'home_fatigue_score': home_fatigue,
            'away_fatigue_score': away_fatigue,
            'fatigue_difference': round(fatigue_difference, 1),
            'advantage': advantage,
            'advantage_team': advantage_team,
            'advantage_significance': self._classify_advantage_significance(fatigue_difference),
            'match_impact': match_impact,
            'prediction_adjustments': self._suggest_prediction_adjustments(
                home_fatigue, away_fatigue, advantage
            )
        }
    
    # Helper methods for calculations and analysis
    
    def _get_league_calibration(self, league_id: Optional[str]) -> Dict[str, float]:
        """Get league-specific calibration parameters"""
        if league_id and str(league_id) in self.league_calibration:
            return self.league_calibration[str(league_id)]
        return self.league_calibration['default']
    
    def _sort_matches_by_date(self, matches: List[Dict]) -> List[Dict]:
        """Sort matches by date (most recent first)"""
        return sorted(matches, 
                     key=lambda x: self._get_match_date(x) or datetime.min, 
                     reverse=True)
    
    def _get_match_date(self, match: Dict) -> Optional[datetime]:
        """Extract match date from match data"""
        try:
            # Try different possible date fields
            date_fields = ['fixture.timestamp', 'timestamp', 'date', 'fixture.date']
            
            for field in date_fields:
                if '.' in field:
                    # Nested field
                    keys = field.split('.')
                    value = match
                    for key in keys:
                        value = value.get(key, {})
                        if not isinstance(value, dict) and key != keys[-1]:
                            break
                    if value:
                        if isinstance(value, (int, float)):
                            return datetime.fromtimestamp(value)
                        elif isinstance(value, str):
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    # Direct field
                    value = match.get(field)
                    if value:
                        if isinstance(value, (int, float)):
                            return datetime.fromtimestamp(value)
                        elif isinstance(value, str):
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
            
            return None
        except:
            return None
    
    def _analyze_match_distribution(self, matches: List[Dict], days: int) -> Dict[str, Any]:
        """Analyze how matches are distributed over the time period"""
        if not matches:
            return {'even_distribution': True, 'clusters': [], 'max_gap': 0}
        
        # Calculate gaps between matches
        match_dates = [self._get_match_date(match) for match in matches]
        match_dates = [date for date in match_dates if date is not None]
        match_dates.sort()
        
        gaps = []
        for i in range(len(match_dates) - 1):
            gap = (match_dates[i + 1] - match_dates[i]).days
            gaps.append(gap)
        
        # Identify clusters (matches close together)
        clusters = []
        current_cluster = [match_dates[0]] if match_dates else []
        
        for i, gap in enumerate(gaps):
            if gap <= 3:  # Matches within 3 days are clustered
                current_cluster.append(match_dates[i + 1])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [match_dates[i + 1]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return {
            'gaps_between_matches': gaps,
            'average_gap': mean(gaps) if gaps else 0,
            'max_gap': max(gaps) if gaps else 0,
            'min_gap': min(gaps) if gaps else 0,
            'clusters': len(clusters),
            'clustered_matches': sum(len(cluster) for cluster in clusters),
            'even_distribution': len(clusters) == 0 and len(set(gaps)) <= 2 if gaps else True
        }
    
    def _calculate_congestion_intensity(self, match_count: int, days: int, 
                                      distribution: Dict) -> float:
        """Calculate congestion intensity score"""
        # Base intensity from match frequency
        if days == 0:
            return 0
        
        matches_per_week = (match_count / days) * 7
        base_intensity = min(matches_per_week * 20, 80)  # Scale to 0-80
        
        # Adjust for distribution clustering
        cluster_penalty = distribution.get('clusters', 0) * 5
        
        # Gap variance penalty - safe check for empty list
        gaps = distribution.get('gaps_between_matches', [])
        if gaps and len(gaps) > 0:
            gap_variance_penalty = max(gaps) - min(gaps)
        else:
            gap_variance_penalty = 0
        
        intensity = base_intensity + cluster_penalty + (gap_variance_penalty * 0.5)
        return min(intensity, 100)
    
    def _classify_congestion_level(self, intensity: float) -> str:
        """Classify congestion level based on intensity score"""
        if intensity < 20:
            return 'low'
        elif intensity < 40:
            return 'moderate'
        elif intensity < 60:
            return 'high'
        elif intensity < 80:
            return 'very_high'
        else:
            return 'extreme'
    
    def _calculate_congestion_trend(self, congestion: Dict) -> Dict[str, Any]:
        """Calculate overall congestion trend"""
        windows = ['short', 'medium', 'long']
        intensities = [congestion[w]['intensity_score'] for w in windows if w in congestion]
        
        if len(intensities) < 2:
            return {'trend': 'stable', 'trend_strength': 0}
        
        # Simple trend analysis
        if intensities[0] > intensities[1] * 1.2:
            trend = 'increasing'
        elif intensities[0] < intensities[1] * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        trend_strength = abs(intensities[0] - intensities[-1]) / max(intensities)
        
        return {
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'current_intensity': intensities[0] if intensities else 0,
            'average_intensity': mean(intensities) if intensities else 0
        }
    
    def _extract_match_performance(self, match: Dict, team_id: int) -> Optional[Dict]:
        """Extract performance metrics from a match"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                return None
            
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                # Team played at home
                goals_for = home_goals
                goals_against = away_goals
                venue = 'home'
            elif away_team.get('id') == team_id:
                # Team played away
                goals_for = away_goals
                goals_against = home_goals
                venue = 'away'
            else:
                return None
            
            # Calculate result points
            if goals_for > goals_against:
                points = 3
                result = 'win'
            elif goals_for == goals_against:
                points = 1
                result = 'draw'
            else:
                points = 0
                result = 'loss'
            
            return {
                'points': points,
                'result': result,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goals_for - goals_against,
                'venue': venue,
                'performance_score': self._calculate_match_performance_score(
                    points, goals_for, goals_against
                )
            }
            
        except Exception as e:
            logger.error(f"Error extracting match performance: {str(e)}")
            return None
    
    def _calculate_match_performance_score(self, points: int, goals_for: int, 
                                         goals_against: int) -> float:
        """Calculate a composite performance score for a match"""
        # Base score from points (0-60)
        point_score = points * 20
        
        # Goal scoring bonus (0-20)
        goal_score = min(goals_for * 5, 20)
        
        # Defensive bonus (0-20)
        defensive_score = max(20 - goals_against * 5, 0)
        
        return point_score + goal_score + defensive_score
    
    def _categorize_rest_period(self, rest_days: int) -> str:
        """Categorize rest period into standard categories"""
        if rest_days <= 2:
            return 'very_short'
        elif rest_days <= 4:
            return 'short'
        elif rest_days <= 7:
            return 'normal'
        elif rest_days <= 14:
            return 'long'
        else:
            return 'very_long'
    
    def _calculate_rest_performance_correlation(self, performance_by_rest: Dict) -> Dict[str, Any]:
        """Calculate correlation between rest periods and performance"""
        correlations = {}
        
        for rest_category, performances in performance_by_rest.items():
            if performances:
                avg_performance = mean([p['performance_score'] for p in performances])
                avg_points = mean([p['points'] for p in performances])
                
                correlations[rest_category] = {
                    'average_performance_score': round(avg_performance, 1),
                    'average_points': round(avg_points, 2),
                    'sample_size': len(performances),
                    'win_rate': len([p for p in performances if p['result'] == 'win']) / len(performances)
                }
        
        # Find optimal rest category
        if correlations:
            optimal_category = max(correlations.keys(), 
                                 key=lambda k: correlations.get(k, {}).get('average_performance_score', 0))
        else:
            optimal_category = 'normal'
        
        return {
            'by_category': correlations,
            'optimal_rest_category': optimal_category,
            'performance_variance': self._calculate_performance_variance(correlations)
        }
    
    def _analyze_recent_rest_pattern(self, recent_matches: List[Dict], team_id: int) -> Dict[str, Any]:
        """Analyze recent rest patterns"""
        if len(recent_matches) < 3:
            return {'insufficient_data': True}
        
        rest_periods = []
        for i in range(len(recent_matches) - 1):
            current_match = recent_matches[i]
            next_match = recent_matches[i + 1]
            
            rest_days = self._calculate_rest_between_matches(current_match, next_match)
            if rest_days is not None:
                rest_periods.append(rest_days)
        
        if not rest_periods:
            return {'insufficient_data': True}
        
        return {
            'recent_rest_periods': rest_periods,
            'average_recent_rest': round(mean(rest_periods), 1),
            'minimum_recent_rest': min(rest_periods),
            'rest_consistency': self._calculate_rest_consistency(rest_periods),
            'trend': self._analyze_rest_trend(rest_periods)
        }
    
    def _assess_optimal_rest(self, rest_periods: List[int], performance_by_rest: Dict) -> Dict[str, Any]:
        """Assess optimal rest requirements for the team"""
        if not performance_by_rest:
            return {'optimal_days': 4, 'confidence': 'low'}
        
        # Find rest category with best performance
        best_category = None
        best_score = 0
        
        for category, data in performance_by_rest.items():
            if data and data['sample_size'] >= 2:  # Minimum sample size
                score = data['average_performance_score']
                if score > best_score:
                    best_score = score
                    best_category = category
        
        # Map category to days
        category_to_days = {
            'very_short': 2,
            'short': 3,
            'normal': 5,
            'long': 10,
            'very_long': 14
        }
        
        optimal_days = category_to_days.get(best_category, 4)
        
        # Calculate confidence based on sample size and performance difference
        confidence = self._calculate_optimal_rest_confidence(performance_by_rest, best_category)
        
        return {
            'optimal_days': optimal_days,
            'optimal_category': best_category,
            'confidence': confidence,
            'performance_at_optimal': best_score
        }
    
    def _predict_next_match_rest(self, matches: List[Dict], reference_date: datetime) -> Dict[str, Any]:
        """Predict rest days before next match"""
        if not matches:
            return {'prediction': 'unknown'}
        
        last_match_date = self._get_match_date(matches[0])
        if not last_match_date:
            return {'prediction': 'unknown'}
        
        days_since_last = (reference_date - last_match_date).days
        
        # Simple prediction based on recent patterns
        recent_gaps = []
        for i in range(min(5, len(matches) - 1)):
            current_match = matches[i]
            next_match = matches[i + 1]
            gap = self._calculate_rest_between_matches(current_match, next_match)
            if gap is not None:
                recent_gaps.append(gap)
        
        predicted_gap = round(mean(recent_gaps)) if recent_gaps else 7
        estimated_next_match = reference_date + timedelta(days=predicted_gap - days_since_last)
        
        return {
            'days_since_last_match': days_since_last,
            'predicted_rest_days': max(0, predicted_gap - days_since_last),
            'estimated_next_match_date': estimated_next_match.isoformat(),
            'prediction_confidence': 'medium' if len(recent_gaps) >= 3 else 'low'
        }
    
    def _calculate_match_travel(self, match: Dict, team_id: int) -> Optional[Dict[str, Any]]:
        """Calculate travel information for a match"""
        try:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            
            # Determine if team is home or away
            if home_team.get('id') == team_id:
                # Team is at home, no travel
                return {
                    'distance': 0,
                    'travel_zone': 'local',
                    'fatigue_impact': 0,
                    'is_home': True
                }
            elif away_team.get('id') == team_id:
                # Team is away, calculate travel
                home_city = self._get_team_city(home_team)
                away_city = self._get_team_city(away_team)
                
                if home_city and away_city:
                    distance = self._calculate_distance(away_city, home_city)
                    travel_zone = self._determine_travel_zone(distance)
                    fatigue_impact = self._calculate_travel_fatigue_impact(distance, travel_zone)
                    
                    return {
                        'distance': distance,
                        'travel_zone': travel_zone,
                        'fatigue_impact': fatigue_impact,
                        'is_home': False,
                        'from_city': away_city,
                        'to_city': home_city
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating match travel: {str(e)}")
            return None
    
    def _get_team_city(self, team: Dict) -> Optional[str]:
        """Get team's city from team data"""
        team_name = team.get('name', '').lower()
        
        # Simple city mapping based on team names
        city_mappings = {
            'arsenal': 'London', 'chelsea': 'London', 'tottenham': 'London',
            'west ham': 'London', 'fulham': 'London', 'brentford': 'London',
            'manchester united': 'Manchester', 'manchester city': 'Manchester',
            'liverpool': 'Liverpool', 'everton': 'Liverpool',
            'real madrid': 'Madrid', 'atletico madrid': 'Madrid',
            'barcelona': 'Barcelona',
            'ac milan': 'Milan', 'inter milan': 'Milan',
            'roma': 'Rome', 'lazio': 'Rome',
            'bayern munich': 'Munich', 'borussia dortmund': 'Dortmund',
            'psg': 'Paris', 'paris saint-germain': 'Paris',
            'galatasaray': 'Istanbul', 'fenerbahce': 'Istanbul',
            'besiktas': 'Istanbul', 'trabzonspor': 'Istanbul'
        }
        
        for key, city in city_mappings.items():
            if key in team_name:
                return city
        
        # Default to a major city if not found
        return 'London'  # Default assumption
    
    def _calculate_distance(self, city1: str, city2: str) -> float:
        """Calculate distance between two cities"""
        if city1 == city2:
            return 0
        
        coord1 = self.city_coordinates.get(city1)
        coord2 = self.city_coordinates.get(city2)
        
        if not coord1 or not coord2:
            # Return estimated distance based on zone
            return 500  # Default medium distance
        
        # Haversine formula for distance calculation
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def _determine_travel_zone(self, distance: float) -> str:
        """Determine travel zone based on distance"""
        for zone, info in self.travel_zones.items():
            if distance <= info['max_distance']:
                return zone
        return 'intercontinental'
    
    def _calculate_travel_fatigue_impact(self, distance: float, travel_zone: str) -> float:
        """Calculate fatigue impact from travel distance and zone"""
        zone_info = self.travel_zones.get(travel_zone, self.travel_zones['local'])
        
        # Base fatigue from zone
        base_fatigue = zone_info['fatigue_factor'] - 1  # 0 for local, up to 0.6 for intercontinental
        
        # Additional fatigue based on exact distance
        distance_factor = min(distance / 5000, 1)  # Scale up to 5000km
        
        return base_fatigue + (distance_factor * 0.5)
    
    def _analyze_travel_zones(self, travel_data: List[Dict]) -> Dict[str, Any]:
        """Analyze travel patterns by zones"""
        zone_counts = defaultdict(int)
        zone_distances = defaultdict(list)
        zone_fatigue = defaultdict(list)
        
        for travel in travel_data:
            zone = travel['travel_zone']
            zone_counts[zone] += 1
            zone_distances[zone].append(travel['distance'])
            zone_fatigue[zone].append(travel['fatigue_impact'])
        
        zone_analysis = {}
        for zone in zone_counts:
            zone_analysis[zone] = {
                'travel_count': zone_counts[zone],
                'average_distance': round(mean(zone_distances[zone]), 0) if zone_distances[zone] else 0,
                'total_distance': round(sum(zone_distances[zone]), 0),
                'average_fatigue': round(mean(zone_fatigue[zone]), 2) if zone_fatigue[zone] else 0,
                'total_fatigue': round(sum(zone_fatigue[zone]), 2)
            }
        
        return zone_analysis
    
    def _calculate_travel_coefficient(self, cumulative_fatigue: float, 
                                   total_distance: float, travel_count: int) -> float:
        """Calculate overall travel fatigue coefficient"""
        if travel_count == 0:
            return 0
        
        # Normalize factors
        fatigue_factor = min(cumulative_fatigue / 10, 1)  # Scale cumulative fatigue
        distance_factor = min(total_distance / 10000, 1)  # Scale total distance
        frequency_factor = min(travel_count / 10, 1)      # Scale travel frequency
        
        # Weighted combination
        coefficient = (fatigue_factor * 0.4 + distance_factor * 0.3 + frequency_factor * 0.3)
        
        return coefficient
    
    def _classify_travel_fatigue(self, coefficient: float) -> str:
        """Classify travel fatigue level"""
        if coefficient < 0.2:
            return 'minimal'
        elif coefficient < 0.4:
            return 'low'
        elif coefficient < 0.6:
            return 'moderate'
        elif coefficient < 0.8:
            return 'high'
        else:
            return 'severe'
    
    def _calculate_travel_recovery_penalty(self, travel_data: List[Dict]) -> float:
        """Calculate recovery penalty from recent travel"""
        if not travel_data:
            return 0
        
        penalty = 0
        for travel in travel_data[-5:]:  # Last 5 travels
            zone = travel['travel_zone']
            zone_info = self.travel_zones.get(zone, self.travel_zones['local'])
            penalty += zone_info['recovery_penalty']
        
        return penalty
    
    def _calculate_period_performance(self, matches: List[Dict], team_id: int) -> Dict[str, Any]:
        """Calculate average performance for a period"""
        if not matches:
            return {'points_per_game': 0, 'goals_per_game': 0, 'performance_score': 0}
        
        total_points = 0
        total_goals = 0
        total_performance = 0
        
        for match in matches:
            performance = self._extract_match_performance(match, team_id)
            if performance:
                total_points += performance['points']
                total_goals += performance['goals_for']
                total_performance += performance['performance_score']
        
        match_count = len([m for m in matches if self._extract_match_performance(m, team_id)])
        
        if match_count == 0:
            return {'points_per_game': 0, 'goals_per_game': 0, 'performance_score': 0}
        
        return {
            'points_per_game': round(total_points / match_count, 2),
            'goals_per_game': round(total_goals / match_count, 2),
            'performance_score': round(total_performance / match_count, 1)
        }
    
    def _categorize_frequency(self, matches_per_week: float) -> str:
        """Categorize match frequency"""
        if matches_per_week < 1:
            return 'very_low'
        elif matches_per_week < 1.5:
            return 'low'
        elif matches_per_week < 2.5:
            return 'normal'
        elif matches_per_week < 3.5:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_frequency_impact_score(self, frequency_periods: List[Dict]) -> float:
        """Calculate impact score of match frequency on performance"""
        if not frequency_periods:
            return 0
        
        # Find optimal frequency point and calculate deviation
        best_performance = max(period['average_performance']['performance_score'] 
                             for period in frequency_periods)
        
        impact_scores = []
        for period in frequency_periods:
            performance = period['average_performance']['performance_score']
            deviation = abs(performance - best_performance) / best_performance
            impact_scores.append(deviation)
        
        return mean(impact_scores) * 100  # Scale to 0-100
    
    def _determine_optimal_frequency(self, performance_by_frequency: Dict) -> Dict[str, Any]:
        """Determine optimal match frequency range"""
        if not performance_by_frequency:
            return {'optimal_range': 'normal', 'confidence': 'low'}
        
        best_category = None
        best_score = 0
        
        for category, performances in performance_by_frequency.items():
            if performances:
                avg_score = mean([p['performance_score'] for p in performances])
                if avg_score > best_score:
                    best_score = avg_score
                    best_category = category
        
        return {
            'optimal_range': best_category or 'normal',
            'optimal_performance_score': best_score,
            'confidence': 'high' if len(performance_by_frequency) >= 3 else 'medium'
        }
    
    def _assess_current_frequency(self, frequency_periods: List[Dict]) -> Dict[str, Any]:
        """Assess current frequency level"""
        if not frequency_periods:
            return {'assessment': 'unknown'}
        
        # Use shortest period (most recent)
        current_period = frequency_periods[0]
        frequency = current_period['frequency_per_week']
        category = self._categorize_frequency(frequency)
        
        return {
            'current_frequency': frequency,
            'category': category,
            'assessment': self._assess_frequency_appropriateness(category),
            'recommendation': self._recommend_frequency_adjustment(category)
        }
    
    def _calculate_match_intensity(self, match: Dict, team_id: int) -> float:
        """Calculate the intensity/importance of a match"""
        # Base intensity
        intensity = 50
        
        # Competition importance
        competition = match.get('competition', {})
        comp_name = competition.get('name', '').lower()
        
        if 'champions league' in comp_name:
            intensity += 30
        elif 'europa league' in comp_name:
            intensity += 20
        elif 'cup' in comp_name or 'final' in comp_name:
            intensity += 15
        elif 'derby' in comp_name or 'clasico' in comp_name:
            intensity += 25
        
        # Match context (from score/result)
        performance = self._extract_match_performance(match, team_id)
        if performance:
            # Close matches are more intense
            goal_diff = abs(performance['goal_difference'])
            if goal_diff <= 1:
                intensity += 10
            
            # Results impact
            if performance['result'] == 'win':
                intensity += 5
            elif performance['result'] == 'loss':
                intensity += 10  # Losses are often more draining
        
        return min(intensity, 100)
    
    def _calculate_rest_between_matches(self, match1: Dict, match2: Dict) -> Optional[int]:
        """Calculate rest days between two matches"""
        date1 = self._get_match_date(match1)
        date2 = self._get_match_date(match2)
        
        if date1 and date2:
            return abs((date1 - date2).days)
        return None
    
    def _assess_recovery_quality(self, match1: Dict, match2: Dict, team_id: int, 
                               rest_days: int) -> float:
        """Assess the quality of recovery between matches"""
        # Base recovery score
        recovery_score = 50
        
        # Rest days impact
        if rest_days >= 5:
            recovery_score += 30
        elif rest_days >= 3:
            recovery_score += 20
        elif rest_days >= 2:
            recovery_score += 10
        else:
            recovery_score -= 20  # Very short rest
        
        # Match intensity impact
        match1_intensity = self._calculate_match_intensity(match1, team_id)
        if match1_intensity > 80:
            recovery_score -= 15
        elif match1_intensity > 60:
            recovery_score -= 10
        
        # Travel impact
        travel_info = self._calculate_match_travel(match2, team_id)
        if travel_info and travel_info['fatigue_impact'] > 0.5:
            recovery_score -= 10
        
        return max(0, min(100, recovery_score))
    
    def _calculate_ideal_recovery_time(self, recovery_data: List[Dict]) -> float:
        """Calculate ideal recovery time based on historical data"""
        if not recovery_data:
            return 4  # Default
        
        # Find rest periods that led to best performance
        good_recoveries = [
            r for r in recovery_data 
            if r.get('recovery_quality', 0) > 70 and r.get('next_match_performance')
        ]
        
        if good_recoveries:
            good_rest_periods = [r['rest_days'] for r in good_recoveries]
            return mean(good_rest_periods)
        
        # Fallback to average rest that led to decent performance
        decent_recoveries = [
            r for r in recovery_data 
            if r.get('recovery_quality', 0) > 50
        ]
        
        if decent_recoveries:
            decent_rest_periods = [r['rest_days'] for r in decent_recoveries]
            return mean(decent_rest_periods)
        
        return 4  # Default
    
    def _model_recovery_curve(self, recovery_data: List[Dict]) -> Dict[str, Any]:
        """Model the recovery curve based on rest days"""
        if not recovery_data:
            return {'curve_type': 'linear', 'parameters': {}}
        
        # Group by rest days and calculate average recovery quality
        rest_to_quality = defaultdict(list)
        for r in recovery_data:
            rest_days = r['rest_days']
            quality = r.get('recovery_quality', 50)
            rest_to_quality[rest_days].append(quality)
        
        # Calculate averages
        curve_points = {}
        for rest_days, qualities in rest_to_quality.items():
            curve_points[rest_days] = mean(qualities)
        
        # Simple curve analysis
        if len(curve_points) >= 3:
            rest_days = sorted(curve_points.keys())
            qualities = [curve_points[rd] for rd in rest_days]
            
            # Check if curve saturates (diminishing returns)
            saturation_point = None
            for i in range(1, len(qualities)):
                if qualities[i] - qualities[i-1] < 5:  # Small improvement
                    saturation_point = rest_days[i]
                    break
            
            return {
                'curve_points': curve_points,
                'saturation_point': saturation_point,
                'optimal_range': [rd for rd in rest_days if curve_points[rd] > 75]
            }
        
        return {'curve_type': 'insufficient_data', 'parameters': {}}
    
    def _predict_upcoming_recovery(self, matches: List[Dict], reference_date: datetime,
                                 ideal_recovery: float) -> Dict[str, Any]:
        """Predict recovery status for upcoming period"""
        if not matches:
            return {'status': 'unknown'}
        
        last_match_date = self._get_match_date(matches[0])
        if not last_match_date:
            return {'status': 'unknown'}
        
        days_since_last = (reference_date - last_match_date).days
        
        # Recovery status
        if days_since_last >= ideal_recovery:
            status = 'fully_recovered'
        elif days_since_last >= ideal_recovery * 0.7:
            status = 'mostly_recovered'
        elif days_since_last >= ideal_recovery * 0.4:
            status = 'partially_recovered'
        else:
            status = 'insufficient_recovery'
        
        recovery_percentage = min(100, (days_since_last / ideal_recovery) * 100)
        
        return {
            'status': status,
            'days_since_last_match': days_since_last,
            'ideal_recovery_days': ideal_recovery,
            'recovery_percentage': round(recovery_percentage, 1),
            'estimated_full_recovery_date': (last_match_date + timedelta(days=ideal_recovery)).isoformat()
        }
    
    def _calculate_recovery_efficiency(self, recovery_data: List[Dict]) -> float:
        """Calculate team's recovery efficiency score"""
        if not recovery_data:
            return 50
        
        # Calculate average recovery quality vs rest time
        efficiency_scores = []
        
        for r in recovery_data:
            rest_days = r['rest_days']
            quality = r.get('recovery_quality', 50)
            
            # Expected quality based on rest days
            if rest_days <= 2:
                expected_quality = 30
            elif rest_days <= 4:
                expected_quality = 60
            elif rest_days <= 7:
                expected_quality = 80
            else:
                expected_quality = 90
            
            # Efficiency is actual vs expected
            efficiency = quality / expected_quality
            efficiency_scores.append(efficiency)
        
        avg_efficiency = mean(efficiency_scores)
        return min(100, avg_efficiency * 100)
    
    # Fatigue score calculation methods
    
    def _calculate_congestion_fatigue_score(self, congestion_metrics: Dict) -> float:
        """Calculate fatigue score from congestion metrics"""
        if not congestion_metrics:
            return 0
        
        # Weight different time windows
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        
        fatigue_score = 0
        for window, weight in weights.items():
            if window in congestion_metrics:
                intensity = congestion_metrics[window]['intensity_score']
                fatigue_score += intensity * weight
        
        return fatigue_score
    
    def _calculate_rest_fatigue_score(self, rest_analysis: Dict) -> float:
        """Calculate fatigue score from rest analysis"""
        if 'rest_statistics' not in rest_analysis:
            return 50
        
        avg_rest = rest_analysis['rest_statistics']['average_rest_days']
        min_rest = rest_analysis['rest_statistics']['minimum_rest']
        
        # Lower rest = higher fatigue
        if avg_rest >= 5:
            base_score = 20
        elif avg_rest >= 3:
            base_score = 40
        elif avg_rest >= 2:
            base_score = 60
        else:
            base_score = 80
        
        # Penalty for very short minimum rest
        if min_rest == 0:
            base_score += 20
        elif min_rest == 1:
            base_score += 10
        
        return min(100, base_score)
    
    def _calculate_travel_fatigue_score(self, travel_analysis: Dict) -> float:
        """Calculate fatigue score from travel analysis"""
        if 'travel_coefficient' not in travel_analysis:
            return 0
        
        coefficient = travel_analysis['travel_coefficient']
        return coefficient * 100
    
    def _calculate_frequency_fatigue_score(self, frequency_impact: Dict) -> float:
        """Calculate fatigue score from frequency impact"""
        if 'impact_score' not in frequency_impact:
            return 50
        
        return frequency_impact['impact_score']
    
    def _classify_fatigue_level(self, fatigue_score: float) -> str:
        """Classify overall fatigue level"""
        if fatigue_score < 30:
            return 'fresh'
        elif fatigue_score < 50:
            return 'normal'
        elif fatigue_score < 70:
            return 'tired'
        elif fatigue_score < 85:
            return 'very_tired'
        else:
            return 'exhausted'
    
    def _identify_performance_windows(self, rest_analysis: Dict, recovery_analysis: Dict,
                                    reference_date: datetime) -> Dict[str, Any]:
        """Identify optimal performance windows"""
        windows = {
            'current_status': 'unknown',
            'next_optimal_window': None,
            'performance_trend': 'stable'
        }
        
        if 'upcoming_recovery_status' in recovery_analysis:
            recovery_status = recovery_analysis['upcoming_recovery_status']
            
            if recovery_status['recovery_percentage'] >= 90:
                windows['current_status'] = 'optimal'
            elif recovery_status['recovery_percentage'] >= 70:
                windows['current_status'] = 'good'
            elif recovery_status['recovery_percentage'] >= 50:
                windows['current_status'] = 'declining'
            else:
                windows['current_status'] = 'poor'
            
            # Predict next optimal window
            if recovery_status['status'] != 'fully_recovered':
                full_recovery_date = datetime.fromisoformat(
                    recovery_status['estimated_full_recovery_date']
                )
                optimal_start = full_recovery_date + timedelta(days=1)
                optimal_end = full_recovery_date + timedelta(days=7)
                
                windows['next_optimal_window'] = {
                    'start_date': optimal_start.isoformat(),
                    'end_date': optimal_end.isoformat(),
                    'days_until_optimal': (optimal_start - reference_date).days
                }
        
        return windows
    
    def _generate_recommendations(self, fatigue_score: Dict, congestion_metrics: Dict,
                                rest_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        overall_fatigue = fatigue_score.get('overall_fatigue_score', 0)
        
        # General fatigue recommendations
        if overall_fatigue > 85:
            recommendations.append("CRITICAL: Team shows signs of extreme fatigue. Consider squad rotation.")
            recommendations.append("Implement intensive recovery protocols including extra rest days.")
        elif overall_fatigue > 70:
            recommendations.append("HIGH: Significant fatigue detected. Monitor player fitness closely.")
            recommendations.append("Consider tactical adjustments to preserve energy.")
        elif overall_fatigue > 50:
            recommendations.append("MODERATE: Some fatigue accumulation. Plan recovery carefully.")
        
        # Congestion-specific recommendations
        if congestion_metrics.get('overall_trend', {}).get('trend') == 'increasing':
            recommendations.append("Fixture congestion is increasing. Prioritize squad depth.")
        
        # Rest-specific recommendations
        if 'rest_statistics' in rest_analysis:
            avg_rest = rest_analysis['rest_statistics']['average_rest_days']
            if avg_rest < 3:
                recommendations.append("Very short average rest periods. Consider training load reduction.")
        
        # Travel-specific recommendations would be added here
        
        if not recommendations:
            recommendations.append("Team fatigue levels are within normal ranges.")
        
        return recommendations
    
    def _assess_risk_level(self, fatigue_score: float) -> str:
        """Assess injury/performance risk level"""
        if fatigue_score < 40:
            return 'low'
        elif fatigue_score < 60:
            return 'moderate'
        elif fatigue_score < 80:
            return 'high'
        else:
            return 'critical'
    
    def _get_league_baseline(self, league_id: Optional[str]) -> Dict[str, Any]:
        """Get baseline fatigue metrics for the league"""
        league_cal = self._get_league_calibration(league_id)
        
        return {
            'average_fatigue_score': 45 * league_cal['fatigue_multiplier'],
            'typical_recovery_days': league_cal['recovery_base'],
            'travel_impact_factor': league_cal['travel_impact'],
            'league_intensity_level': self._classify_league_intensity(league_cal)
        }
    
    def _classify_league_intensity(self, league_cal: Dict) -> str:
        """Classify league intensity level"""
        multiplier = league_cal['fatigue_multiplier']
        
        if multiplier >= 1.3:
            return 'very_high'
        elif multiplier >= 1.1:
            return 'high'
        elif multiplier >= 1.0:
            return 'normal'
        else:
            return 'low'
    
    # Match impact calculation methods
    
    def _calculate_match_fatigue_impact(self, home_fatigue: float, away_fatigue: float,
                                      fatigue_difference: float) -> Dict[str, Any]:
        """Calculate how fatigue difference impacts the match"""
        
        # Base impact on team performance
        home_impact = self._calculate_team_fatigue_impact(home_fatigue)
        away_impact = self._calculate_team_fatigue_impact(away_fatigue)
        
        # Relative advantage calculation
        if fatigue_difference < 5:
            relative_impact = 'minimal'
            advantage_magnitude = 0
        elif fatigue_difference < 15:
            relative_impact = 'small'
            advantage_magnitude = 0.05
        elif fatigue_difference < 25:
            relative_impact = 'moderate'
            advantage_magnitude = 0.10
        else:
            relative_impact = 'significant'
            advantage_magnitude = 0.15
        
        return {
            'home_performance_impact': home_impact,
            'away_performance_impact': away_impact,
            'relative_impact': relative_impact,
            'advantage_magnitude': advantage_magnitude,
            'expected_goal_adjustment': self._calculate_goal_adjustment(
                home_fatigue, away_fatigue, advantage_magnitude
            )
        }
    
    def _calculate_team_fatigue_impact(self, fatigue_score: float) -> Dict[str, Any]:
        """Calculate individual team fatigue impact"""
        if fatigue_score < 30:
            performance_multiplier = 1.05  # Fresh team bonus
            injury_risk = 'very_low'
        elif fatigue_score < 50:
            performance_multiplier = 1.0   # Normal performance
            injury_risk = 'low'
        elif fatigue_score < 70:
            performance_multiplier = 0.95  # Slight decrease
            injury_risk = 'moderate'
        elif fatigue_score < 85:
            performance_multiplier = 0.90  # Noticeable decrease
            injury_risk = 'high'
        else:
            performance_multiplier = 0.80  # Significant decrease
            injury_risk = 'very_high'
        
        return {
            'performance_multiplier': performance_multiplier,
            'injury_risk': injury_risk,
            'concentration_level': self._assess_concentration_level(fatigue_score),
            'physical_capacity': self._assess_physical_capacity(fatigue_score)
        }
    
    def _calculate_goal_adjustment(self, home_fatigue: float, away_fatigue: float,
                                 advantage_magnitude: float) -> Dict[str, float]:
        """Calculate goal expectation adjustments based on fatigue"""
        
        # Base adjustments
        home_adjustment = 0
        away_adjustment = 0
        
        # Apply fatigue effects
        if home_fatigue > away_fatigue:
            # Home team more tired
            home_adjustment = -advantage_magnitude
            away_adjustment = advantage_magnitude * 0.5
        else:
            # Away team more tired
            away_adjustment = -advantage_magnitude
            home_adjustment = advantage_magnitude * 0.5
        
        return {
            'home_goal_adjustment': round(home_adjustment, 3),
            'away_goal_adjustment': round(away_adjustment, 3)
        }
    
    def _classify_advantage_significance(self, fatigue_difference: float) -> str:
        """Classify the significance of fatigue advantage"""
        if fatigue_difference < 5:
            return 'negligible'
        elif fatigue_difference < 15:
            return 'minor'
        elif fatigue_difference < 25:
            return 'moderate'
        elif fatigue_difference < 35:
            return 'significant'
        else:
            return 'major'
    
    def _suggest_prediction_adjustments(self, home_fatigue: float, away_fatigue: float,
                                      advantage: str) -> Dict[str, Any]:
        """Suggest adjustments to match predictions based on fatigue"""
        adjustments = {
            'win_probability_adjustment': 0,
            'total_goals_adjustment': 0,
            'tactical_considerations': []
        }
        
        fatigue_diff = abs(home_fatigue - away_fatigue)
        
        if advantage == 'home' and fatigue_diff > 10:
            adjustments['win_probability_adjustment'] = min(fatigue_diff * 0.2, 8)
        elif advantage == 'away' and fatigue_diff > 10:
            adjustments['win_probability_adjustment'] = -min(fatigue_diff * 0.2, 8)
        
        # Total goals adjustment
        avg_fatigue = (home_fatigue + away_fatigue) / 2
        if avg_fatigue > 70:
            adjustments['total_goals_adjustment'] = -0.3  # Tired teams score less
        elif avg_fatigue < 30:
            adjustments['total_goals_adjustment'] = 0.2   # Fresh teams more attacking
        
        # Tactical considerations
        if home_fatigue > 75:
            adjustments['tactical_considerations'].append("Home team likely to sit deeper")
        if away_fatigue > 75:
            adjustments['tactical_considerations'].append("Away team may struggle with pressing")
        if max(home_fatigue, away_fatigue) > 85:
            adjustments['tactical_considerations'].append("Expect more substitutions")
        
        return adjustments
    
    # Additional helper methods
    
    def _assess_concentration_level(self, fatigue_score: float) -> str:
        """Assess concentration level based on fatigue"""
        if fatigue_score < 40:
            return 'high'
        elif fatigue_score < 70:
            return 'normal'
        else:
            return 'impaired'
    
    def _assess_physical_capacity(self, fatigue_score: float) -> str:
        """Assess physical capacity based on fatigue"""
        if fatigue_score < 30:
            return 'peak'
        elif fatigue_score < 60:
            return 'good'
        elif fatigue_score < 80:
            return 'reduced'
        else:
            return 'severely_reduced'
    
    def _assess_frequency_appropriateness(self, category: str) -> str:
        """Assess if current frequency is appropriate"""
        appropriateness_map = {
            'very_low': 'below_optimal',
            'low': 'below_optimal',
            'normal': 'appropriate',
            'high': 'above_optimal',
            'very_high': 'concerning'
        }
        return appropriateness_map.get(category, 'unknown')
    
    def _recommend_frequency_adjustment(self, category: str) -> str:
        """Recommend frequency adjustments"""
        recommendations = {
            'very_low': 'Consider scheduling more matches for rhythm',
            'low': 'Current frequency is manageable',
            'normal': 'Optimal frequency for most teams',
            'high': 'Monitor fatigue levels closely',
            'very_high': 'Reduce match frequency or increase squad rotation'
        }
        return recommendations.get(category, 'Maintain current frequency')
    
    def _calculate_performance_variance(self, correlations: Dict) -> float:
        """Calculate variance in performance across different rest categories"""
        if not correlations:
            return 0
        
        scores = [data['average_performance_score'] for data in correlations.values()]
        if len(scores) < 2:
            return 0
        
        mean_score = mean(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return round(variance, 2)
    
    def _calculate_rest_consistency(self, rest_periods: List[int]) -> float:
        """Calculate consistency of rest periods"""
        if len(rest_periods) < 2:
            return 1.0
        
        try:
            mean_rest = mean(rest_periods)
        except:
            return 1.0
            
        if mean_rest == 0:
            return 1.0
            
        variance = sum((rest - mean_rest) ** 2 for rest in rest_periods) / len(rest_periods)
        coefficient_of_variation = (variance ** 0.5) / mean_rest if mean_rest > 0 else 0
        
        # Convert to consistency score (0-1, higher is more consistent)
        consistency = max(0, 1 - coefficient_of_variation)
        return round(consistency, 2)
    
    def _analyze_rest_trend(self, rest_periods: List[int]) -> str:
        """Analyze trend in rest periods"""
        if len(rest_periods) < 3:
            return 'insufficient_data'
        
        # Simple trend analysis
        first_half = rest_periods[:len(rest_periods)//2]
        second_half = rest_periods[len(rest_periods)//2:]
        
        if not first_half or not second_half:
            return 'insufficient_data'
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        if second_avg > first_avg * 1.2:
            return 'increasing'
        elif second_avg < first_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_optimal_rest_confidence(self, performance_by_rest: Dict, 
                                         best_category: Optional[str]) -> str:
        """Calculate confidence in optimal rest assessment"""
        if not best_category or best_category not in performance_by_rest:
            return 'low'
        
        sample_size = performance_by_rest[best_category]['sample_size']
        performance_difference = 0
        
        # Calculate performance difference from other categories
        best_score = performance_by_rest[best_category]['average_performance_score']
        other_scores = [
            data['average_performance_score'] 
            for cat, data in performance_by_rest.items() 
            if cat != best_category
        ]
        
        if other_scores:
            avg_other_score = mean(other_scores)
            performance_difference = (best_score - avg_other_score) / best_score
        
        # Determine confidence
        if sample_size >= 5 and performance_difference > 0.1:
            return 'high'
        elif sample_size >= 3 and performance_difference > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _get_default_congestion_metrics(self) -> Dict[str, Any]:
        """Return default congestion metrics when calculation fails"""
        return {
            '7_days': {'match_count': 0, 'intensity': 0, 'distribution': {'gaps_between_matches': []}},
            '14_days': {'match_count': 0, 'intensity': 0, 'distribution': {'gaps_between_matches': []}},
            '21_days': {'match_count': 0, 'intensity': 0, 'distribution': {'gaps_between_matches': []}}
        }
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when data is insufficient"""
        return {
            'error': 'insufficient_data',
            'fatigue_score': {
                'overall_fatigue_score': 50,
                'fatigue_level': 'normal'
            },
            'risk_level': 'moderate',
            'recommendations': ['Insufficient data for comprehensive analysis']
        }