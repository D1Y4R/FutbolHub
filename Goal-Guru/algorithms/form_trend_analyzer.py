"""
Form Trend Analysis Module for Football Prediction System
Implements sophisticated form and momentum analysis for Phase 3.1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FormTrendAnalyzer:
    """Advanced form and trend analysis for football predictions"""
    
    def __init__(self):
        self.window_sizes = {
            'short': 5,    # Last 5 matches
            'medium': 10,  # Last 10 matches
            'long': 20     # Last 20 matches
        }
        logger.info("FormTrendAnalyzer initialized")
    
    def analyze_team_form(self, matches: List[Dict], team_id: int) -> Dict:
        """
        Comprehensive form analysis for a team
        
        Args:
            matches: List of match data
            team_id: ID of the team to analyze
            
        Returns:
            Dict containing form metrics
        """
        if not matches:
            return self._get_default_form()
            
        try:
            # Sort matches by date (most recent first)
            sorted_matches = sorted(matches, key=lambda x: x.get('fixture', {}).get('timestamp', 0), reverse=True)
            
            # Calculate various form metrics
            form_metrics = {
                'rolling_windows': self._calculate_rolling_windows(sorted_matches, team_id),
                'weighted_performance': self._calculate_weighted_performance(sorted_matches, team_id),
                'streak_analysis': self._analyze_streaks(sorted_matches, team_id),
                'momentum_shifts': self._detect_momentum_shifts(sorted_matches, team_id),
                'venue_specific': self._analyze_venue_form(sorted_matches, team_id),
                'opponent_adjusted': self._calculate_opponent_adjusted_metrics(sorted_matches, team_id),
                'form_stability': self._calculate_form_stability(sorted_matches, team_id),
                'scoring_trends': self._analyze_scoring_trends(sorted_matches, team_id),
                'defensive_trends': self._analyze_defensive_trends(sorted_matches, team_id),
                'pressure_performance': self._analyze_pressure_performance(sorted_matches, team_id)
            }
            
            # Calculate overall form score (0-100)
            form_metrics['overall_form_score'] = self._calculate_overall_form_score(form_metrics)
            
            # Determine form trajectory
            form_metrics['trajectory'] = self._determine_trajectory(form_metrics)
            
            return form_metrics
            
        except Exception as e:
            logger.error(f"Error in form analysis: {str(e)}")
            return self._get_default_form()
    
    def _calculate_rolling_windows(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate form for different rolling windows"""
        windows = {}
        
        for window_name, window_size in self.window_sizes.items():
            window_matches = matches[:window_size]
            if not window_matches:
                windows[window_name] = {'points_per_game': 0, 'goals_per_game': 0, 'goals_against_per_game': 0}
                continue
                
            points = 0
            goals_for = 0
            goals_against = 0
            
            for match in window_matches:
                home_team = match.get('teams', {}).get('home', {})
                away_team = match.get('teams', {}).get('away', {})
                score = match.get('score', {}).get('fulltime', {})
                
                if not score:
                    continue
                    
                home_goals = score.get('home', 0) or 0
                away_goals = score.get('away', 0) or 0
                
                if home_team.get('id') == team_id:
                    goals_for += home_goals
                    goals_against += away_goals
                    if home_goals > away_goals:
                        points += 3
                    elif home_goals == away_goals:
                        points += 1
                elif away_team.get('id') == team_id:
                    goals_for += away_goals
                    goals_against += home_goals
                    if away_goals > home_goals:
                        points += 3
                    elif away_goals == home_goals:
                        points += 1
            
            num_matches = len(window_matches)
            windows[window_name] = {
                'points_per_game': points / num_matches if num_matches > 0 else 0,
                'goals_per_game': goals_for / num_matches if num_matches > 0 else 0,
                'goals_against_per_game': goals_against / num_matches if num_matches > 0 else 0,
                'win_rate': (points / (num_matches * 3)) if num_matches > 0 else 0
            }
        
        return windows
    
    def _calculate_weighted_performance(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate performance with recency weighting"""
        if not matches:
            return {'weighted_points': 0, 'weighted_goals': 0, 'weighted_defense': 0}
            
        # Exponential decay weighting (more recent matches weighted higher)
        weights = [0.95 ** i for i in range(min(len(matches), 20))]
        total_weight = sum(weights)
        
        weighted_points = 0
        weighted_goals = 0
        weighted_defense = 0
        
        for i, match in enumerate(matches[:20]):
            weight = weights[i] if i < len(weights) else 0
            
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                # Home team
                if home_goals > away_goals:
                    weighted_points += 3 * weight
                elif home_goals == away_goals:
                    weighted_points += 1 * weight
                weighted_goals += home_goals * weight
                weighted_defense += (3 - away_goals) * weight  # Inverted for defense
            elif away_team.get('id') == team_id:
                # Away team
                if away_goals > home_goals:
                    weighted_points += 3 * weight
                elif away_goals == home_goals:
                    weighted_points += 1 * weight
                weighted_goals += away_goals * weight
                weighted_defense += (3 - home_goals) * weight
        
        return {
            'weighted_points': weighted_points / total_weight if total_weight > 0 else 0,
            'weighted_goals': weighted_goals / total_weight if total_weight > 0 else 0,
            'weighted_defense': weighted_defense / total_weight if total_weight > 0 else 0,
            'performance_index': (weighted_points + weighted_goals + weighted_defense) / 3 if total_weight > 0 else 0
        }
    
    def _analyze_streaks(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze winning/losing/unbeaten streaks"""
        current_streak = {'type': None, 'length': 0}
        longest_win_streak = 0
        longest_unbeaten = 0
        current_unbeaten = 0
        
        for match in matches:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            result = None
            if home_team.get('id') == team_id:
                if home_goals > away_goals:
                    result = 'W'
                elif home_goals == away_goals:
                    result = 'D'
                else:
                    result = 'L'
            elif away_team.get('id') == team_id:
                if away_goals > home_goals:
                    result = 'W'
                elif away_goals == home_goals:
                    result = 'D'
                else:
                    result = 'L'
            
            if result:
                # Update current streak
                if current_streak['type'] == result and result in ['W', 'L']:
                    current_streak['length'] += 1
                else:
                    current_streak = {'type': result, 'length': 1}
                
                # Update unbeaten streak
                if result in ['W', 'D']:
                    current_unbeaten += 1
                    longest_unbeaten = max(longest_unbeaten, current_unbeaten)
                else:
                    current_unbeaten = 0
                
                # Update longest win streak
                if result == 'W':
                    if current_streak['type'] == 'W':
                        longest_win_streak = max(longest_win_streak, current_streak['length'])
        
        return {
            'current_streak': current_streak,
            'longest_win_streak': longest_win_streak,
            'longest_unbeaten': longest_unbeaten,
            'current_unbeaten': current_unbeaten,
            'streak_momentum': self._calculate_streak_momentum(current_streak)
        }
    
    def _detect_momentum_shifts(self, matches: List[Dict], team_id: int) -> Dict:
        """Detect momentum shifts in recent performance"""
        if len(matches) < 6:
            return {'momentum_change': 0, 'trend': 'stable', 'turning_point': None}
            
        # Calculate points in 3-match segments
        segments = []
        for i in range(0, min(12, len(matches)), 3):
            segment_matches = matches[i:i+3]
            points = 0
            
            for match in segment_matches:
                home_team = match.get('teams', {}).get('home', {})
                away_team = match.get('teams', {}).get('away', {})
                score = match.get('score', {}).get('fulltime', {})
                
                if not score:
                    continue
                    
                home_goals = score.get('home', 0) or 0
                away_goals = score.get('away', 0) or 0
                
                if home_team.get('id') == team_id:
                    if home_goals > away_goals:
                        points += 3
                    elif home_goals == away_goals:
                        points += 1
                elif away_team.get('id') == team_id:
                    if away_goals > home_goals:
                        points += 3
                    elif away_goals == home_goals:
                        points += 1
            
            segments.append(points / len(segment_matches))
        
        if len(segments) < 2:
            return {'momentum_change': 0, 'trend': 'stable', 'turning_point': None}
            
        # Calculate momentum change
        recent_avg = np.mean(segments[:2])
        older_avg = np.mean(segments[2:]) if len(segments) > 2 else segments[1]
        momentum_change = recent_avg - older_avg
        
        # Determine trend
        if momentum_change > 0.5:
            trend = 'improving'
        elif momentum_change < -0.5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Find turning point
        turning_point = None
        if len(segments) >= 3:
            for i in range(1, len(segments) - 1):
                if (segments[i-1] < segments[i] > segments[i+1]) or (segments[i-1] > segments[i] < segments[i+1]):
                    turning_point = i * 3  # Match index of turning point
                    break
        
        return {
            'momentum_change': momentum_change,
            'trend': trend,
            'turning_point': turning_point,
            'recent_form_avg': recent_avg,
            'older_form_avg': older_avg,
            'volatility': np.std(segments) if len(segments) > 1 else 0
        }
    
    def _analyze_venue_form(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze form based on venue (home/away)"""
        home_stats = {'played': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0}
        away_stats = {'played': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0}
        
        for match in matches[:20]:  # Last 20 matches
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                # Playing at home
                home_stats['played'] += 1
                home_stats['goals_for'] += home_goals
                home_stats['goals_against'] += away_goals
                if home_goals > away_goals:
                    home_stats['points'] += 3
                elif home_goals == away_goals:
                    home_stats['points'] += 1
            elif away_team.get('id') == team_id:
                # Playing away
                away_stats['played'] += 1
                away_stats['goals_for'] += away_goals
                away_stats['goals_against'] += home_goals
                if away_goals > home_goals:
                    away_stats['points'] += 3
                elif away_goals == home_goals:
                    away_stats['points'] += 1
        
        return {
            'home': {
                'ppg': home_stats['points'] / home_stats['played'] if home_stats['played'] > 0 else 0,
                'goals_per_game': home_stats['goals_for'] / home_stats['played'] if home_stats['played'] > 0 else 0,
                'goals_against_per_game': home_stats['goals_against'] / home_stats['played'] if home_stats['played'] > 0 else 0,
                'win_rate': (home_stats['points'] / (home_stats['played'] * 3)) if home_stats['played'] > 0 else 0,
                'matches_played': home_stats['played']
            },
            'away': {
                'ppg': away_stats['points'] / away_stats['played'] if away_stats['played'] > 0 else 0,
                'goals_per_game': away_stats['goals_for'] / away_stats['played'] if away_stats['played'] > 0 else 0,
                'goals_against_per_game': away_stats['goals_against'] / away_stats['played'] if away_stats['played'] > 0 else 0,
                'win_rate': (away_stats['points'] / (away_stats['played'] * 3)) if away_stats['played'] > 0 else 0,
                'matches_played': away_stats['played']
            },
            'venue_advantage': (home_stats['points'] / home_stats['played'] if home_stats['played'] > 0 else 0) - 
                             (away_stats['points'] / away_stats['played'] if away_stats['played'] > 0 else 0)
        }
    
    def _calculate_opponent_adjusted_metrics(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate metrics adjusted for opponent strength"""
        vs_top_half = {'played': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0}
        vs_bottom_half = {'played': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0}
        
        for match in matches[:15]:  # Last 15 matches
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            league = match.get('league', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            # Determine opponent
            opponent = None
            is_home = False
            if home_team.get('id') == team_id:
                opponent = away_team
                is_home = True
            elif away_team.get('id') == team_id:
                opponent = home_team
                is_home = False
            
            if not opponent:
                continue
                
            # Estimate opponent strength (simplified - in real system would use league table)
            # For now, use a simple heuristic based on recent form
            opponent_strong = self._estimate_opponent_strength(opponent.get('id'), league.get('id'))
            
            stats = vs_top_half if opponent_strong else vs_bottom_half
            stats['played'] += 1
            
            if is_home:
                stats['goals_for'] += home_goals
                stats['goals_against'] += away_goals
                if home_goals > away_goals:
                    stats['points'] += 3
                elif home_goals == away_goals:
                    stats['points'] += 1
            else:
                stats['goals_for'] += away_goals
                stats['goals_against'] += home_goals
                if away_goals > home_goals:
                    stats['points'] += 3
                elif away_goals == home_goals:
                    stats['points'] += 1
        
        return {
            'vs_strong_teams': {
                'ppg': vs_top_half['points'] / vs_top_half['played'] if vs_top_half['played'] > 0 else 0,
                'goals_per_game': vs_top_half['goals_for'] / vs_top_half['played'] if vs_top_half['played'] > 0 else 0,
                'goals_against_per_game': vs_top_half['goals_against'] / vs_top_half['played'] if vs_top_half['played'] > 0 else 0,
                'matches': vs_top_half['played']
            },
            'vs_weak_teams': {
                'ppg': vs_bottom_half['points'] / vs_bottom_half['played'] if vs_bottom_half['played'] > 0 else 0,
                'goals_per_game': vs_bottom_half['goals_for'] / vs_bottom_half['played'] if vs_bottom_half['played'] > 0 else 0,
                'goals_against_per_game': vs_bottom_half['goals_against'] / vs_bottom_half['played'] if vs_bottom_half['played'] > 0 else 0,
                'matches': vs_bottom_half['played']
            },
            'strength_differential': (vs_top_half['points'] / vs_top_half['played'] if vs_top_half['played'] > 0 else 0) - 
                                   (vs_bottom_half['points'] / vs_bottom_half['played'] if vs_bottom_half['played'] > 0 else 0)
        }
    
    def _calculate_form_stability(self, matches: List[Dict], team_id: int) -> float:
        """Calculate how stable/consistent the team's form is"""
        if len(matches) < 5:
            return 0.5
            
        results = []
        for match in matches[:10]:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                if home_goals > away_goals:
                    results.append(3)
                elif home_goals == away_goals:
                    results.append(1)
                else:
                    results.append(0)
            elif away_team.get('id') == team_id:
                if away_goals > home_goals:
                    results.append(3)
                elif away_goals == home_goals:
                    results.append(1)
                else:
                    results.append(0)
        
        if len(results) < 3:
            return 0.5
            
        # Calculate standard deviation of results
        std_dev = np.std(results)
        # Normalize to 0-1 scale (inverse - lower std = higher stability)
        stability = 1 - (std_dev / 1.5)  # Max std dev is ~1.5
        return max(0, min(1, stability))
    
    def _analyze_scoring_trends(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze goal scoring trends"""
        goals_timeline = []
        
        for match in matches[:15]:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                goals_timeline.append(home_goals)
            elif away_team.get('id') == team_id:
                goals_timeline.append(away_goals)
        
        if len(goals_timeline) < 3:
            return {'trend': 'stable', 'recent_avg': 0, 'older_avg': 0, 'improvement': 0}
            
        recent_avg = np.mean(goals_timeline[:5]) if len(goals_timeline) >= 5 else np.mean(goals_timeline)
        older_avg = np.mean(goals_timeline[5:10]) if len(goals_timeline) >= 10 else np.mean(goals_timeline[3:])
        
        improvement = recent_avg - older_avg
        
        if improvement > 0.3:
            trend = 'improving'
        elif improvement < -0.3:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'improvement': improvement,
            'consistency': 1 - (np.std(goals_timeline[:5]) / (recent_avg + 0.1)) if recent_avg > 0 else 0.5
        }
    
    def _analyze_defensive_trends(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze defensive trends"""
        goals_conceded_timeline = []
        clean_sheets_recent = 0
        
        for i, match in enumerate(matches[:15]):
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if home_team.get('id') == team_id:
                goals_conceded_timeline.append(away_goals)
                if i < 5 and away_goals == 0:
                    clean_sheets_recent += 1
            elif away_team.get('id') == team_id:
                goals_conceded_timeline.append(home_goals)
                if i < 5 and home_goals == 0:
                    clean_sheets_recent += 1
        
        if len(goals_conceded_timeline) < 3:
            return {'trend': 'stable', 'recent_avg': 0, 'older_avg': 0, 'improvement': 0, 'clean_sheet_rate': 0}
            
        recent_avg = np.mean(goals_conceded_timeline[:5]) if len(goals_conceded_timeline) >= 5 else np.mean(goals_conceded_timeline)
        older_avg = np.mean(goals_conceded_timeline[5:10]) if len(goals_conceded_timeline) >= 10 else np.mean(goals_conceded_timeline[3:])
        
        # For defense, lower is better
        improvement = older_avg - recent_avg
        
        if improvement > 0.3:
            trend = 'improving'
        elif improvement < -0.3:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'improvement': improvement,
            'clean_sheet_rate': clean_sheets_recent / min(5, len(goals_conceded_timeline)) if goals_conceded_timeline else 0,
            'consistency': 1 - (np.std(goals_conceded_timeline[:5]) / (recent_avg + 0.1)) if recent_avg > 0 else 0.5
        }
    
    def _analyze_pressure_performance(self, matches: List[Dict], team_id: int) -> Dict:
        """Analyze performance under pressure (late goals, comebacks, etc.)"""
        late_goals_scored = 0
        late_goals_conceded = 0
        comebacks = 0
        points_dropped_late = 0
        clutch_wins = 0
        
        for match in matches[:20]:
            home_team = match.get('teams', {}).get('home', {})
            away_team = match.get('teams', {}).get('away', {})
            events = match.get('events', [])
            score = match.get('score', {}).get('fulltime', {})
            
            if not score:
                continue
                
            is_home = home_team.get('id') == team_id
            is_away = away_team.get('id') == team_id
            
            if not (is_home or is_away):
                continue
                
            # Analyze match events for late goals (after 75 minutes)
            for event in events:
                if event.get('type') == 'Goal' and event.get('time', {}).get('elapsed', 0) >= 75:
                    if (is_home and event.get('team', {}).get('id') == team_id) or \
                       (is_away and event.get('team', {}).get('id') == team_id):
                        late_goals_scored += 1
                    else:
                        late_goals_conceded += 1
            
            # Check for comebacks and clutch situations
            # This is simplified - in real system would analyze match flow
            home_goals = score.get('home', 0) or 0
            away_goals = score.get('away', 0) or 0
            
            if is_home and home_goals > away_goals and late_goals_scored > 0:
                clutch_wins += 1
            elif is_away and away_goals > home_goals and late_goals_scored > 0:
                clutch_wins += 1
        
        return {
            'late_goals_scored': late_goals_scored,
            'late_goals_conceded': late_goals_conceded,
            'clutch_factor': (late_goals_scored - late_goals_conceded) / 20,  # Normalized
            'pressure_rating': min(1.0, (clutch_wins + late_goals_scored) / 10),  # 0-1 scale
            'mental_strength': 0.5 + (0.5 * (late_goals_scored - late_goals_conceded) / max(1, late_goals_scored + late_goals_conceded))
        }
    
    def _calculate_overall_form_score(self, form_metrics: Dict) -> float:
        """Calculate overall form score from 0-100"""
        score = 0
        
        # Recent form (30%)
        recent_ppg = form_metrics['rolling_windows']['short']['points_per_game']
        score += (recent_ppg / 3.0) * 30
        
        # Weighted performance (20%)
        performance_index = form_metrics['weighted_performance']['performance_index']
        score += min(20, performance_index * 5)
        
        # Current streak (15%)
        streak = form_metrics['streak_analysis']['current_streak']
        if streak['type'] == 'W':
            score += min(15, streak['length'] * 5)
        elif streak['type'] == 'D':
            score += 7.5
        
        # Momentum (15%)
        momentum = form_metrics['momentum_shifts']['momentum_change']
        score += max(0, min(15, 7.5 + (momentum * 7.5)))
        
        # Form stability (10%)
        stability = form_metrics['form_stability']
        score += stability * 10
        
        # Scoring form (10%)
        scoring_improvement = form_metrics['scoring_trends']['improvement']
        score += max(0, min(10, 5 + (scoring_improvement * 5)))
        
        return min(100, max(0, score))
    
    def _determine_trajectory(self, form_metrics: Dict) -> str:
        """Determine overall trajectory based on multiple factors"""
        momentum = form_metrics['momentum_shifts']['trend']
        scoring = form_metrics['scoring_trends']['trend']
        defensive = form_metrics['defensive_trends']['trend']
        
        # Weight different aspects
        trajectory_score = 0
        if momentum == 'improving':
            trajectory_score += 2
        elif momentum == 'declining':
            trajectory_score -= 2
            
        if scoring == 'improving':
            trajectory_score += 1
        elif scoring == 'declining':
            trajectory_score -= 1
            
        if defensive == 'improving':
            trajectory_score += 1
        elif defensive == 'declining':
            trajectory_score -= 1
        
        if trajectory_score >= 2:
            return 'strongly_improving'
        elif trajectory_score >= 1:
            return 'improving'
        elif trajectory_score <= -2:
            return 'strongly_declining'
        elif trajectory_score <= -1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_streak_momentum(self, streak: Dict) -> float:
        """Calculate momentum from current streak"""
        if not streak['type']:
            return 0.5
            
        if streak['type'] == 'W':
            return min(1.0, 0.5 + (streak['length'] * 0.1))
        elif streak['type'] == 'L':
            return max(0.0, 0.5 - (streak['length'] * 0.1))
        else:  # Draw
            return 0.5
    
    def _estimate_opponent_strength(self, opponent_id: int, league_id: int) -> bool:
        """Estimate if opponent is strong (top half of table)"""
        # Simplified estimation - in real system would use actual league table
        # For now, use a simple heuristic based on team ID patterns
        # This should be replaced with actual league standings data
        
        # Teams with lower IDs tend to be stronger (established clubs)
        # This is a very rough approximation
        if league_id in [39, 140, 61, 78, 135]:  # Top leagues
            return opponent_id < 100
        else:
            return opponent_id < 500
    
    def _get_default_form(self) -> Dict:
        """Return default form metrics when data is unavailable"""
        return {
            'rolling_windows': {
                'short': {'points_per_game': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'win_rate': 0},
                'medium': {'points_per_game': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'win_rate': 0},
                'long': {'points_per_game': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'win_rate': 0}
            },
            'weighted_performance': {'weighted_points': 0, 'weighted_goals': 0, 'weighted_defense': 0, 'performance_index': 0},
            'streak_analysis': {'current_streak': {'type': None, 'length': 0}, 'longest_win_streak': 0, 'longest_unbeaten': 0, 'current_unbeaten': 0, 'streak_momentum': 0.5},
            'momentum_shifts': {'momentum_change': 0, 'trend': 'stable', 'turning_point': None, 'recent_form_avg': 0, 'older_form_avg': 0, 'volatility': 0},
            'venue_specific': {
                'home': {'ppg': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'win_rate': 0, 'matches_played': 0},
                'away': {'ppg': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'win_rate': 0, 'matches_played': 0},
                'venue_advantage': 0
            },
            'opponent_adjusted': {
                'vs_strong_teams': {'ppg': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'matches': 0},
                'vs_weak_teams': {'ppg': 0, 'goals_per_game': 0, 'goals_against_per_game': 0, 'matches': 0},
                'strength_differential': 0
            },
            'form_stability': 0.5,
            'scoring_trends': {'trend': 'stable', 'recent_avg': 0, 'older_avg': 0, 'improvement': 0, 'consistency': 0.5},
            'defensive_trends': {'trend': 'stable', 'recent_avg': 0, 'older_avg': 0, 'improvement': 0, 'clean_sheet_rate': 0, 'consistency': 0.5},
            'pressure_performance': {'late_goals_scored': 0, 'late_goals_conceded': 0, 'clutch_factor': 0, 'pressure_rating': 0, 'mental_strength': 0.5},
            'overall_form_score': 50,
            'trajectory': 'stable'
        }
    
    def compare_team_forms(self, home_form: Dict, away_form: Dict) -> Dict:
        """Compare two teams' forms and provide insights"""
        comparison = {
            'form_advantage': home_form['overall_form_score'] - away_form['overall_form_score'],
            'momentum_comparison': {
                'home': home_form['momentum_shifts']['trend'],
                'away': away_form['momentum_shifts']['trend'],
                'advantage': self._compare_trends(home_form['momentum_shifts']['trend'], away_form['momentum_shifts']['trend'])
            },
            'venue_impact': {
                'home_at_home': home_form['venue_specific']['home']['ppg'],
                'away_at_away': away_form['venue_specific']['away']['ppg'],
                'expected_advantage': home_form['venue_specific']['home']['ppg'] - away_form['venue_specific']['away']['ppg']
            },
            'scoring_matchup': {
                'home_attack': home_form['scoring_trends']['recent_avg'],
                'away_defense': away_form['defensive_trends']['recent_avg'],
                'home_expected': (home_form['scoring_trends']['recent_avg'] + away_form['defensive_trends']['recent_avg']) / 2
            },
            'defensive_matchup': {
                'away_attack': away_form['scoring_trends']['recent_avg'],
                'home_defense': home_form['defensive_trends']['recent_avg'],
                'away_expected': (away_form['scoring_trends']['recent_avg'] + home_form['defensive_trends']['recent_avg']) / 2
            },
            'pressure_factor': {
                'home': home_form['pressure_performance']['pressure_rating'],
                'away': away_form['pressure_performance']['pressure_rating'],
                'clutch_advantage': home_form['pressure_performance']['pressure_rating'] - away_form['pressure_performance']['pressure_rating']
            },
            'prediction_confidence': self._calculate_prediction_confidence(home_form, away_form)
        }
        
        return comparison
    
    def _compare_trends(self, trend1: str, trend2: str) -> str:
        """Compare two trends and determine advantage"""
        trend_values = {
            'strongly_improving': 2,
            'improving': 1,
            'stable': 0,
            'declining': -1,
            'strongly_declining': -2
        }
        
        value1 = trend_values.get(trend1, 0)
        value2 = trend_values.get(trend2, 0)
        
        diff = value1 - value2
        
        if diff >= 2:
            return 'strong_home'
        elif diff >= 1:
            return 'slight_home'
        elif diff <= -2:
            return 'strong_away'
        elif diff <= -1:
            return 'slight_away'
        else:
            return 'neutral'
    
    def _calculate_prediction_confidence(self, home_form: Dict, away_form: Dict) -> float:
        """Calculate confidence in prediction based on form analysis"""
        confidence = 0.5  # Base confidence
        
        # Form difference impact
        form_diff = abs(home_form['overall_form_score'] - away_form['overall_form_score'])
        confidence += min(0.2, form_diff / 100)
        
        # Stability impact (stable teams are more predictable)
        avg_stability = (home_form['form_stability'] + away_form['form_stability']) / 2
        confidence += avg_stability * 0.1
        
        # Trajectory alignment (clear trajectories increase confidence)
        if home_form['trajectory'] in ['strongly_improving', 'strongly_declining'] or \
           away_form['trajectory'] in ['strongly_improving', 'strongly_declining']:
            confidence += 0.1
        
        # Venue form clarity
        home_venue_strength = home_form['venue_specific']['home']['ppg']
        away_venue_strength = away_form['venue_specific']['away']['ppg']
        if abs(home_venue_strength - away_venue_strength) > 1:
            confidence += 0.1
        
        return min(1.0, float(confidence))