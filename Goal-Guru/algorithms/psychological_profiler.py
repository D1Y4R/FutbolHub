"""
Enhanced Psychological Profiler Module for Football Prediction System
Analyzes psychological factors, motivation levels, and critical match situations.

Implements comprehensive psychological analysis including:
- Critical Match Detection (Derby, Title race, Relegation battles, etc.)
- Pressure Situation Analysis (Must-win scenarios, Comeback requirements)
- Motivation Index Calculator (0-100 scoring system)
- Psychological Momentum Tracker (Confidence and mental fatigue)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class PsychologicalProfiler:
    """
    Advanced psychological profiler for football predictions
    Analyzes psychological factors that significantly impact match outcomes
    """
    
    def __init__(self):
        """Initialize the psychological profiler with comprehensive databases"""
        
        # Derby and rivalry detection database
        self.rivalry_database = {
            # Turkish Rivalries
            'turkey': {
                'galatasaray': ['fenerbahce', 'besiktas', 'trabzonspor'],
                'fenerbahce': ['galatasaray', 'besiktas', 'trabzonspor'],
                'besiktas': ['galatasaray', 'fenerbahce'],
                'trabzonspor': ['galatasaray', 'fenerbahce']
            },
            # English Rivalries
            'england': {
                'manchester united': ['manchester city', 'liverpool', 'arsenal', 'chelsea'],
                'manchester city': ['manchester united', 'liverpool'],
                'liverpool': ['manchester united', 'manchester city', 'everton', 'arsenal'],
                'arsenal': ['tottenham', 'chelsea', 'manchester united', 'liverpool'],
                'chelsea': ['arsenal', 'tottenham', 'manchester united'],
                'tottenham': ['arsenal', 'chelsea'],
                'everton': ['liverpool']
            },
            # Spanish Rivalries
            'spain': {
                'real madrid': ['barcelona', 'atletico madrid', 'athletic bilbao'],
                'barcelona': ['real madrid', 'espanyol', 'atletico madrid'],
                'atletico madrid': ['real madrid', 'barcelona'],
                'valencia': ['levante'],
                'sevilla': ['real betis']
            },
            # Italian Rivalries
            'italy': {
                'juventus': ['inter', 'milan', 'napoli', 'roma'],
                'inter': ['milan', 'juventus'],
                'milan': ['inter', 'juventus'],
                'roma': ['lazio', 'juventus'],
                'lazio': ['roma'],
                'napoli': ['juventus']
            },
            # German Rivalries
            'germany': {
                'bayern munich': ['borussia dortmund', 'schalke'],
                'borussia dortmund': ['bayern munich', 'schalke'],
                'schalke': ['borussia dortmund', 'bayern munich']
            },
            # French Rivalries
            'france': {
                'paris saint-germain': ['marseille', 'lyon'],
                'marseille': ['paris saint-germain'],
                'lyon': ['saint-etienne', 'paris saint-germain'],
                'saint-etienne': ['lyon']
            }
        }
        
        # City-based derby detection
        self.city_derbies = {
            'manchester': ['manchester united', 'manchester city'],
            'liverpool': ['liverpool', 'everton'],
            'london': ['arsenal', 'chelsea', 'tottenham', 'west ham', 'crystal palace', 'fulham', 'brentford'],
            'madrid': ['real madrid', 'atletico madrid', 'getafe', 'rayo vallecano'],
            'milan': ['milan', 'inter'],
            'rome': ['roma', 'lazio'],
            'istanbul': ['galatasaray', 'fenerbahce', 'besiktas'],
            'paris': ['paris saint-germain', 'paris fc']
        }
        
        # League importance weights for different match types
        self.league_importance_weights = {
            # Top European leagues - higher psychological impact
            'premier_league': 1.3,
            'la_liga': 1.25,
            'serie_a': 1.2,
            'bundesliga': 1.15,
            'ligue_1': 1.1,
            # European competitions - very high impact
            'champions_league': 1.5,
            'europa_league': 1.4,
            'conference_league': 1.3,
            # Domestic cups - medium-high impact
            'fa_cup': 1.2,
            'copa_del_rey': 1.2,
            'dfb_pokal': 1.15,
            # Other leagues - standard impact
            'default': 1.0
        }
        
        # Critical match type weights
        self.match_type_weights = {
            'derby': 1.4,
            'title_race': 1.5,
            'relegation_battle': 1.6,
            'european_qualification': 1.3,
            'cup_final': 1.7,
            'playoff': 1.6,
            'must_win': 1.5,
            'revenge_match': 1.2,
            'manager_debut': 1.1,
            'season_finale': 1.3
        }
        
        # Pressure thresholds and multipliers
        self.pressure_thresholds = {
            'low_pressure': 30,      # Below 30 = low pressure
            'medium_pressure': 60,   # 30-60 = medium pressure
            'high_pressure': 80,     # 60-80 = high pressure
            'extreme_pressure': 100  # Above 80 = extreme pressure
        }
        
        # Motivation factors database
        self.motivation_factors = {
            'streak_breaking': {
                'losing_streak_3+': 15,    # +15 motivation to break losing streak
                'losing_streak_5+': 25,    # +25 for longer streaks
                'winless_streak_5+': 12,   # +12 for winless streaks
                'winning_streak_5+': -8    # -8 motivation when on good run
            },
            'revenge_factor': {
                'recent_defeat': 12,       # +12 for revenge after recent loss
                'heavy_defeat': 18,        # +18 for revenge after heavy loss (3+ goals)
                'multiple_defeats': 20     # +20 for multiple recent defeats
            },
            'manager_effect': {
                'new_manager_honeymoon': 15,    # +15 for first 5 matches
                'new_manager_adjustment': 8,     # +8 for matches 6-15
                'manager_pressure': -10          # -10 when manager under pressure
            },
            'player_impact': {
                'star_player_return': 12,        # +12 for key player return
                'star_player_injury': -15,       # -15 for key player injury
                'new_signing_debut': 8           # +8 for new signing motivation
            }
        }
        
        logger.info("PsychologicalProfiler initialized with comprehensive databases")
    
    def analyze_psychological_profile(self, home_team_data: Dict, away_team_data: Dict, 
                                    match_context: Dict) -> Dict[str, Any]:
        """
        Complete psychological analysis for both teams
        
        Args:
            home_team_data: Home team comprehensive data
            away_team_data: Away team comprehensive data  
            match_context: Match context including league, H2H data, etc.
            
        Returns:
            Dict containing complete psychological analysis
        """
        try:
            # Ensure match_context is a dictionary
            if not isinstance(match_context, dict):
                match_context = {}
            # 1. Critical Match Detection
            critical_match_analysis = self._detect_critical_match_types(
                home_team_data, away_team_data, match_context
            )
            
            # 2. Pressure Situation Analysis
            pressure_analysis = self._analyze_pressure_situations(
                home_team_data, away_team_data, match_context, critical_match_analysis
            )
            
            # 3. Motivation Index Calculation
            motivation_analysis = self._calculate_motivation_indices(
                home_team_data, away_team_data, match_context, critical_match_analysis
            )
            
            # 4. Psychological Momentum Tracking
            momentum_analysis = self._track_psychological_momentum(
                home_team_data, away_team_data, match_context
            )
            
            # 5. Overall Psychological Assessment
            overall_assessment = self._generate_overall_assessment(
                critical_match_analysis, pressure_analysis, 
                motivation_analysis, momentum_analysis
            )
            
            return {
                'critical_match_analysis': critical_match_analysis,
                'pressure_analysis': pressure_analysis,
                'motivation_analysis': motivation_analysis,
                'momentum_analysis': momentum_analysis,
                'overall_assessment': overall_assessment,
                'psychological_advantage': self._determine_psychological_advantage(
                    motivation_analysis, momentum_analysis, pressure_analysis
                ),
                'match_importance_score': self._calculate_match_importance_score(
                    critical_match_analysis, pressure_analysis
                ),
                'confidence_levels': {
                    'home': momentum_analysis['home_team']['confidence_level'],
                    'away': momentum_analysis['away_team']['confidence_level']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in psychological analysis: {str(e)}")
            return self._get_default_psychological_profile()
    
    def _detect_critical_match_types(self, home_data: Dict, away_data: Dict, 
                                   match_context: Dict) -> Dict[str, Any]:
        """
        Detect different types of critical matches
        """
        critical_types = []
        importance_multiplier = 1.0
        derby_details = {}
        
        home_name = home_data.get('name', '').lower()
        away_name = away_data.get('name', '').lower()
        
        # Ensure league_info is a dict
        league_info = match_context.get('league', {})
        if not isinstance(league_info, dict):
            league_info = {}
        
        # Ensure league_table is a list
        league_table = match_context.get('league_table', [])
        if not isinstance(league_table, list):
            league_table = []
        
        # 1. Derby/Rivalry Detection
        derby_analysis = self._detect_derby_rivalry(home_name, away_name, league_info)
        if derby_analysis['is_derby']:
            critical_types.append('derby')
            importance_multiplier *= self.match_type_weights['derby']
            derby_details = derby_analysis
        
        # 2. Title Race Detection
        title_race_analysis = self._detect_title_race(home_data, away_data, league_table)
        if title_race_analysis['is_title_race']:
            critical_types.append('title_race')
            importance_multiplier *= self.match_type_weights['title_race']
        
        # 3. Relegation Battle Detection
        relegation_analysis = self._detect_relegation_battle(home_data, away_data, league_table)
        if relegation_analysis['is_relegation_battle']:
            critical_types.append('relegation_battle')
            importance_multiplier *= self.match_type_weights['relegation_battle']
        
        # 4. European Qualification Detection
        european_analysis = self._detect_european_qualification(home_data, away_data, league_table)
        if european_analysis['is_european_race']:
            critical_types.append('european_qualification')
            importance_multiplier *= self.match_type_weights['european_qualification']
        
        # 5. Cup Finals and Important Matches
        cup_analysis = self._detect_cup_importance(match_context)
        if cup_analysis['is_important_cup_match']:
            critical_types.extend(cup_analysis['cup_types'])
            importance_multiplier *= cup_analysis['importance_multiplier']
        
        # 6. Must-Win Scenarios
        must_win_analysis = self._detect_must_win_scenarios(home_data, away_data, league_table)
        if must_win_analysis['has_must_win']:
            critical_types.append('must_win')
            importance_multiplier *= self.match_type_weights['must_win']
        
        return {
            'critical_types': critical_types,
            'is_critical_match': len(critical_types) > 0,
            'importance_multiplier': min(2.5, importance_multiplier),  # Cap at 2.5x
            'derby_analysis': derby_details,
            'title_race_analysis': title_race_analysis,
            'relegation_analysis': relegation_analysis,
            'european_analysis': european_analysis,
            'cup_analysis': cup_analysis,
            'must_win_analysis': must_win_analysis,
            'critical_match_intensity': self._calculate_critical_intensity(critical_types)
        }
    
    def _detect_derby_rivalry(self, home_name: str, away_name: str, 
                            league_info: Dict) -> Dict[str, Any]:
        """
        Detect derby and rivalry matches
        """
        # Clean team names for matching
        home_clean = self._clean_team_name(home_name)
        away_clean = self._clean_team_name(away_name)
        
        derby_type = None
        rivalry_intensity = 0
        
        # Check historical rivalries
        for country, rivalries in self.rivalry_database.items():
            for team, rivals in rivalries.items():
                if self._fuzzy_team_match(home_clean, team):
                    if any(self._fuzzy_team_match(away_clean, rival) for rival in rivals):
                        derby_type = 'historical_rivalry'
                        rivalry_intensity = 0.9
                        break
                elif self._fuzzy_team_match(away_clean, team):
                    if any(self._fuzzy_team_match(home_clean, rival) for rival in rivals):
                        derby_type = 'historical_rivalry'
                        rivalry_intensity = 0.9
                        break
        
        # Check city derbies
        if not derby_type:
            for city, teams in self.city_derbies.items():
                home_in_city = any(self._fuzzy_team_match(home_clean, team) for team in teams)
                away_in_city = any(self._fuzzy_team_match(away_clean, team) for team in teams)
                
                if home_in_city and away_in_city:
                    derby_type = 'city_derby'
                    rivalry_intensity = 0.8
                    break
        
        # Check regional rivalries (same region/area)
        if not derby_type and self._is_regional_rivalry(home_clean, away_clean):
            derby_type = 'regional_rivalry'
            rivalry_intensity = 0.6
        
        return {
            'is_derby': derby_type is not None,
            'derby_type': derby_type,
            'rivalry_intensity': rivalry_intensity,
            'rivalry_description': self._get_rivalry_description(derby_type or 'none', home_clean, away_clean)
        }
    
    def _detect_title_race(self, home_data: Dict, away_data: Dict, 
                         league_table: List[Dict]) -> Dict[str, Any]:
        """
        Detect title race implications
        """
        if not league_table or len(league_table) < 4:
            return {'is_title_race': False, 'title_implications': None}
        
        # Get team positions
        home_position = home_data.get('league_position', 99)
        away_position = away_data.get('league_position', 99)
        
        # Title race if both teams in top 4 or one team challenging leader
        is_title_race = False
        title_implications = []
        
        if home_position <= 4 and away_position <= 4:
            is_title_race = True
            title_implications.append('top_4_clash')
        
        if home_position == 1 or away_position == 1:
            is_title_race = True
            title_implications.append('leader_involved')
        
        if home_position <= 2 and away_position <= 2:
            is_title_race = True
            title_implications.append('title_contenders_clash')
        
        # Check points gap for title race relevance
        if len(league_table) >= max(home_position, away_position):
            try:
                home_team_data = next((team for team in league_table if team.get('position') == home_position), None)
                away_team_data = next((team for team in league_table if team.get('position') == away_position), None)
                leader_data = league_table[0] if league_table else None
                
                if home_team_data and away_team_data and leader_data:
                    home_points = home_team_data.get('points', 0)
                    away_points = away_team_data.get('points', 0)
                    leader_points = leader_data.get('points', 0)
                    
                    # Within 12 points of leader = title race relevant
                    if (leader_points - home_points <= 12) or (leader_points - away_points <= 12):
                        is_title_race = True
                        title_implications.append('within_title_range')
                        
            except Exception as e:
                logger.warning(f"Error calculating title race points gap: {e}")
        
        return {
            'is_title_race': is_title_race,
            'title_implications': title_implications,
            'title_race_intensity': len(title_implications) * 0.3
        }
    
    def _detect_relegation_battle(self, home_data: Dict, away_data: Dict, 
                                league_table: List[Dict]) -> Dict[str, Any]:
        """
        Detect relegation battle scenarios
        """
        if not league_table:
            return {'is_relegation_battle': False, 'relegation_implications': None}
        
        total_teams = len(league_table)
        if total_teams < 16:  # Not enough teams for meaningful relegation zone
            return {'is_relegation_battle': False, 'relegation_implications': None}
        
        # Typical relegation zone is bottom 3 teams
        relegation_zone = max(3, total_teams // 6)  # Bottom 3 or 1/6 of teams
        relegation_threshold = total_teams - relegation_zone
        danger_zone = relegation_threshold - 3  # 3 positions above relegation zone
        
        home_position = home_data.get('league_position', 99)
        away_position = away_data.get('league_position', 99)
        
        is_relegation_battle = False
        relegation_implications = []
        
        # Direct relegation battle
        if home_position >= relegation_threshold or away_position >= relegation_threshold:
            is_relegation_battle = True
            relegation_implications.append('direct_relegation_battle')
        
        # Danger zone battle
        if home_position >= danger_zone or away_position >= danger_zone:
            is_relegation_battle = True
            relegation_implications.append('relegation_threatened')
        
        # Six-pointer (both teams in danger)
        if (home_position >= danger_zone and away_position >= danger_zone):
            is_relegation_battle = True
            relegation_implications.append('six_pointer')
        
        return {
            'is_relegation_battle': is_relegation_battle,
            'relegation_implications': relegation_implications,
            'relegation_pressure_level': len(relegation_implications) * 0.4,
            'relegation_zone_threshold': relegation_threshold
        }
    
    def _detect_european_qualification(self, home_data: Dict, away_data: Dict, 
                                     league_table: List[Dict]) -> Dict[str, Any]:
        """
        Detect European qualification race
        """
        if not league_table:
            return {'is_european_race': False, 'european_implications': None}
        
        total_teams = len(league_table)
        if total_teams < 12:
            return {'is_european_race': False, 'european_implications': None}
        
        # European spots typically: Top 4 = Champions League, 5-6 = Europa League, 7 = Conference League
        champions_league_spots = 4
        europa_league_spots = 2
        conference_league_spots = 1
        
        total_european_spots = champions_league_spots + europa_league_spots + conference_league_spots
        european_race_zone = total_european_spots + 3  # Include teams within 3 positions
        
        home_position = home_data.get('league_position', 99)
        away_position = away_data.get('league_position', 99)
        
        is_european_race = False
        european_implications = []
        
        # Champions League race
        if home_position <= champions_league_spots + 2 or away_position <= champions_league_spots + 2:
            is_european_race = True
            european_implications.append('champions_league_race')
        
        # Europa League race
        if ((champions_league_spots < home_position <= total_european_spots + 2) or 
            (champions_league_spots < away_position <= total_european_spots + 2)):
            is_european_race = True
            european_implications.append('europa_league_race')
        
        # General European qualification
        if home_position <= european_race_zone or away_position <= european_race_zone:
            is_european_race = True
            european_implications.append('european_qualification_race')
        
        return {
            'is_european_race': is_european_race,
            'european_implications': european_implications,
            'european_race_intensity': len(european_implications) * 0.25
        }
    
    def _detect_cup_importance(self, match_context: Dict) -> Dict[str, Any]:
        """
        Detect cup final and important cup match scenarios
        """
        competition_name = match_context.get('competition', '').lower()
        match_round = match_context.get('round', '').lower()
        
        is_important = False
        cup_types = []
        importance_multiplier = 1.0
        
        # Finals detection
        if any(keyword in match_round for keyword in ['final', 'finale', 'championship']):
            is_important = True
            cup_types.append('cup_final')
            importance_multiplier *= self.match_type_weights['cup_final']
        
        # Semi-finals
        elif any(keyword in match_round for keyword in ['semi', 'semifinal', 'semi-final']):
            is_important = True
            cup_types.append('semi_final')
            importance_multiplier *= 1.4
        
        # Quarter-finals of major competitions
        elif any(keyword in match_round for keyword in ['quarter', 'quarterfinal']):
            if any(comp in competition_name for comp in ['champions', 'europa', 'cup', 'copa']):
                is_important = True
                cup_types.append('quarter_final')
                importance_multiplier *= 1.2
        
        # Playoffs
        elif any(keyword in match_round for keyword in ['playoff', 'play-off', 'promotion']):
            is_important = True
            cup_types.append('playoff')
            importance_multiplier *= self.match_type_weights['playoff']
        
        return {
            'is_important_cup_match': is_important,
            'cup_types': cup_types,
            'importance_multiplier': importance_multiplier,
            'cup_round': match_round
        }
    
    def _detect_must_win_scenarios(self, home_data: Dict, away_data: Dict, 
                                 league_table: List[Dict]) -> Dict[str, Any]:
        """
        Detect must-win scenarios for teams
        """
        home_position = home_data.get('league_position', 99)
        away_position = away_data.get('league_position', 99)
        
        # Get recent form to detect crisis situations
        home_recent_points = home_data.get('recent_form', {}).get('points_per_game', 1.0) * 3
        away_recent_points = away_data.get('recent_form', {}).get('points_per_game', 1.0) * 3
        
        must_win_scenarios = []
        
        # Relegation must-win
        if league_table and len(league_table) > 15:
            relegation_zone = len(league_table) - 3
            if home_position >= relegation_zone - 1:
                must_win_scenarios.append('home_relegation_must_win')
            if away_position >= relegation_zone - 1:
                must_win_scenarios.append('away_relegation_must_win')
        
        # Poor form must-win (less than 1 point per game in recent matches)
        if home_recent_points < 3:
            must_win_scenarios.append('home_form_crisis')
        if away_recent_points < 3:
            must_win_scenarios.append('away_form_crisis')
        
        # Title race must-win (if close to leader)
        if league_table and len(league_table) > 5:
            if home_position <= 3:
                must_win_scenarios.append('home_title_pressure')
            if away_position <= 3:
                must_win_scenarios.append('away_title_pressure')
        
        return {
            'has_must_win': len(must_win_scenarios) > 0,
            'must_win_scenarios': must_win_scenarios,
            'must_win_intensity': len(must_win_scenarios) * 0.3
        }
    
    def _analyze_pressure_situations(self, home_data: Dict, away_data: Dict, 
                                   match_context: Dict, critical_analysis: Dict) -> Dict[str, Any]:
        """
        Analyze pressure situations for both teams
        """
        # Recent critical match count
        home_critical_matches = self._count_recent_critical_matches(home_data, match_context)
        away_critical_matches = self._count_recent_critical_matches(away_data, match_context)
        
        # Home crowd pressure assessment
        home_crowd_pressure = self._assess_home_crowd_pressure(home_data, critical_analysis)
        
        # Comeback requirement situations
        home_comeback_pressure = self._assess_comeback_requirements(home_data, match_context)
        away_comeback_pressure = self._assess_comeback_requirements(away_data, match_context)
        
        # Overall pressure calculation
        home_pressure = self._calculate_overall_pressure(
            home_data, critical_analysis, home_critical_matches, 
            home_crowd_pressure, home_comeback_pressure, is_home=True
        )
        
        away_pressure = self._calculate_overall_pressure(
            away_data, critical_analysis, away_critical_matches, 
            0, away_comeback_pressure, is_home=False
        )
        
        return {
            'home_team': {
                'pressure_level': home_pressure,
                'critical_matches_last_5': home_critical_matches,
                'crowd_pressure': home_crowd_pressure,
                'comeback_pressure': home_comeback_pressure,
                'pressure_category': self._categorize_pressure(home_pressure)
            },
            'away_team': {
                'pressure_level': away_pressure,
                'critical_matches_last_5': away_critical_matches,
                'crowd_pressure': 0,  # Away team doesn't benefit from home crowd
                'comeback_pressure': away_comeback_pressure,
                'pressure_category': self._categorize_pressure(away_pressure)
            },
            'pressure_differential': home_pressure - away_pressure,
            'high_pressure_match': max(home_pressure, away_pressure) > self.pressure_thresholds['high_pressure']
        }
    
    def _calculate_motivation_indices(self, home_data: Dict, away_data: Dict, 
                                    match_context: Dict, critical_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive motivation indices (0-100) for both teams
        """
        home_motivation = self._calculate_team_motivation(home_data, match_context, critical_analysis, is_home=True)
        away_motivation = self._calculate_team_motivation(away_data, match_context, critical_analysis, is_home=False)
        
        motivation_diff = home_motivation['total_motivation'] - away_motivation['total_motivation']
        
        # Determine motivation advantage
        if motivation_diff > 15:
            motivation_advantage = 'strong_home_advantage'
        elif motivation_diff > 5:
            motivation_advantage = 'slight_home_advantage'
        elif motivation_diff < -15:
            motivation_advantage = 'strong_away_advantage'
        elif motivation_diff < -5:
            motivation_advantage = 'slight_away_advantage'
        else:
            motivation_advantage = 'balanced_motivation'
        
        return {
            'home_team': home_motivation,
            'away_team': away_motivation,
            'motivation_differential': motivation_diff,
            'motivation_advantage': motivation_advantage
        }
    
    def _calculate_team_motivation(self, team_data: Dict, match_context: Dict, 
                                 critical_analysis: Dict, is_home: bool) -> Dict[str, Any]:
        """
        Calculate detailed motivation score for a team (0-100)
        """
        base_motivation = 50  # Neutral starting point
        motivation_factors = {}
        
        # 1. Streak breaking motivation
        streak_factor = self._calculate_streak_motivation(team_data)
        base_motivation += streak_factor
        motivation_factors['streak_breaking'] = streak_factor
        
        # 2. Revenge factor
        revenge_factor = self._calculate_revenge_motivation(team_data, match_context)
        base_motivation += revenge_factor
        motivation_factors['revenge_factor'] = revenge_factor
        
        # 3. Manager effect
        manager_factor = self._calculate_manager_motivation(team_data)
        base_motivation += manager_factor
        motivation_factors['manager_effect'] = manager_factor
        
        # 4. Star player impact
        player_factor = self._calculate_player_motivation(team_data)
        base_motivation += player_factor
        motivation_factors['player_impact'] = player_factor
        
        # 5. Critical match motivation boost
        critical_factor = self._calculate_critical_match_motivation(critical_analysis, is_home)
        base_motivation += critical_factor
        motivation_factors['critical_match_boost'] = critical_factor
        
        # 6. Home advantage motivation
        if is_home:
            home_factor = self._calculate_home_motivation(team_data, critical_analysis)
            base_motivation += home_factor
            motivation_factors['home_advantage'] = home_factor
        
        # 7. League position motivation
        position_factor = self._calculate_position_motivation(team_data)
        base_motivation += position_factor
        motivation_factors['position_pressure'] = position_factor
        
        # Cap motivation between 0-100
        total_motivation = max(0, min(100, base_motivation))
        
        return {
            'total_motivation': total_motivation,
            'motivation_factors': motivation_factors,
            'motivation_level': self._categorize_motivation(total_motivation)
        }
    
    def _track_psychological_momentum(self, home_data: Dict, away_data: Dict, 
                                    match_context: Dict) -> Dict[str, Any]:
        """
        Track psychological momentum for both teams
        """
        home_momentum = self._calculate_team_momentum(home_data, match_context, is_home=True)
        away_momentum = self._calculate_team_momentum(away_data, match_context, is_home=False)
        
        return {
            'home_team': home_momentum,
            'away_team': away_momentum,
            'momentum_shift': self._detect_momentum_shifts(home_data, away_data, match_context),
            'momentum_advantage': self._determine_momentum_advantage(home_momentum, away_momentum)
        }
    
    def _calculate_team_momentum(self, team_data: Dict, match_context: Dict, 
                               is_home: bool) -> Dict[str, Any]:
        """
        Calculate psychological momentum for a team
        """
        # Confidence level calculation
        confidence_level = self._calculate_confidence_level(team_data)
        
        # Mental fatigue assessment
        mental_fatigue = self._assess_mental_fatigue(team_data, match_context)
        
        # Success/failure cycle analysis
        cycle_analysis = self._analyze_success_failure_cycle(team_data)
        
        # Recent performance trend
        performance_trend = self._calculate_performance_trend(team_data)
        
        return {
            'confidence_level': confidence_level,
            'mental_fatigue': mental_fatigue,
            'cycle_analysis': cycle_analysis,
            'performance_trend': performance_trend,
            'momentum_score': self._calculate_momentum_score(
                confidence_level, mental_fatigue, cycle_analysis, performance_trend
            )
        }
    
    # Additional helper methods will be implemented in the continuation...
    def _clean_team_name(self, team_name: str) -> str:
        """Clean team name for matching"""
        # Remove common prefixes and suffixes
        name = team_name.lower().strip()
        name = re.sub(r'\b(fc|cf|ac|sc|united|city|town|rovers|wanderers|athletic|sporting)\b', '', name)
        name = re.sub(r'[^\w\s]', '', name).strip()
        return name
    
    def _fuzzy_team_match(self, name1: str, name2: str) -> bool:
        """Fuzzy matching for team names"""
        name1_clean = self._clean_team_name(name1)
        name2_clean = self._clean_team_name(name2)
        
        # Direct match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Word overlap
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        if words1 & words2:  # Any common words
            return True
        
        return False
    
    def _is_regional_rivalry(self, home_name: str, away_name: str) -> bool:
        """Check if teams are from same region (basic implementation)"""
        # This is a simplified implementation
        # In reality, you'd want a more comprehensive regional database
        return False
    
    def _get_rivalry_description(self, derby_type: str, home_name: str, away_name: str) -> str:
        """Generate rivalry description"""
        descriptions = {
            'historical_rivalry': f"Tarihsel rakiplik: {home_name.title()} vs {away_name.title()}",
            'city_derby': f"Şehir derbisi: {home_name.title()} vs {away_name.title()}",
            'regional_rivalry': f"Bölgesel rekabet: {home_name.title()} vs {away_name.title()}"
        }
        return descriptions.get(derby_type, "Standart maç")
    
    def _calculate_critical_intensity(self, critical_types: List[str]) -> float:
        """Calculate overall critical match intensity"""
        if not critical_types:
            return 0.0
        
        # Weight different critical types
        type_weights = {
            'derby': 0.4,
            'title_race': 0.5,
            'relegation_battle': 0.6,
            'european_qualification': 0.3,
            'cup_final': 0.7,
            'must_win': 0.5
        }
        
        total_intensity = sum(type_weights.get(ct, 0.2) for ct in critical_types)
        return min(1.0, total_intensity)  # Cap at 1.0
    
    def _count_recent_critical_matches(self, team_data: Dict, match_context: Dict) -> int:
        """Count critical matches in last 5 games"""
        # This would analyze recent match history for critical match types
        # Simplified implementation
        recent_matches = team_data.get('recent_matches', [])
        critical_count = 0
        
        for match in recent_matches[:5]:  # Last 5 matches
            # Check if match was against strong opposition or in critical situation
            if self._was_critical_match(match):
                critical_count += 1
        
        return critical_count
    
    def _was_critical_match(self, match: Dict) -> bool:
        """Determine if a past match was critical"""
        # Simplified implementation - in reality would check opposition strength,
        # league position at time of match, etc.
        return False  # Placeholder
    
    def _assess_home_crowd_pressure(self, home_data: Dict, critical_analysis: Dict) -> float:
        """Assess home crowd pressure effect"""
        base_pressure = 10  # Base home crowd effect
        
        # Increase pressure for critical matches
        if critical_analysis['is_critical_match']:
            base_pressure += critical_analysis['critical_match_intensity'] * 20
        
        # Increase pressure based on team's recent form
        recent_form = home_data.get('recent_form', {}).get('points_per_game', 1.0)
        if recent_form < 1.0:  # Poor form increases crowd pressure
            base_pressure += (1.0 - recent_form) * 15
        
        return min(30, base_pressure)  # Cap at 30
    
    def _assess_comeback_requirements(self, team_data: Dict, match_context: Dict) -> float:
        """Assess if team needs comeback in season/competition"""
        comeback_pressure = 0
        
        # League position comeback requirement
        position = team_data.get('league_position', 10)
        if position > 15:  # Lower table teams need comeback
            comeback_pressure += (position - 15) * 2
        
        # Recent form comeback requirement
        recent_form = team_data.get('recent_form', {}).get('points_per_game', 1.0)
        if recent_form < 1.0:
            comeback_pressure += (1.0 - recent_form) * 20
        
        return min(25, comeback_pressure)  # Cap at 25
    
    def _calculate_overall_pressure(self, team_data: Dict, critical_analysis: Dict,
                                  critical_matches: int, crowd_pressure: float,
                                  comeback_pressure: float, is_home: bool) -> float:
        """Calculate overall pressure level for team"""
        base_pressure = 30  # Neutral pressure
        
        # Critical match pressure
        if critical_analysis['is_critical_match']:
            base_pressure += critical_analysis['critical_match_intensity'] * 30
        
        # Recent critical matches pressure
        base_pressure += critical_matches * 5
        
        # Crowd pressure (only for home team)
        if is_home:
            base_pressure += crowd_pressure
        
        # Comeback pressure
        base_pressure += comeback_pressure
        
        # League position pressure
        position = team_data.get('league_position', 10)
        if position <= 4:  # Top teams have title pressure
            base_pressure += 10
        elif position >= 17:  # Bottom teams have relegation pressure
            base_pressure += 15
        
        return min(100, base_pressure)  # Cap at 100
    
    def _categorize_pressure(self, pressure_level: float) -> str:
        """Categorize pressure level"""
        if pressure_level < self.pressure_thresholds['low_pressure']:
            return 'low_pressure'
        elif pressure_level < self.pressure_thresholds['medium_pressure']:
            return 'medium_pressure'
        elif pressure_level < self.pressure_thresholds['high_pressure']:
            return 'high_pressure'
        else:
            return 'extreme_pressure'
    
    def _calculate_streak_motivation(self, team_data: Dict) -> float:
        """Calculate motivation from streak breaking potential"""
        recent_form = team_data.get('recent_form', {})
        streak_data = team_data.get('streak_analysis', {})
        
        motivation_change = 0
        
        # Losing streak motivation
        losing_streak = streak_data.get('current_losing_streak', 0)
        if losing_streak >= 5:
            motivation_change += self.motivation_factors['streak_breaking']['losing_streak_5+']
        elif losing_streak >= 3:
            motivation_change += self.motivation_factors['streak_breaking']['losing_streak_3+']
        
        # Winless streak motivation
        winless_streak = streak_data.get('current_winless_streak', 0)
        if winless_streak >= 5:
            motivation_change += self.motivation_factors['streak_breaking']['winless_streak_5+']
        
        # Winning streak complacency
        winning_streak = streak_data.get('current_winning_streak', 0)
        if winning_streak >= 5:
            motivation_change += self.motivation_factors['streak_breaking']['winning_streak_5+']
        
        return motivation_change
    
    def _calculate_revenge_motivation(self, team_data: Dict, match_context: Dict) -> float:
        """Calculate revenge motivation from recent H2H results"""
        h2h_data = match_context.get('h2h_data', {})
        recent_matches = h2h_data.get('recent_matches', [])
        
        motivation_change = 0
        
        if recent_matches:
            # Check last few matches against this opponent
            recent_defeats = 0
            heavy_defeats = 0
            
            for match in recent_matches[:3]:  # Last 3 H2H matches
                if self._team_lost_match(team_data, match):
                    recent_defeats += 1
                    if self._was_heavy_defeat(team_data, match):
                        heavy_defeats += 1
            
            # Apply revenge motivation
            if heavy_defeats > 0:
                motivation_change += self.motivation_factors['revenge_factor']['heavy_defeat']
            elif recent_defeats >= 2:
                motivation_change += self.motivation_factors['revenge_factor']['multiple_defeats']
            elif recent_defeats >= 1:
                motivation_change += self.motivation_factors['revenge_factor']['recent_defeat']
        
        return motivation_change
    
    def _calculate_manager_motivation(self, team_data: Dict) -> float:
        """Calculate manager effect on motivation"""
        manager_data = team_data.get('manager_info', {})
        matches_since_appointment = manager_data.get('matches_since_appointment', 100)
        
        motivation_change = 0
        
        if matches_since_appointment <= 5:
            motivation_change += self.motivation_factors['manager_effect']['new_manager_honeymoon']
        elif matches_since_appointment <= 15:
            motivation_change += self.motivation_factors['manager_effect']['new_manager_adjustment']
        
        # Manager under pressure
        if manager_data.get('under_pressure', False):
            motivation_change += self.motivation_factors['manager_effect']['manager_pressure']
        
        return motivation_change
    
    def _calculate_player_motivation(self, team_data: Dict) -> float:
        """Calculate player-related motivation factors"""
        player_news = team_data.get('player_news', {})
        
        motivation_change = 0
        
        # Star player return from injury
        if player_news.get('star_player_returning', False):
            motivation_change += self.motivation_factors['player_impact']['star_player_return']
        
        # Star player injury
        if player_news.get('star_player_injured', False):
            motivation_change += self.motivation_factors['player_impact']['star_player_injury']
        
        # New signing debut
        if player_news.get('new_signing_debut', False):
            motivation_change += self.motivation_factors['player_impact']['new_signing_debut']
        
        return motivation_change
    
    def _calculate_critical_match_motivation(self, critical_analysis: Dict, is_home: bool) -> float:
        """Calculate motivation boost from critical match"""
        if not critical_analysis['is_critical_match']:
            return 0
        
        base_boost = critical_analysis['critical_match_intensity'] * 15
        
        # Extra boost for home team in critical matches
        if is_home:
            base_boost *= 1.2
        
        return base_boost
    
    def _calculate_home_motivation(self, home_data: Dict, critical_analysis: Dict) -> float:
        """Calculate home advantage motivation"""
        base_home_motivation = 5  # Base home advantage
        
        # Increase for critical matches
        if critical_analysis['is_critical_match']:
            base_home_motivation += critical_analysis['critical_match_intensity'] * 8
        
        # Home form factor
        home_form = home_data.get('venue_form', {}).get('points_per_game', 1.0)
        if home_form > 2.0:  # Good home form
            base_home_motivation += 3
        elif home_form < 1.0:  # Poor home form - pressure to improve
            base_home_motivation += 2
        
        return base_home_motivation
    
    def _calculate_position_motivation(self, team_data: Dict) -> float:
        """Calculate motivation based on league position"""
        position = team_data.get('league_position', 10)
        
        motivation_change = 0
        
        # Top teams - title pressure motivation
        if position <= 3:
            motivation_change += 5
        # Mid-table - European competition motivation
        elif 4 <= position <= 8:
            motivation_change += 3
        # Bottom teams - relegation avoidance motivation
        elif position >= 17:
            motivation_change += 8
        
        return motivation_change
    
    def _categorize_motivation(self, motivation_level: float) -> str:
        """Categorize motivation level"""
        if motivation_level >= 75:
            return 'extremely_motivated'
        elif motivation_level >= 60:
            return 'highly_motivated'
        elif motivation_level >= 40:
            return 'neutral_motivated'
        elif motivation_level >= 25:
            return 'low_motivated'
        else:
            return 'demotivated'
    
    def _calculate_confidence_level(self, team_data: Dict) -> float:
        """Calculate team confidence level"""
        recent_form = team_data.get('recent_form', {})
        points_per_game = recent_form.get('points_per_game', 1.0) * 3  # Convert to points out of 9
        
        # Base confidence from recent form
        confidence = (points_per_game / 9) * 100  # 0-100 scale
        
        # Adjust for consistency
        consistency = recent_form.get('consistency', 0.5)
        confidence *= (0.8 + consistency * 0.4)  # Consistent teams get boost
        
        # Adjust for recent big wins/losses
        if team_data.get('recent_big_win', False):
            confidence += 10
        if team_data.get('recent_big_loss', False):
            confidence -= 15
        
        return max(0, min(100, confidence))
    
    def _assess_mental_fatigue(self, team_data: Dict, match_context: Dict) -> float:
        """Assess mental fatigue level"""
        # Base fatigue from fixture congestion
        fixture_data = team_data.get('fixture_congestion', {})
        fatigue = fixture_data.get('fatigue_score', 30)  # 0-100 scale
        
        # Increase fatigue for teams under pressure
        position = team_data.get('league_position', 10)
        if position >= 17:  # Relegation zone
            fatigue += 15
        elif position <= 4:  # Title race
            fatigue += 10
        
        # Recent critical matches increase fatigue
        recent_critical = self._count_recent_critical_matches(team_data, match_context)
        fatigue += recent_critical * 5
        
        return min(100, fatigue)
    
    def _analyze_success_failure_cycle(self, team_data: Dict) -> Dict[str, Any]:
        """Analyze success/failure psychological cycle"""
        recent_results = team_data.get('recent_results', [])
        
        if len(recent_results) < 5:
            return {'cycle_type': 'insufficient_data', 'cycle_strength': 0}
        
        # Analyze pattern
        wins = sum(1 for r in recent_results[:5] if r == 'W')
        losses = sum(1 for r in recent_results[:5] if r == 'L')
        draws = 5 - wins - losses
        
        if wins >= 4:
            return {'cycle_type': 'success_cycle', 'cycle_strength': 0.8}
        elif losses >= 4:
            return {'cycle_type': 'failure_cycle', 'cycle_strength': 0.8}
        elif draws >= 3:
            return {'cycle_type': 'stagnation_cycle', 'cycle_strength': 0.6}
        else:
            return {'cycle_type': 'mixed_results', 'cycle_strength': 0.3}
    
    def _calculate_performance_trend(self, team_data: Dict) -> Dict[str, Any]:
        """Calculate recent performance trend"""
        recent_form = team_data.get('recent_form', {})
        last_5_ppg = recent_form.get('points_per_game', 1.0) * 3
        last_10_ppg = team_data.get('medium_form', {}).get('points_per_game', 1.0) * 3
        
        trend_direction = 'stable'
        trend_strength = 0
        
        if last_5_ppg > last_10_ppg + 1:
            trend_direction = 'improving'
            trend_strength = min(1.0, (last_5_ppg - last_10_ppg) / 3)
        elif last_5_ppg < last_10_ppg - 1:
            trend_direction = 'declining'
            trend_strength = min(1.0, (last_10_ppg - last_5_ppg) / 3)
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'recent_ppg': last_5_ppg,
            'medium_term_ppg': last_10_ppg
        }
    
    def _calculate_momentum_score(self, confidence: float, fatigue: float, 
                                cycle: Dict, trend: Dict) -> float:
        """Calculate overall momentum score"""
        base_momentum = confidence * 0.4  # 40% from confidence
        
        # Subtract fatigue effect
        base_momentum -= (fatigue * 0.2)  # 20% penalty from fatigue
        
        # Cycle effect
        cycle_multiplier = {
            'success_cycle': 1.2,
            'failure_cycle': 0.7,
            'stagnation_cycle': 0.9,
            'mixed_results': 1.0
        }
        base_momentum *= cycle_multiplier.get(cycle['cycle_type'], 1.0)
        
        # Trend effect
        if trend['trend_direction'] == 'improving':
            base_momentum += trend['trend_strength'] * 15
        elif trend['trend_direction'] == 'declining':
            base_momentum -= trend['trend_strength'] * 15
        
        return max(0, min(100, base_momentum))
    
    def _detect_momentum_shifts(self, home_data: Dict, away_data: Dict, 
                              match_context: Dict) -> Dict[str, Any]:
        """Detect recent momentum shifts"""
        shifts = []
        
        # Check for recent manager changes
        if home_data.get('manager_info', {}).get('matches_since_appointment', 100) <= 3:
            shifts.append('home_manager_change')
        if away_data.get('manager_info', {}).get('matches_since_appointment', 100) <= 3:
            shifts.append('away_manager_change')
        
        # Check for dramatic form changes
        home_trend = self._calculate_performance_trend(home_data)
        away_trend = self._calculate_performance_trend(away_data)
        
        if home_trend['trend_strength'] > 0.7:
            if home_trend['trend_direction'] == 'improving':
                shifts.append('home_upward_momentum')
            else:
                shifts.append('home_downward_momentum')
        
        if away_trend['trend_strength'] > 0.7:
            if away_trend['trend_direction'] == 'improving':
                shifts.append('away_upward_momentum')
            else:
                shifts.append('away_downward_momentum')
        
        return {
            'shifts_detected': shifts,
            'has_momentum_shift': len(shifts) > 0,
            'shift_count': len(shifts)
        }
    
    def _determine_momentum_advantage(self, home_momentum: Dict, away_momentum: Dict) -> str:
        """Determine which team has momentum advantage"""
        home_score = home_momentum['momentum_score']
        away_score = away_momentum['momentum_score']
        
        diff = home_score - away_score
        
        if diff > 15:
            return 'strong_home_advantage'
        elif diff > 5:
            return 'slight_home_advantage'
        elif diff < -15:
            return 'strong_away_advantage'
        elif diff < -5:
            return 'slight_away_advantage'
        else:
            return 'balanced_momentum'
    
    def _team_lost_match(self, team_data: Dict, match: Dict) -> bool:
        """Check if team lost a specific match"""
        # This is a simplified implementation
        # In reality, you'd check the match result against team's perspective
        return False  # Placeholder
    
    def _was_heavy_defeat(self, team_data: Dict, match: Dict) -> bool:
        """Check if team suffered heavy defeat (3+ goal margin)"""
        # This is a simplified implementation
        return False  # Placeholder
    
    def _generate_overall_assessment(self, critical_analysis: Dict, pressure_analysis: Dict,
                                   motivation_analysis: Dict, momentum_analysis: Dict) -> Dict[str, Any]:
        """Generate overall psychological assessment"""
        
        # Calculate psychological edge
        home_psych_score = (
            motivation_analysis['home_team']['total_motivation'] * 0.4 +
            momentum_analysis['home_team']['momentum_score'] * 0.3 +
            (100 - pressure_analysis['home_team']['pressure_level']) * 0.3
        )
        
        away_psych_score = (
            motivation_analysis['away_team']['total_motivation'] * 0.4 +
            momentum_analysis['away_team']['momentum_score'] * 0.3 +
            (100 - pressure_analysis['away_team']['pressure_level']) * 0.3
        )
        
        psych_advantage = home_psych_score - away_psych_score
        
        return {
            'home_psychological_score': home_psych_score,
            'away_psychological_score': away_psych_score,
            'psychological_advantage': psych_advantage,
            'dominant_factors': self._identify_dominant_factors(
                critical_analysis, pressure_analysis, motivation_analysis, momentum_analysis
            ),
            'psychological_prediction_impact': self._assess_prediction_impact(
                critical_analysis, pressure_analysis, motivation_analysis, momentum_analysis
            )
        }
    
    def _determine_psychological_advantage(self, motivation_analysis: Dict, 
                                         momentum_analysis: Dict, pressure_analysis: Dict) -> str:
        """Determine overall psychological advantage"""
        
        motivation_diff = motivation_analysis['motivation_differential']
        momentum_home = momentum_analysis['home_team']['momentum_score']
        momentum_away = momentum_analysis['away_team']['momentum_score']
        pressure_diff = pressure_analysis['pressure_differential']
        
        # Combined psychological advantage score
        total_advantage = motivation_diff + (momentum_home - momentum_away) - pressure_diff
        
        if total_advantage > 20:
            return 'strong_home_psychological_advantage'
        elif total_advantage > 8:
            return 'moderate_home_psychological_advantage'
        elif total_advantage < -20:
            return 'strong_away_psychological_advantage'
        elif total_advantage < -8:
            return 'moderate_away_psychological_advantage'
        else:
            return 'balanced_psychological_state'
    
    def _calculate_match_importance_score(self, critical_analysis: Dict, 
                                        pressure_analysis: Dict) -> float:
        """Calculate match importance score (0-10)"""
        
        base_importance = 5.0  # Neutral match importance
        
        # Critical match factors
        if critical_analysis['is_critical_match']:
            base_importance += critical_analysis['critical_match_intensity'] * 3
        
        # Pressure factors
        max_pressure = max(
            pressure_analysis['home_team']['pressure_level'],
            pressure_analysis['away_team']['pressure_level']
        )
        base_importance += (max_pressure / 100) * 2
        
        # Multiple critical types increase importance
        critical_types_count = len(critical_analysis['critical_types'])
        if critical_types_count > 1:
            base_importance += (critical_types_count - 1) * 0.5
        
        return min(10.0, max(0.0, base_importance))
    
    def _identify_dominant_factors(self, critical_analysis: Dict, pressure_analysis: Dict,
                                 motivation_analysis: Dict, momentum_analysis: Dict) -> List[str]:
        """Identify the most dominant psychological factors"""
        dominant_factors = []
        
        # Critical match types
        if critical_analysis['is_critical_match']:
            dominant_factors.extend(critical_analysis['critical_types'])
        
        # High pressure situations
        if pressure_analysis['high_pressure_match']:
            dominant_factors.append('high_pressure_environment')
        
        # Extreme motivation differences
        if abs(motivation_analysis['motivation_differential']) > 20:
            if motivation_analysis['motivation_differential'] > 0:
                dominant_factors.append('home_motivation_advantage')
            else:
                dominant_factors.append('away_motivation_advantage')
        
        # Strong momentum shifts
        if momentum_analysis['momentum_shift']['has_momentum_shift']:
            dominant_factors.append('momentum_shift_detected')
        
        return dominant_factors
    
    def _assess_prediction_impact(self, critical_analysis: Dict, pressure_analysis: Dict,
                                motivation_analysis: Dict, momentum_analysis: Dict) -> Dict[str, float]:
        """Assess how psychological factors should impact predictions"""
        
        impact_factors = {
            'outcome_probability_adjustment': 0.0,  # Adjust 1X2 probabilities
            'goal_expectation_adjustment': 0.0,     # Adjust expected goals
            'variance_adjustment': 0.0,             # Adjust prediction variance
            'confidence_adjustment': 0.0            # Adjust prediction confidence
        }
        
        # Critical match impact
        if critical_analysis['is_critical_match']:
            intensity = critical_analysis['critical_match_intensity']
            impact_factors['outcome_probability_adjustment'] += intensity * 0.15
            impact_factors['variance_adjustment'] += intensity * 0.1
        
        # Motivation impact
        motivation_diff = abs(motivation_analysis['motivation_differential'])
        if motivation_diff > 15:
            impact_factors['outcome_probability_adjustment'] += (motivation_diff / 100) * 0.2
            impact_factors['confidence_adjustment'] += (motivation_diff / 100) * 0.1
        
        # Momentum impact
        momentum_advantage = momentum_analysis['momentum_advantage']
        if 'strong' in momentum_advantage:
            impact_factors['outcome_probability_adjustment'] += 0.12
            impact_factors['goal_expectation_adjustment'] += 0.08
        
        # Pressure impact (increases variance, decreases predictability)
        if pressure_analysis['high_pressure_match']:
            impact_factors['variance_adjustment'] += 0.15
            impact_factors['confidence_adjustment'] -= 0.1
        
        return impact_factors
    
    def _get_default_psychological_profile(self) -> Dict[str, Any]:
        """Get default psychological profile when analysis fails"""
        return {
            'critical_match_analysis': {
                'critical_types': [],
                'is_critical_match': False,
                'importance_multiplier': 1.0,
                'critical_match_intensity': 0.0
            },
            'pressure_analysis': {
                'home_team': {
                    'pressure_level': 30,
                    'pressure_category': 'low_pressure'
                },
                'away_team': {
                    'pressure_level': 30,
                    'pressure_category': 'low_pressure'
                },
                'high_pressure_match': False
            },
            'motivation_analysis': {
                'home_team': {
                    'total_motivation': 50,
                    'motivation_level': 'neutral_motivated'
                },
                'away_team': {
                    'total_motivation': 50,
                    'motivation_level': 'neutral_motivated'
                },
                'motivation_differential': 0
            },
            'momentum_analysis': {
                'home_team': {
                    'momentum_score': 50,
                    'confidence_level': 50
                },
                'away_team': {
                    'momentum_score': 50,
                    'confidence_level': 50
                },
                'momentum_advantage': 'balanced_momentum'
            },
            'overall_assessment': {
                'psychological_advantage': 0,
                'dominant_factors': [],
                'psychological_prediction_impact': {
                    'outcome_probability_adjustment': 0.0,
                    'goal_expectation_adjustment': 0.0,
                    'variance_adjustment': 0.0,
                    'confidence_adjustment': 0.0
                }
            },
            'psychological_advantage': 'balanced_psychological_state',
            'match_importance_score': 5.0,
            'confidence_levels': {'home': 50, 'away': 50}
        }