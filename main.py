import asyncio
import json
import time
import os
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from pydantic_ai import Agent
import openai
import re

# Load racing constants from config or use defaults
def load_racing_constants() -> Dict:
    """Load racing constants from config file"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        return config.get('racing_constants', {})
    except FileNotFoundError:
        return {}

# Get racing constants with fallbacks
racing_config = load_racing_constants()
METERS_PER_HORSE_LENGTH = racing_config.get('meters_per_horse_length', 2.4)
LEADER_GROUP_THRESHOLD = racing_config.get('leader_group_threshold', 1.0)
TAILED_OFF_THRESHOLD = racing_config.get('tailed_off_threshold', 3.0)
MAX_LEADER_GROUP = racing_config.get('max_leader_group', 3)
LEADER_PERCENTAGE = racing_config.get('leader_percentage', 0.25)

def setup_model_client(config: Dict):
    """Setup model client based on config provider"""
    model_provider = config.get('model_provider', 'anthropic').lower()
    
    if model_provider in ['anthropic', 'claude']:
        # Use Claude/Anthropic
        api_key = config.get('anthropic_api_key')
        if not api_key:
            raise ValueError("Missing 'anthropic_api_key' for Claude/Anthropic")
        
        os.environ['ANTHROPIC_API_KEY'] = api_key
        model_name = config.get('anthropic_model', 'claude-3-5-sonnet-20241022')
        print(f"🤖 SELECTED MODEL: ANTHROPIC CLAUDE - {model_name}")
        
        return 'anthropic', Agent, model_name
        
    elif model_provider == 'deepseek':
        # Use DeepSeek
        api_key = config.get('deepseek_api_key')
        if not api_key:
            raise ValueError("Missing 'deepseek_api_key' for DeepSeek")
        
        model_name = config.get('deepseek_model', 'deepseek-chat')
        print(f"🤖 SELECTED MODEL: DEEPSEEK - {model_name}")
        
        # Setup DeepSeek client
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        return 'deepseek', client, model_name
        
    else:
        # Fallback to Claude
        print(f"⚠️ UNKNOWN PROVIDER: {model_provider}, falling back to Claude")
        api_key = config.get('anthropic_api_key')
        if not api_key:
            raise ValueError("Missing 'anthropic_api_key' for fallback")
        
        os.environ['ANTHROPIC_API_KEY'] = api_key
        model_name = config.get('anthropic_model', 'claude-3-5-sonnet-20241022')
        print(f"🤖 FALLBACK MODEL: ANTHROPIC CLAUDE - {model_name}")
        
        return 'anthropic', Agent, model_name

class PerformanceMetrics(BaseModel):
    finish_time: Optional[float] = None
    position: Optional[str] = None
    run_out_speed: Optional[float] = None  # ROS
    velocity_peak: Optional[float] = None  # VP (Top Speed)
    avg_stride_length: Optional[float] = None  # ASL
    avg_stride_frequency: Optional[float] = None  # ASF
    time_to_reach_30: Optional[float] = None  # TTR
    distance_beaten: Optional[float] = None  # DB
    best_position: Optional[int] = None  # BP
    avg_speed: Optional[float] = None  # AS (calculated or from TPD)

class GPSPosition(BaseModel):
    furlong: str  # "6f", "5f", "4f", etc.
    position: int
    distance_back: Optional[float] = None  # Distance behind leader in horse lengths
    running_time: float
    racing_term: Optional[str] = None  # Store the intelligent cluster term
    
class RaceData(BaseModel):
    final_position: int
    performance_metrics: Optional[PerformanceMetrics] = None
    gps_positions: List[GPSPosition] = []
    gate_performance: Optional[str] = None
    break_point: Optional[str] = None
    speed_metrics: Optional[Dict] = None
    stride_data: Optional[Dict] = None
    final_phase: Optional[str] = None
    position_changes: List[Dict] = []
    distance_back_data: Dict[str, int] = {}
    assigned_performance_metric: Optional[Dict] = None  # NEW: Stores assigned metric info
    final_furlong_surge: bool = False

class CommentaryResult(BaseModel):
    commentary: str
    accurate: bool
    issues: List[str]
    suggestions: List[str]
    uniqueness_score: float = 0.0
    data_completeness: float = 0.0

class TPDClient:
    def __init__(self, api_key: str, live_recording_api_key: str):
        self.api_key = api_key
        self.live_recording_api_key = live_recording_api_key
        self.base_url = "https://www.tpd.zone/json-rpc/v3/performance/"
        self.live_recording_url = "https://tpdapi.tpd.zone/live-recording/raw/"
    
    def get_performance_metrics(self, race_sharecode: str) -> Dict:
        """Fetch performance metrics from TPD API"""
        headers = {"k": self.api_key}
        params = {"sc": race_sharecode}
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ TPD Performance API Error: {e}")
            return {"success": False, "runners": {}}
    
    def get_live_recording_url(self, race_sharecode: str) -> Optional[str]:
        """Get S3 presigned URL for raw GPS data"""
        params = {
            "sc": race_sharecode,
            "k": self.live_recording_api_key
        }
        
        try:
            response = requests.get(self.live_recording_url, params=params, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                try:
                    api_data = response.json()
                    s3_url = (api_data.get('url') or 
                             api_data.get('download_url') or 
                             api_data.get('file_url') or
                             api_data.get('link') or
                             api_data.get('s3_url') or
                             api_data.get('bucket_url'))
                    
                    if not s3_url and isinstance(api_data, dict):
                        for key, value in api_data.items():
                            if isinstance(value, str) and ('s3.amazonaws.com' in value or 'amazonaws.com' in value):
                                s3_url = value
                                break
                    return s3_url
                except json.JSONDecodeError:
                    return None
            else:
                s3_url = response.text.strip()
                if s3_url.startswith('http') and 'amazonaws.com' in s3_url:
                    return s3_url
                    
            return None
        except requests.RequestException as e:
            print(f"❌ TPD Live Recording API Error: {e}")
            return None
    
    def fetch_gps_data(self, s3_url: str) -> List[Dict]:
        """Fetch and parse raw GPS data from S3"""
        try:
            print(f"Downloading from: {s3_url}")
            response = requests.get(s3_url, timeout=60, stream=True)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 100 * 1024 * 1024:
                raise ValueError(f"File too large: {content_length} bytes")
            
            file_content = response.text
            
            try:
                lines = [line.strip() for line in file_content.strip().split('\n') if line.strip()]
                
                if len(lines) == 1:
                    final_json = json.loads(lines[0])
                    print("Single JSON object found")
                    return [final_json] if isinstance(final_json, dict) else final_json
                else:
                    print(f"Found {len(lines)} JSON objects, processing all...")
                    
                    all_objects = []
                    for i, line in enumerate(lines):
                        try:
                            obj = json.loads(line)
                            all_objects.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON on line {i+1}: {e}")
                            continue
                    
                    if not all_objects:
                        raise ValueError("No valid JSON objects found")
                        
                    print(f"Loaded {len(all_objects)} JSON objects")
                    return all_objects
                    
            except Exception as e:
                raise ValueError(f"JSON processing failed: {e}")
                
        except requests.RequestException as e:
            print(f"❌ S3 GPS Data Fetch Error: {e}")
            return []
    
    def extract_gps_position_data(self, gps_records: List[Dict]) -> Dict[str, Dict]:
        """Extract K=5 records containing position and distance data"""
        position_data = {}
        
        for record in gps_records:
            if record.get('K') == 5:
                gate = record.get('G', '')
                if gate:
                    position_data[gate] = {
                        'order': record.get('O', []),
                        'distance_back': record.get('B', []),
                        'running_time': record.get('R', 0)
                    }
        
        return position_data

class EnhancedRacingAgent:
    def __init__(self, config: Dict, tpd_api_key: str, live_recording_api_key: str):
        # Setup model client
        self.provider, self.client_class, self.model_name = setup_model_client(config)
        self.tpd_client = TPDClient(tpd_api_key, live_recording_api_key)
        self.previous_commentaries = []
        
        # Create agents/clients based on provider
        if self.provider == 'anthropic':
            self.generator = self.client_class(
                self.model_name,
                output_type=str,
                system_prompt=self.get_generator_prompt()
            )
            
            self.critic = self.client_class(
                self.model_name,
                output_type=CommentaryResult,
                system_prompt=self.get_critic_prompt()
            )
            
        elif self.provider == 'deepseek':
            self.generator = self.client_class
            self.critic = self.client_class

    def get_generator_prompt(self) -> str:
        """FIXED: Bulletproof prompt that forces structured format"""
        return """You are a horse racing commentator. You MUST respond in this EXACT format:

<commentary>your racing commentary here</commentary>
<notes>your explanation here</notes>

SYSTEM REQUIREMENT: Any response not using these tags will cause system failure.

COMMENTARY RULES:
- 1-3 lines, max 100 words, basic UK English
- MUST include finishing position using "finished [position]"
- NO explanations, notes, or meta-commentary in the <commentary> section
- ONLY the race description goes in <commentary>

HIERARCHICAL PERFORMANCE PLACEMENT:
- START: TTR mentions after gate break ("quickest to reach pace in 2.1s, led from the start")
- MIDDLE: Top Speed, Stride mentions during race progression ("made ground with fastest speed of 42mph")  
- END: Run Out Speed mentions before finishing position ("strongest finish at 38mph, finished 2nd")

POSITION LANGUAGE (USE THESE EXACT TERMS):
- "clear leader", "disputed lead", "tightly bunched leaders"
- "up with the leaders", "tracked the leaders"
- "sat mid-field", "held up", "in the main group"
- "towards the back", "at the rear"
- "detached", "tailed off"

PERFORMANCE MENTIONS:
- TTR (START): "quickest to reach pace in X seconds" OR "2nd quickest to reach pace"
- Top Speed (MIDDLE): "hit top speed of X mph" OR "recorded 2nd fastest speed of X mph"
- Stride Length (MIDDLE): "longest stride length of X feet" OR "2nd longest stride length"
- Stride Frequency (MIDDLE): "recorded stride frequency of X per second" 
- Run Out Speed (END): "strongest finish at X mph" OR "2nd strongest finish at X mph"
- Average Speed (END): "maintained average speed of X mph"
- Final Furlong Surge: If gained 2+ positions - "flew home in final furlong"

MANDATORY STRUCTURE:
1. Gate break + START performance (TTR)
2. Race progression + MIDDLE performance (Top Speed/Stride)
3. Key position changes with distances when provided
4. END performance (ROS) + Final furlong surge if applicable
5. FINISHING POSITION (MANDATORY)

CRITICAL: Put racing commentary ONLY in <commentary> tags. Put explanations ONLY in <notes> tags.

EXAMPLE:
<commentary>Broke well to dispute the early lead, recorded fastest speed of 42mph when making ground at the 3f marker, strongest finish at 38mph and finished 2nd.</commentary>
<notes>Used hierarchical placement with speed mention in middle section and finish position at end as required.</notes>

YOU MUST USE THE STRUCTURED FORMAT. NO EXCEPTIONS."""

    def get_critic_prompt(self) -> str:
        """FIXED: JSON-only critic prompt with no ambiguity"""
        return """You are a racing commentary validator. You MUST respond in JSON format ONLY.

CRITICAL VALIDATION CHECKS:
1. MUST contain finishing position (e.g., "finished 3rd")
2. Position must match exactly with provided final_position
3. NO specific mid-race positions (reject "4th", "5th", etc.)
4. Use racing vernacular ("midfield", "leaders", "clear leader", "disputed lead")
5. Distance back claims must match data (integer lengths)
6. Performance metric mentions must be accurate
7. Natural commentary flow
8. HIERARCHICAL PERFORMANCE PLACEMENT: TTR at start, Speed/Stride in middle, ROS at end

ACCEPTABLE POSITION TERMS:
- "clear leader", "disputed lead", "tightly bunched leaders"
- "up with the leaders", "tracked the leaders"
- "midfield", "main group", "held up"
- "towards the rear", "at the back"
- "detached", "tailed off"

RESPOND IN THIS EXACT JSON FORMAT:
{
  "commentary": "corrected commentary text if needed",
  "accurate": true/false,
  "issues": ["list of specific issues"],
  "suggestions": ["list of improvements"],
  "uniqueness_score": 0.8,
  "data_completeness": 0.9
}

DO NOT include explanations outside the JSON. ONLY respond with valid JSON."""

    def parse_deepseek_critique(self, critique_text: str) -> CommentaryResult:
        """Parse DeepSeek critique text to extract structured feedback"""
        try:
            json_match = re.search(r'\{.*\}', critique_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                critique_data = json.loads(json_str)
                
                return CommentaryResult(
                    commentary=critique_data.get('commentary', ''),
                    accurate=critique_data.get('accurate', False),
                    issues=critique_data.get('issues', []),
                    suggestions=critique_data.get('suggestions', []),
                    uniqueness_score=critique_data.get('uniqueness_score', 0.0),
                    data_completeness=critique_data.get('data_completeness', 0.0)
                )
            else:
                return CommentaryResult(
                    commentary=critique_text,
                    accurate=False,
                    issues=["Unable to parse critique response"],
                    suggestions=["Please regenerate commentary"],
                    uniqueness_score=0.0,
                    data_completeness=0.0
                )
        except Exception as e:
            print(f"⚠️ DeepSeek critique parsing error: {e}")
            return CommentaryResult(
                commentary=critique_text,
                accurate=False,
                issues=[f"Parsing error: {str(e)}"],
                suggestions=["Please regenerate commentary"],
                uniqueness_score=0.0,
                data_completeness=0.0
            )

    def analyze_field_clustering(self, gps_position_data: Dict[str, Dict], furlong: str, total_runners: Optional[int] = None) -> Dict[str, str]:
        """Analyze field clustering using configurable constants"""
        data = gps_position_data.get(furlong, {})
        order = data.get('order', [])
        distances = data.get('distance_back', [])
        
        if not order:
            return {}
        
        distance_lengths = [d / METERS_PER_HORSE_LENGTH if d is not None else 0 for d in distances]
        clustering = {horse: "in the midfield" for horse in order}
        
        # Rule 1: Check for clear leader using configurable threshold
        if len(distance_lengths) > 1 and distance_lengths[1] > LEADER_GROUP_THRESHOLD:
            clustering[order[0]] = "clear leader"
        else:
            # Rule 2: Check for tight leaders using configurable max group
            leader_count = 1
            for i in range(1, min(MAX_LEADER_GROUP * 2, len(order))):
                if distance_lengths[i] <= LEADER_GROUP_THRESHOLD:
                    leader_count = i + 1
                else:
                    break
            
            if leader_count == 1:
                clustering[order[0]] = "led"
            elif leader_count == 2:
                clustering[order[0]] = "disputed the lead"
                clustering[order[1]] = "disputed the lead"
            elif leader_count >= 3:
                for i in range(leader_count):
                    clustering[order[i]] = "among the tightly bunched leaders"
        
        # Rule 3: Check for detached horses using configurable threshold
        rear_horses = []
        for i in range(1, len(order)):
            gap_from_previous = distance_lengths[i] - distance_lengths[i-1]
            if gap_from_previous > TAILED_OFF_THRESHOLD:
                rear_horses = order[i:]
                break
        
        for horse in rear_horses:
            if order.index(horse) == len(order) - 1:
                gap = distance_lengths[-1] - distance_lengths[-2] if len(distance_lengths) > 1 else 0
                if gap > TAILED_OFF_THRESHOLD:
                    clustering[horse] = "tailed off"
                else:
                    clustering[horse] = "at the rear"
            else:
                clustering[horse] = "towards the rear"
        
        # Rule 4: Fallback using configurable percentage
        if all(term == "in the midfield" for term in clustering.values()) and total_runners:
            leader_cutoff = min(MAX_LEADER_GROUP, max(1, int(total_runners * LEADER_PERCENTAGE)))
            rear_cutoff = max(total_runners - max(1, int(total_runners * LEADER_PERCENTAGE)), leader_cutoff + 1)
            
            for i, horse in enumerate(order):
                if i < leader_cutoff:
                    clustering[horse] = "up with the leaders"
                elif i >= rear_cutoff:
                    clustering[horse] = "towards the rear"
        
        # DEBUG: Print FINAL clustering results
        print(f"🔍 FINAL {furlong} clustering: {dict(list(clustering.items())[:5])}")
        
        return clustering

    def sort_furlongs_by_distance(self, furlongs: List[str]) -> List[str]:
        """       
        The final furlong marker (0.5f, 1f, etc.) contains FINISH data, not mid-race data
        """
        def furlong_to_distance(furlong: str) -> float:
            if furlong.lower() == 'finish':
                return 0.0
            elif furlong.endswith('f'):
                try:
                    return float(furlong[:-1])
                except ValueError:
                    return -1.0
            return -1.0
        
        # Filter out 'Finish' first
        racing_only = [f for f in furlongs if f.lower() != 'finish' and f.endswith('f')]
        
        if racing_only:
            # Find the smallest distance (final furlong marker = finish data)
            distances = []
            for f in racing_only:
                try:
                    distances.append(float(f[:-1]))
                except ValueError:
                    continue
            
            if distances:
                min_distance = min(distances)
                final_furlong = f"{min_distance}f"
                
                # CORRECT: Exclude final furlong marker (it's finish data, not mid-race data)
                racing_furlongs = [f for f in racing_only if f != final_furlong]
            else:
                racing_furlongs = racing_only
        else:
            racing_furlongs = []
        
        # Sort remaining furlongs
        sorted_furlongs = sorted(racing_furlongs, key=furlong_to_distance, reverse=True)
        
        print(f"🔧 FIXED: Mid-race furlongs: {sorted_furlongs}")
        return sorted_furlongs

    def build_position_timeline_from_gps(self, gps_position_data: Dict[str, Dict], horse_id: str, total_runners: int) -> List[GPSPosition]:
        """
        FIXED: Build position timeline using DYNAMIC furlong processing
        - Uses actual available furlongs instead of hardcoded list
        - Skips horses not present in specific furlong data
        - No more default "in the midfield" for missing horses
        """
        positions = []
        horse_num = str(int(horse_id[-2:]))
        
        # DYNAMIC: Use actual available furlongs, sorted by distance
        available_furlongs = list(gps_position_data.keys())
        sorted_furlongs = self.sort_furlongs_by_distance(available_furlongs)
        
        print(f"🔍 Horse {horse_num}: Processing {len(sorted_furlongs)} furlongs: {sorted_furlongs}")
        
        for furlong in sorted_furlongs:
            data = gps_position_data[furlong]
            order = data.get('order', [])
            distances = data.get('distance_back', [])
            running_time = data.get('running_time', 0)
            
            # CRITICAL: Only process horse if present in this furlong's data
            if horse_num in order:
                idx = order.index(horse_num)
                position = idx + 1
                
                distance_back = None
                if idx < len(distances):
                    distance_back = distances[idx] / METERS_PER_HORSE_LENGTH
                
                # Get intelligent clustering term for this furlong
                field_clustering = self.analyze_field_clustering(gps_position_data, furlong, total_runners)
                racing_term = field_clustering.get(horse_num)  # No default fallback!
                
                # Only create GPSPosition if we have a valid racing term
                if racing_term:
                    positions.append(GPSPosition(
                        furlong=furlong,
                        position=position,
                        distance_back=distance_back,
                        running_time=running_time,
                        racing_term=racing_term
                    ))
                    print(f"🔍 Horse {horse_num} at {furlong}: {racing_term} (pos {position})")
                else:
                    print(f"⚠️ Horse {horse_num} at {furlong}: No racing term available")
            else:
                print(f"⚠️ Horse {horse_num}: Missing from {furlong} order array")
        
        print(f"🔍 Horse {horse_num}: Created {len(positions)} GPS positions")
        return positions

    def get_distance_back_for_horse(self, gps_position_data: Dict[str, Dict], horse_id: str) -> Dict[str, int]:
        """Extract distance back at key furlongs with intelligent rounding"""
        distance_data = {}
        horse_num = str(int(horse_id[-2:]))
        
        raw_distances = {}
        for gate, data in gps_position_data.items():
            order = data.get('order', [])
            distances = data.get('distance_back', [])
            
            if horse_num in order:
                idx = order.index(horse_num)
                if idx < len(distances):
                    distance_in_lengths = distances[idx] / METERS_PER_HORSE_LENGTH
                    raw_distances[gate] = distance_in_lengths
        
        # DYNAMIC: Use actual available furlongs instead of hardcoded list
        available_furlongs = list(gps_position_data.keys())
        sorted_furlongs = self.sort_furlongs_by_distance(available_furlongs)
        previous_distance = None
        
        for gate in sorted_furlongs:
            if gate in raw_distances:
                current_distance = raw_distances[gate]
                
                # Apply rounding rules
                if 1.0 <= current_distance <= 1.3:
                    rounded_distance = 1
                elif 1.7 <= current_distance <= 1.9:
                    rounded_distance = 2
                elif 0.7 <= current_distance <= 0.9:
                    rounded_distance = 1
                elif current_distance >= 2.0:
                    rounded_distance = round(current_distance)
                elif current_distance <= 0.3:
                    rounded_distance = None
                else:
                    rounded_distance = None
                
                if rounded_distance is not None:
                    should_report = False
                    
                    if rounded_distance >= 3:
                        should_report = True
                    elif previous_distance is not None:
                        change = abs(current_distance - previous_distance)
                        if change >= 1.5:
                            should_report = True
                    elif gate in ['2f', '1f'] and rounded_distance <= 3:
                        should_report = True
                    elif gate in ['6f', '5f', '4f'] and rounded_distance >= 4:
                        should_report = True
                    
                    if should_report:
                        distance_data[gate] = rounded_distance
                    
                    previous_distance = current_distance
        
        return distance_data

    def find_top_performance_horses(self, tpd_data: Dict) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        FIXED: Find top DECILE performers for each metric and assign one metric per horse
        
        Returns:
            - performance_rankings: Dict with top performers per metric
            - horse_assignments: Dict mapping horse_id to assigned performance metric info
        """
        tpd_runners = tpd_data.get('runners', {})
        total_horses = len(tpd_runners)
        
        # TRUE DECILE SYSTEM: Top 10% of field
        top_n = max(1, int(total_horses * 0.1))  # True decile
        
        print(f"🔍 TRUE DECILE SYSTEM: Field size {total_horses} horses, top {top_n} per metric (10%)")
        
        # Available metrics with TPD field mapping
        metrics_config = {
            'top_speed': ('VP', 'max'),           # TS - Priority 1
            'run_out_speed': ('ROS', 'max'),      # ROS - Priority 2  
            'time_to_reach': ('TTR', 'min'),      # TTR - Priority 3
            'avg_speed': ('AS', 'max'),           # AS - Priority 4 (may need calculation)
            'avg_stride_length': ('ASL', 'max'),  # ASL - Priority 5
            'avg_stride_frequency': ('ASF', 'max') # ASF - Priority 6
        }
        
        # Priority hierarchy (higher number = higher priority)
        priority_order = ['top_speed', 'run_out_speed', 'time_to_reach', 'avg_speed', 'avg_stride_length', 'avg_stride_frequency']
        
        performance_rankings = {}
        
        # Extract and rank performers for each metric
        for metric_name, (tpd_field, direction) in metrics_config.items():
            valid_performers = []
            
            for horse_id, runner_data in tpd_runners.items():
                value = runner_data.get(tpd_field)
                
                # Special handling for avg_speed if not directly available
                if metric_name == 'avg_speed' and value is None:
                    # Try to calculate from other fields or skip
                    continue
                
                if value is not None:
                    try:
                        valid_performers.append((horse_id, float(value)))
                    except (ValueError, TypeError):
                        continue
            
            if valid_performers:
                # Sort by performance (ascending for TTR, descending for others)
                reverse_sort = (direction == 'max')
                sorted_performers = sorted(valid_performers, key=lambda x: x[1], reverse=reverse_sort)
                
                # Take top N performers
                top_performers = sorted_performers[:top_n]
                performance_rankings[metric_name] = [horse_id for horse_id, _ in top_performers]
                
                print(f"📊 {metric_name.upper()}: {[f'{h}({v:.1f})' for h, v in top_performers]}")
        
        # Assign one metric per horse using priority hierarchy
        horse_assignments = {}
        assigned_metrics = set()
        
        # Process horses in priority order
        for metric_name in priority_order:
            if metric_name not in performance_rankings:
                continue
                
            for rank_idx, horse_id in enumerate(performance_rankings[metric_name]):
                if horse_id not in horse_assignments:
                    # Get the actual performance value
                    tpd_field = metrics_config[metric_name][0]
                    value = tpd_runners[horse_id].get(tpd_field)
                    
                    rank_text = self.get_rank_text(rank_idx + 1, metric_name)  # PASS METRIC TYPE
                    
                    horse_assignments[horse_id] = {
                        'metric': metric_name,
                        'rank': rank_idx + 1,
                        'rank_text': rank_text,
                        'value': value,
                        'display_name': self.get_metric_display_name(metric_name)
                    }
                    assigned_metrics.add(f"{metric_name}_{rank_idx + 1}")
                    # break  # Horse gets first qualifying metric only
        
        print(f"🎯 ASSIGNMENTS: {len(horse_assignments)} horses assigned performance metrics")
        
        return performance_rankings, horse_assignments

    def get_rank_text(self, rank: int, metric_type: str = None) -> str:
        """Convert rank number to text - handles stride length differently"""
        if metric_type == 'avg_stride_length':
            # Stride Length: Shortest to Longest
            if rank == 1:
                return "longest"
            elif rank == 2:
                return "2nd longest"
            elif rank == 3:
                return "3rd longest"
            else:
                return f"{rank}th longest"
        else:
            # Speed metrics: fastest/slowest
            if rank == 1:
                return "fastest"
            elif rank == 2:
                return "2nd fastest"
            elif rank == 3:
                return "3rd fastest"
            else:
                return f"{rank}th fastest"

    def get_metric_display_name(self, metric_name: str) -> str:
        """Get display name for metric"""
        display_names = {
            'top_speed': 'top speed',
            'run_out_speed': 'run out speed',
            'time_to_reach': 'time to reach pace',
            'avg_speed': 'average speed',
            'avg_stride_length': 'stride length',
            'avg_stride_frequency': 'stride frequency'
        }
        return display_names.get(metric_name, metric_name)

    def get_horse_final_position(self, tpd_data: Dict, horse_id: str) -> int:
        """Get final position from TPD performance data"""
        cloth_num_short = f"{int(horse_id[-2:]):02d}"
        tpd_runners = tpd_data.get('runners', {})
        
        if cloth_num_short in tpd_runners:
            position = tpd_runners[cloth_num_short].get('position')
            if position:
                try:
                    return int(position)
                except (ValueError, TypeError):
                    return 999
        
        return 999

    async def extract_race_data(self, race_data: Dict, tpd_data: Dict, gps_position_data: Dict, horse_assignments: Dict) -> RaceData:
        """Extract race data with enhanced performance metric assignment"""
        
        horse_id = race_data.get('horse_id')
        total_runners = len(tpd_data.get('runners', {}))
        
        cloth_num_short = f"{int(horse_id[-2:]):02d}"
        
        # Get assigned performance metric for this horse
        assigned_metric = horse_assignments.get(cloth_num_short)
        
        gps_positions = self.build_position_timeline_from_gps(gps_position_data, horse_id, total_runners)
        
        # Check for final furlong surge (2+ position gain from 2f to 1f)
        final_furlong_surge = False
        if len(gps_positions) >= 2:
            pos_2f = None
            pos_1f = None
            for pos in gps_positions:
                if pos.furlong == '2f':
                    pos_2f = pos.position
                elif pos.furlong == '1f':
                    pos_1f = pos.position
            
            if pos_2f and pos_1f and pos_2f - pos_1f >= 2:
                final_furlong_surge = True
        
        # Extract performance metrics from TPD data
        performance_metrics = None
        tpd_runners = tpd_data.get('runners', {})
        
        if cloth_num_short in tpd_runners:
            tpd_runner = tpd_runners[cloth_num_short]
            performance_metrics = PerformanceMetrics(
                finish_time=tpd_runner.get('finish_time'),
                position=str(tpd_runner.get('position')),
                run_out_speed=tpd_runner.get('ROS'),
                velocity_peak=tpd_runner.get('VP'),
                avg_stride_length=tpd_runner.get('ASL'),
                avg_stride_frequency=tpd_runner.get('ASF'),
                time_to_reach_30=tpd_runner.get('TTR'),
                distance_beaten=tpd_runner.get('DB'),
                best_position=tpd_runner.get('BP'),
                avg_speed=tpd_runner.get('AS')
            )
        
        distance_back_for_horse = self.get_distance_back_for_horse(gps_position_data, horse_id)
        
        # Determine gate performance using intelligent clustering
        gate_performance = None
        if gps_positions:
            early_term = gps_positions[0].racing_term
            if "clear leader" in early_term:
                gate_performance = "broke sharply to establish a clear lead"
            elif "disputed" in early_term:
                gate_performance = "broke well to dispute the early lead"
            elif "tightly bunched leaders" in early_term:
                gate_performance = "broke well among the tightly bunched leaders"
            elif "leaders" in early_term:
                gate_performance = "broke well to track the leaders"
            elif "midfield" in early_term:
                gate_performance = "settled into midfield after the break"
            elif "detached" in early_term or "tailed off" in early_term:
                gate_performance = "broke slowly and was immediately detached"
            else:
                gate_performance = "settled towards the rear after the break"
        
        # Find significant position changes
        position_changes = []
        break_point = None
        best_position = 999
        best_furlong = None
        
        for i in range(1, len(gps_positions)):
            prev = gps_positions[i-1]
            curr = gps_positions[i]
            
            if curr.position < best_position:
                best_position = curr.position
                best_furlong = curr.furlong
            
            change = curr.position - prev.position
            term_changed = prev.racing_term != curr.racing_term
            
            if abs(change) >= 2 or term_changed:
                if "clear leader" in curr.racing_term and "clear leader" not in prev.racing_term:
                    desc = f"surged clear to lead at {curr.furlong}"
                elif "disputed" in curr.racing_term and change < 0:
                    desc = f"moved up to dispute the lead at {curr.furlong}"
                elif "leaders" in curr.racing_term and "leaders" not in prev.racing_term:
                    desc = f"made ground to join the leaders at {curr.furlong}"
                elif "detached" in curr.racing_term or "tailed off" in curr.racing_term:
                    desc = f"lost touch with the field at {curr.furlong}"
                elif change < 0:
                    desc = f"made ground from {prev.furlong} to {curr.furlong}"
                else:
                    desc = f"dropped back from {prev.furlong} to {curr.furlong}"
                
                position_changes.append({
                    'from_furlong': prev.furlong,
                    'to_furlong': curr.furlong,
                    'from_position': prev.position,
                    'to_position': curr.position,
                    'change': change,
                    'description': desc
                })
                
                if not break_point and change > 0:
                    break_point = desc
        
        final_position = race_data.get('final_position', 999)
        if best_position < final_position and best_furlong:
            break_point = f"challenged strongly at {best_furlong} before weakening"
        
        # Speed metrics
        speed_metrics = {}
        if performance_metrics:
            speed_metrics = {
                'max_speed_mph': performance_metrics.velocity_peak,
                'run_out_speed': performance_metrics.run_out_speed,
            }
        
        # Stride data
        stride_data = {}
        if performance_metrics and performance_metrics.avg_stride_length and performance_metrics.avg_stride_frequency:
            stride_data = {
                'avg_stride_length': performance_metrics.avg_stride_length,
                'avg_stride_frequency': performance_metrics.avg_stride_frequency
            }
        
        # Final phase analysis
        final_phase = None
        if performance_metrics and performance_metrics.run_out_speed:
            if performance_metrics.run_out_speed > 35:
                final_phase = "finished strongly"
            elif performance_metrics.run_out_speed < 30:
                final_phase = "weakened late"
            else:
                final_phase = "maintained pace"
        
        return RaceData(
            final_position=final_position,
            performance_metrics=performance_metrics,
            gps_positions=gps_positions,
            gate_performance=gate_performance,
            break_point=break_point,
            speed_metrics=speed_metrics,
            stride_data=stride_data,
            final_phase=final_phase,
            position_changes=position_changes,
            distance_back_data=distance_back_for_horse,
            assigned_performance_metric=assigned_metric,  # NEW: Include assigned metric
            final_furlong_surge=final_furlong_surge
        )

    async def generate_concise_commentary(self, race_data: RaceData) -> str:
        """Generate concise commentary with HIERARCHICAL performance metrics placement"""
        
        print(f"DEBUG: Horse {race_data.final_position} has {len(race_data.gps_positions)} GPS positions")

        position_timeline = []
        for pos in race_data.gps_positions:
            pos_text = f"{pos.furlong}: {pos.racing_term}"
            
            if pos.furlong in race_data.distance_back_data:
                distance = race_data.distance_back_data[pos.furlong]
                if distance == 1:
                    pos_text += f" ({distance} length behind)"
                else:
                    pos_text += f" ({distance} lengths behind)"
            position_timeline.append(pos_text)
        
        moves = []
        for change in race_data.position_changes:
            moves.append(change.get('description', 'Made positional change'))
        
        # HIERARCHICAL PERFORMANCE MENTIONS
        start_performance = ""   # TTR goes here
        middle_performance = ""  # Top Speed, Stride Length, Stride Frequency go here
        end_performance = ""     # ROS, Late Speed go here
        
        if race_data.assigned_performance_metric:
            metric_info = race_data.assigned_performance_metric
            metric_name = metric_info['metric']
            rank = metric_info['rank']
            rank_text = metric_info['rank_text']
            value = metric_info['value']
            display_name = metric_info['display_name']
            
            if metric_name == 'time_to_reach':
                start_performance = f"{rank_text} to reach pace in {value:.1f}s"
            elif metric_name == 'top_speed':
                if rank == 1:
                    middle_performance = f"recorded the race's fastest speed of {value:.1f}mph"
                else:
                    middle_performance = f"recorded {rank_text} speed of {value:.1f}mph"
            elif metric_name == 'avg_stride_length':
                middle_performance = f"{rank_text} stride length of {value:.1f}ft"
            elif metric_name == 'avg_stride_frequency':
                middle_performance = f"{rank_text} stride frequency of {value:.1f}/s"
            elif metric_name in ['run_out_speed', 'avg_speed']:
                end_performance = f"{rank_text} {display_name} of {value:.1f}mph"
        
        final_furlong_mention = ""
        if race_data.final_furlong_surge:
            final_furlong_mention = "flew home in final furlong"
        
        data_summary = f"""
FINAL POSITION: {race_data.final_position} (MUST include in commentary as "finished {self.ordinal(race_data.final_position)}")

GPS Position Flow (use the exact terms provided):
{chr(10).join(position_timeline)}

Key Moves:
{chr(10).join(moves) if moves else "Held position throughout"}

HIERARCHICAL Performance Data:
- Gate: {race_data.gate_performance or 'Standard break'}
- START Performance: {start_performance if start_performance else 'None'}
- MIDDLE Performance: {middle_performance if middle_performance else 'None'}  
- END Performance: {end_performance if end_performance else 'None'}
- Break Point: {race_data.break_point or 'No significant break point'}
- Final Phase: {race_data.final_phase or 'Maintained position'}
- Final Furlong: {final_furlong_mention if final_furlong_mention else 'Standard finish'}

CRITICAL HIERARCHY RULES:
- START mentions (TTR) go at the beginning after gate break
- MIDDLE mentions (Top Speed, Stride) go in the middle during race progression  
- END mentions (ROS, Late Speed) go at the end before finishing position

Generate simple racing commentary (1-3 lines) using basic UK English.
Use the exact positioning terms from the GPS Position Flow.
Place performance metrics in correct hierarchy positions.
Mention final furlong surge if applicable.
MUST end with "finished {self.ordinal(race_data.final_position)}"
"""
        
        try:
            if self.provider == 'anthropic':
                result = await self.generator.run(data_summary)
                return result.output
            elif self.provider == 'deepseek':
                response = self.generator.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.get_generator_prompt()},
                        {"role": "user", "content": data_summary}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Commentary generation error: {e}")
            raise

    async def reflect_and_improve(self, commentary: str, race_data: RaceData) -> Tuple[str, bool]:
        """FIXED: Unified validation with strict parsing and bulletproof prompts"""
        max_iterations = 3
        current_commentary = commentary
        
        for iteration in range(max_iterations):
            print(f"🔄 Validation iteration {iteration + 1}/{max_iterations}")
            
            gps_positions = {pos.furlong: pos.racing_term for pos in race_data.gps_positions}
            
            # Include assigned performance metric in validation
            assigned_metric_text = ""
            if race_data.assigned_performance_metric:
                metric_info = race_data.assigned_performance_metric
                assigned_metric_text = f"Should mention: {metric_info['rank_text']} {metric_info['display_name']}"
            
            validation_input = f"""
Commentary: {current_commentary}
Final Position: {race_data.final_position}
Expected Position Text: "finished {self.ordinal(race_data.final_position)}"

GPS Position Flow (should use these exact terms):
{chr(10).join([f"{k}: {v}" for k, v in gps_positions.items()])}

Distance Back Data (significant instances only):
{chr(10).join([f"{k}: {v} lengths" for k, v in race_data.distance_back_data.items()])}

Assigned Performance Metric:
{assigned_metric_text if assigned_metric_text else "No performance metric assigned"}

Validate that:
1. Commentary contains the exact finishing position
2. Uses intelligent positioning language (clear leader, disputed lead, tightly bunched, etc)
3. NO specific mid-race positions (reject "4th", "5th", etc.)
4. Performance metric mention matches assigned metric (if any)
5. Speed metrics are in correct units and ranking
6. Distance back claims match provided data (integer lengths, significant only)
7. Flows naturally like racing commentary
8. HIERARCHICAL PERFORMANCE PLACEMENT: TTR at start, Speed/Stride in middle, ROS at end
"""
            
            try:
                if self.provider == 'anthropic':
                    critique = await self.critic.run(validation_input)
                    result = critique.output
                    
                elif self.provider == 'deepseek':
                    response = self.critic.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.get_critic_prompt()},
                            {"role": "user", "content": validation_input}
                        ],
                        temperature=0.3,
                        max_tokens=300
                    )
                    result = self.parse_deepseek_critique(response.choices[0].message.content)
                
                if result.accurate and len(result.issues) == 0:
                    print(f"✅ {self.provider.upper()} validation passed (iter {iteration + 1})")
                    self.previous_commentaries.append(current_commentary)
                    return current_commentary, True
                
                print(f"🔄 Issues found: {result.issues}")
                
                if not result.accurate or any("position" in issue.lower() for issue in result.issues):
                    print(f"🔧 Fixing positioning issue (iter {iteration + 1})")
                    
                    expected_pos = f"finished {self.ordinal(race_data.final_position)}"
                    if expected_pos not in current_commentary:
                        if "finished" in current_commentary:
                            parts = current_commentary.split("finished")
                            current_commentary = parts[0].rstrip(", ") + f", {expected_pos}"
                        else:
                            current_commentary = current_commentary.rstrip(".") + f", {expected_pos}"
                else:
                    # FIXED: Bulletproof improvement prompt
                    improvement_prompt = f"""
SYSTEM ALERT: Your previous response had validation issues.

Original Commentary: {current_commentary}
Issues Found: {result.issues}
Suggestions: {result.suggestions}

CRITICAL: Do NOT change assigned performance metric types.

YOU MUST RESPOND IN THIS EXACT FORMAT OR THE SYSTEM WILL FAIL:

<commentary>your improved racing commentary here</commentary>
<notes>explain what you changed and why</notes>

REQUIREMENTS FOR <commentary> SECTION:
- Simple UK English, maximum 3 sentences
- MUST end with "finished {self.ordinal(race_data.final_position)}"
- NO explanations, notes, or meta-commentary
- ONLY the race description
- Follow hierarchical performance placement rules

REQUIREMENTS FOR <notes> SECTION:
- Explain your changes and reasoning
- Reference hierarchy placement decisions
- Mention data usage

EXAMPLE FORMAT:
<commentary>Settled in midfield after the break, made ground to join the leaders at 4f, weakened in final stages and finished 3rd.</commentary>
<notes>Simplified language, used exact GPS terms "midfield" and "leaders", ensured proper finish position.</notes>

RESPOND NOW IN THE REQUIRED FORMAT. NO OTHER FORMAT WILL BE ACCEPTED.
"""
                    
                    if self.provider == 'anthropic':
                        improved = await self.generator.run(improvement_prompt)
                        raw_response = improved.output
                    elif self.provider == 'deepseek':
                        response = self.generator.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": self.get_generator_prompt()},
                                {"role": "user", "content": improvement_prompt}
                            ],
                            temperature=0.5,
                            max_tokens=150
                        )
                        raw_response = response.choices[0].message.content

                    # FIXED: Strict parsing with retry mechanism
                    COMMENTARY_START_TAG = "<commentary>"
                    COMMENTARY_END_TAG = "</commentary>"
                    NOTES_START_TAG = "<notes>"
                    NOTES_END_TAG = "</notes>"

                    if COMMENTARY_START_TAG in raw_response and COMMENTARY_END_TAG in raw_response:
                        commentary_start = raw_response.find(COMMENTARY_START_TAG) + len(COMMENTARY_START_TAG)
                        commentary_end = raw_response.find(COMMENTARY_END_TAG)
                        current_commentary = raw_response[commentary_start:commentary_end].strip()
                        
                        # Extract notes for debugging  
                        if NOTES_START_TAG in raw_response and NOTES_END_TAG in raw_response:
                            notes_start = raw_response.find(NOTES_START_TAG) + len(NOTES_START_TAG)
                            notes_end = raw_response.find(NOTES_END_TAG)
                            improvement_notes = raw_response[notes_start:notes_end].strip()
                            print(f"🔧 AI Improvement Notes: {improvement_notes}")
                        
                        print(f"✅ Structured format used correctly")
                    else:
                        # STRICT MODE: One retry with ultra-strict prompt
                        print(f"❌ AI failed to use required format. Response: {raw_response[:200]}...")
                        print(f"🔄 Retrying with stricter prompt...")
                        
                        ultra_strict_prompt = f"""
FINAL WARNING: You FAILED to use the required format. 

REQUIRED FORMAT (copy exactly):
<commentary>race commentary only</commentary>
<notes>explanation only</notes>

Your task: Fix this commentary: {current_commentary}
Issues: {result.issues}

If you don't use the exact format above, the system will crash.
RESPOND NOW WITH THE REQUIRED TAGS.
"""
                        
                        if self.provider == 'anthropic':
                            retry_improved = await self.generator.run(ultra_strict_prompt)
                            retry_response = retry_improved.output
                        elif self.provider == 'deepseek':
                            retry_response = self.generator.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {"role": "system", "content": self.get_generator_prompt()},
                                    {"role": "user", "content": ultra_strict_prompt}
                                ],
                                temperature=0.3,
                                max_tokens=150
                            )
                            retry_response = retry_response.choices[0].message.content
                        
                        # Try parsing the retry
                        if COMMENTARY_START_TAG in retry_response and COMMENTARY_END_TAG in retry_response:
                            commentary_start = retry_response.find(COMMENTARY_START_TAG) + len(COMMENTARY_START_TAG)
                            commentary_end = retry_response.find(COMMENTARY_END_TAG)
                            current_commentary = retry_response[commentary_start:commentary_end].strip()
                            print(f"✅ Retry successful with structured format")
                        else:
                            # Last resort: aggressive cleanup
                            print(f"❌ Retry failed. Applying emergency cleanup...")
                            current_commentary = re.sub(r'\s*Note:.*$', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
                            current_commentary = current_commentary.strip()
                            print(f"🆘 Emergency cleanup result: {current_commentary}")

                    # Final validation to ensure no notes leaked through
                    if "Note:" in current_commentary or "Commentary" in current_commentary:
                        current_commentary = re.sub(r'\s*Note:.*$', '', current_commentary, flags=re.DOTALL | re.IGNORECASE)
                        current_commentary = re.sub(r'\s*Commentary.*$', '', current_commentary, flags=re.DOTALL | re.IGNORECASE)
                        current_commentary = current_commentary.strip()
                        print(f"🧹 Final cleanup applied: {current_commentary}")
                
            except Exception as e:
                print(f"❌ Validation error: {e}")
                break
        
        expected_pos = f"finished {self.ordinal(race_data.final_position)}"
        if expected_pos not in current_commentary:
            current_commentary = current_commentary.rstrip(".") + f", {expected_pos}"
        
        print(f"✅ {self.provider.upper()} validation completed")
        return current_commentary, True

    def ordinal(self, n: int) -> str:
        """Convert number to ordinal"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    async def process_horse(self, race_data: Dict, tpd_data: Dict, gps_position_data: Dict, horse_assignments: Dict) -> str:
        """Process single horse with enhanced performance metrics"""
        horse_name = race_data.get('horse_name', 'Unknown')
        print(f"🔄 Processing {horse_name}...")
        
        try:
            extracted_data = await self.extract_race_data(race_data, tpd_data, gps_position_data, horse_assignments)
            initial_commentary = await self.generate_concise_commentary(extracted_data)
            final_commentary, success = await self.reflect_and_improve(initial_commentary, extracted_data)
            
            print(f"✅ {horse_name}: Generated commentary using {self.provider.upper()}")
            return final_commentary
                
        except Exception as e:
            print(f"❌ {horse_name}: Error - {e}")
            raise

# Main execution
async def main():
    """Enhanced main execution with top-DECILE performance metrics"""
    
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ FATAL: config/config.json not found!")
        raise
    
    tpd_api_key = config.get('tpd_api_key')
    live_recording_api_key = config.get('tpd_live_recording_api_key')
    race_sharecode = config.get('race_sharecode', '01202506191430')
    
    missing_keys = []
    if not tpd_api_key:
        missing_keys.append('tpd_api_key')
    if not live_recording_api_key:
        missing_keys.append('tpd_live_recording_api_key')
    
    if missing_keys:
        print(f"❌ FATAL: Missing TPD API keys: {', '.join(missing_keys)}")
        raise ValueError(f"Missing configuration: {missing_keys}")
    
    # Print configured constants
    print(f"🔧 Racing Constants: METERS_PER_HORSE_LENGTH={METERS_PER_HORSE_LENGTH}, LEADER_GROUP_THRESHOLD={LEADER_GROUP_THRESHOLD}")
    
    agent = EnhancedRacingAgent(config, tpd_api_key, live_recording_api_key)
    
    tpd_data = agent.tpd_client.get_performance_metrics(race_sharecode)
    
    print("📡 Fetching GPS position data...")
    s3_url = agent.tpd_client.get_live_recording_url(race_sharecode)
    gps_position_data = {}
    
    if s3_url:
        gps_records = agent.tpd_client.fetch_gps_data(s3_url)
        gps_position_data = agent.tpd_client.extract_gps_position_data(gps_records)
        print(f"✅ Loaded GPS data: {len(gps_records)} records, {len(gps_position_data)} furlong markers")
    else:
        print("❌ FATAL: Could not fetch GPS data - position data unavailable")
        raise ValueError("GPS data required for position tracking")
    
    # ENHANCED: Get top-DECILE performance assignments
    performance_rankings, horse_assignments = agent.find_top_performance_horses(tpd_data)
    print(f"🏆 Enhanced Performance System: {len(horse_assignments)} horses with assigned metrics")
    
    print(f"\n🚀 Enhanced Racing Commentary System with Dynamic Furlong Processing")
    print(f"🤖 Provider: {agent.provider.upper()}")
    print(f"📋 Model: {agent.model_name}")
    print(f"🏁 Race: {race_sharecode}")
    print("📊 Using dynamic GPS furlong processing with hierarchical performance placement")
    print(f"✅ Loaded TPD performance data: {tpd_data.get('success', False)}")
    print(f"✅ GPS K=5 markers: {list(gps_position_data.keys())}")
    
    commentaries = {}
    agent.previous_commentaries = []
    
    tpd_runners = tpd_data.get('runners', {})
    
    for horse_id, runner_info in tpd_runners.items():
        horse_name = runner_info.get('horse', f"Horse {horse_id}")
        position = runner_info.get('position')
        
        if not position or position in ['NR', 'W']:
            continue
        
        try:
            final_position = int(position)
        except (ValueError, TypeError):
            final_position = 999
        
        full_horse_id = f"{race_sharecode}{horse_id}"
        
        race_data = {
            'horse_id': full_horse_id,
            'horse_name': horse_name,
            'final_position': final_position
        }
        
        commentary = await agent.process_horse(race_data, tpd_data, gps_position_data, horse_assignments)
        commentaries[horse_name] = commentary

        # Add small delay to prevent overwhelming
        await asyncio.sleep(1)
    
    print(f"\n{'='*80}")
    print(f"🎯 ENHANCED COMMENTARY RESULTS - DYNAMIC FURLONG PROCESSING")
    print(f"{'='*80}")
    
    position_order = sorted(
        tpd_runners.items(), 
        key=lambda x: (
            999 if not x[1].get('position') or x[1].get('position') in ['NR', 'W'] else int(x[1].get('position', 999))
        )
    )
    
    for horse_id, runner_info in position_order:
        horse_name = runner_info.get('horse', f"Horse {horse_id}")
        position = runner_info.get('position')
        
        if position and position not in ['NR', 'W']:
            print(f"\n🏇 {horse_name.upper()} - Horse #{int(horse_id)}")
            print(f"📍 Position: {position}")
            
            # Show assigned performance metric
            if horse_id in horse_assignments:
                metric_info = horse_assignments[horse_id]
                print(f"🎯 ASSIGNED METRIC: {metric_info['rank_text']} {metric_info['display_name']} ({metric_info['value']})")
            
            vp = runner_info.get('VP')
            ros = runner_info.get('ROS')
            asl = runner_info.get('ASL')
            asf = runner_info.get('ASF')
            
            if vp and ros:
                print(f"📊 Peak: {vp:.1f}mph | Finish: {ros:.1f}mph [TPD DATA]")
            
            if asl and asf:
                print(f"🏃 Stride: {asl:.1f}ft @ {asf:.1f}/s [TPD DATA]")
            
            full_horse_id = f"{race_sharecode}{horse_id}"
            horse_num = str(int(horse_id))
            gps_positions = []
            
            # DYNAMIC: Show actual GPS positions found
            available_furlongs = [f for f in gps_position_data.keys() if f.lower() != 'finish']
            sorted_display_furlongs = agent.sort_furlongs_by_distance(available_furlongs)
            for furlong in sorted_display_furlongs:
                if furlong in gps_position_data:
                    data = gps_position_data[furlong]
                    order = data.get('order', [])
                    if horse_num in order:
                        pos = order.index(horse_num) + 1
                        gps_positions.append(f"{furlong}:{pos}")
            
            if gps_positions:
                print(f"📍 GPS Positions: {' → '.join(gps_positions)}")
            
            print(f"🤖 {agent.provider.upper()}: {commentaries.get(horse_name, 'Processing failed')}")

if __name__ == "__main__":
    asyncio.run(main())