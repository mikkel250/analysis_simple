"""
Advanced Pattern Recognition Service

This module provides services for detecting various chart patterns,
including harmonic patterns and Elliott Waves.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Placeholder for Fibonacci and other utility functions if needed
# from ..utils import some_utility_function 

@dataclass
class HarmonicPattern:
    name: str
    points: Dict[str, float]  # e.g., {"X": 100, "A": 105, "B": 98, "C": 102, "D": 95}
    completion_price_range: Optional[tuple[float, float]] = None
    probability: Optional[float] = None
    educational_notes: str = ""

@dataclass
class ElliottWaveSegment:
    wave_type: str  # 'impulse' or 'correction'
    start_price: float
    end_price: float
    sub_waves: List[Any] = field(default_factory=list) # Can be List[ElliottWaveSegment]
    degree: Optional[str] = None # e.g., Primary, Intermediate, Minor

class HarmonicPatternDetector:
    """
    Detects various harmonic patterns.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']

    def _find_extreme_points(self, window: int = 5, min_change_pct: float = 0.02) -> pd.DataFrame:
        """
        Identifies potential swing points (highs and lows) with improved logic.
        
        Args:
            window: Number of periods to look back/forward for local extremes
            min_change_pct: Minimum percentage change to qualify as a swing point
        """
        if len(self.data) < window * 2 + 1:
            return pd.DataFrame(columns=['price', 'type'])
        
        swings = []
        
        # Find local highs
        for i in range(window, len(self.data) - window):
            current_high = self.data['high'].iloc[i]
            is_local_high = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - window, i + window + 1):
                if j != i and self.data['high'].iloc[j] >= current_high:
                    is_local_high = False
                    break
            
            if is_local_high:
                # Check minimum change requirement
                prev_low = self.data['low'].iloc[max(0, i-window):i].min()
                if (current_high - prev_low) / prev_low >= min_change_pct:
                    swings.append({
                        'timestamp': self.data.index[i],
                        'price': current_high,
                        'type': 'high'
                    })
        
        # Find local lows
        for i in range(window, len(self.data) - window):
            current_low = self.data['low'].iloc[i]
            is_local_low = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - window, i + window + 1):
                if j != i and self.data['low'].iloc[j] <= current_low:
                    is_local_low = False
                    break
            
            if is_local_low:
                # Check minimum change requirement
                prev_high = self.data['high'].iloc[max(0, i-window):i].max()
                if (prev_high - current_low) / prev_high >= min_change_pct:
                    swings.append({
                        'timestamp': self.data.index[i],
                        'price': current_low,
                        'type': 'low'
                    })
        
        if not swings:
            return pd.DataFrame(columns=['price', 'type'])
        
        # Convert to DataFrame and sort by timestamp
        swings_df = pd.DataFrame(swings)
        swings_df.set_index('timestamp', inplace=True)
        swings_df = swings_df.sort_index()
        
        # Remove consecutive same-type swings, keeping the more extreme one
        filtered_swings = []
        for i, (timestamp, row) in enumerate(swings_df.iterrows()):
            if i == 0:
                filtered_swings.append({'timestamp': timestamp, 'price': row['price'], 'type': row['type']})
            else:
                prev_swing = filtered_swings[-1]
                if row['type'] != prev_swing['type']:
                    filtered_swings.append({'timestamp': timestamp, 'price': row['price'], 'type': row['type']})
                else:
                    # Same type, keep the more extreme one
                    if ((row['type'] == 'high' and row['price'] > prev_swing['price']) or
                        (row['type'] == 'low' and row['price'] < prev_swing['price'])):
                        filtered_swings[-1] = {'timestamp': timestamp, 'price': row['price'], 'type': row['type']}
        
        if not filtered_swings:
            return pd.DataFrame(columns=['price', 'type'])
        
        result_df = pd.DataFrame(filtered_swings)
        result_df.set_index('timestamp', inplace=True)
        return result_df

    def _check_fibonacci_ratio(self, value: float, target: float, tolerance: float = 0.05) -> bool:
        """
        Check if a value is within tolerance of a Fibonacci ratio.
        
        Args:
            value: The calculated ratio
            target: The target Fibonacci ratio
            tolerance: Allowed deviation (default 5%)
        """
        return abs(value - target) <= tolerance

    def _calculate_retracement(self, start_price: float, end_price: float, current_price: float) -> float:
        """
        Calculate the retracement ratio.
        
        Args:
            start_price: Starting price of the move
            end_price: Ending price of the move
            current_price: Current retracement price
        """
        if start_price == end_price:
            return 0.0
        return abs(current_price - end_price) / abs(start_price - end_price)

    def detect_gartley_pattern(self) -> List[HarmonicPattern]:
        """
        Detects Gartley patterns using Fibonacci ratios.
        
        Gartley pattern rules:
        - B retraces 0.618 of XA
        - C retraces 0.382-0.886 of AB
        - D retraces 0.786 of XA
        - D extends 1.272-1.618 of BC
        """
        patterns = []
        swing_points = self._find_extreme_points()
        
        if len(swing_points) < 5:
            return patterns
        
        # Convert to list for easier indexing
        swings = [(idx, row['price'], row['type']) for idx, row in swing_points.iterrows()]
        
        # Look for XABCD patterns (need alternating high-low or low-high)
        for i in range(len(swings) - 4):
            x_time, x_price, x_type = swings[i]
            a_time, a_price, a_type = swings[i + 1]
            b_time, b_price, b_type = swings[i + 2]
            c_time, c_price, c_type = swings[i + 3]
            d_time, d_price, d_type = swings[i + 4]
            
            # Check if we have alternating pattern
            if not (x_type != a_type != b_type != c_type != d_type):
                continue
            
            # Check Gartley ratios
            try:
                # B retracement of XA
                ab_retracement = self._calculate_retracement(x_price, a_price, b_price)
                if not self._check_fibonacci_ratio(ab_retracement, 0.618, 0.1):
                    continue
                
                # C retracement of AB
                bc_retracement = self._calculate_retracement(a_price, b_price, c_price)
                if not (0.382 <= bc_retracement <= 0.886):
                    continue
                
                # D retracement of XA
                ad_retracement = self._calculate_retracement(x_price, a_price, d_price)
                if not self._check_fibonacci_ratio(ad_retracement, 0.786, 0.1):
                    continue
                
                # D extension of BC
                cd_extension = abs(d_price - c_price) / abs(b_price - c_price) if b_price != c_price else 0
                if not (1.272 <= cd_extension <= 1.618):
                    continue
                
                # If all ratios check out, we have a Gartley pattern
                pattern_type = "Bullish Gartley" if d_type == "low" else "Bearish Gartley"
                
                # Calculate completion probability based on how close ratios are to ideal
                ratio_scores = [
                    1 - abs(ab_retracement - 0.618) / 0.1,
                    1 - min(abs(bc_retracement - 0.382), abs(bc_retracement - 0.886)) / 0.1,
                    1 - abs(ad_retracement - 0.786) / 0.1,
                    1 - min(abs(cd_extension - 1.272), abs(cd_extension - 1.618)) / 0.1
                ]
                probability = max(0.5, min(0.95, sum(ratio_scores) / len(ratio_scores)))
                
                # Calculate potential reversal zone
                if d_type == "low":
                    # Bullish pattern - expect bounce from D
                    target_1 = d_price + (a_price - d_price) * 0.382
                    target_2 = d_price + (a_price - d_price) * 0.618
                else:
                    # Bearish pattern - expect decline from D
                    target_1 = d_price - (d_price - a_price) * 0.382
                    target_2 = d_price - (d_price - a_price) * 0.618
                
                pattern = HarmonicPattern(
                    name=pattern_type,
                    points={
                        "X": x_price,
                        "A": a_price,
                        "B": b_price,
                        "C": c_price,
                        "D": d_price
                    },
                    completion_price_range=(min(target_1, target_2), max(target_1, target_2)),
                    probability=probability,
                    educational_notes=f"Gartley pattern with AB={ab_retracement:.3f}, BC={bc_retracement:.3f}, AD={ad_retracement:.3f}, CD={cd_extension:.3f} ratios. This is a reversal pattern suggesting price may reverse at point D."
                )
                patterns.append(pattern)
                
            except (ZeroDivisionError, ValueError):
                # Skip if calculation fails
                continue
        
        return patterns

    def detect_butterfly_pattern(self) -> List[HarmonicPattern]:
        """
        Detects Butterfly patterns using Fibonacci ratios.
        
        Butterfly pattern rules:
        - B retraces 0.786 of XA
        - C retraces 0.382-0.886 of AB
        - D extends 1.272-1.618 of XA
        - D extends 1.618-2.24 of BC
        """
        patterns = []
        swing_points = self._find_extreme_points()
        
        if len(swing_points) < 5:
            return patterns
        
        # Convert to list for easier indexing
        swings = [(idx, row['price'], row['type']) for idx, row in swing_points.iterrows()]
        
        # Look for XABCD patterns (need alternating high-low or low-high)
        for i in range(len(swings) - 4):
            x_time, x_price, x_type = swings[i]
            a_time, a_price, a_type = swings[i + 1]
            b_time, b_price, b_type = swings[i + 2]
            c_time, c_price, c_type = swings[i + 3]
            d_time, d_price, d_type = swings[i + 4]
            
            # Check if we have alternating pattern
            if not (x_type != a_type != b_type != c_type != d_type):
                continue
            
            # Check Butterfly ratios
            try:
                # B retracement of XA
                ab_retracement = self._calculate_retracement(x_price, a_price, b_price)
                if not self._check_fibonacci_ratio(ab_retracement, 0.786, 0.1):
                    continue
                
                # C retracement of AB
                bc_retracement = self._calculate_retracement(a_price, b_price, c_price)
                if not (0.382 <= bc_retracement <= 0.886):
                    continue
                
                # D extension of XA (beyond A)
                xa_distance = abs(a_price - x_price)
                if d_type == x_type:  # D should be on same side as X relative to A
                    continue
                
                # Calculate D extension ratio relative to XA
                if (x_type == "low" and a_type == "high"):  # Bullish setup
                    if d_price <= a_price:  # D should extend beyond A
                        continue
                    ad_extension = (d_price - a_price) / xa_distance
                else:  # Bearish setup
                    if d_price >= a_price:  # D should extend beyond A
                        continue
                    ad_extension = (a_price - d_price) / xa_distance
                
                if not (1.272 <= ad_extension <= 1.618):
                    continue
                
                # D extension of BC
                cd_extension = abs(d_price - c_price) / abs(b_price - c_price) if b_price != c_price else 0
                if not (1.618 <= cd_extension <= 2.24):
                    continue
                
                # If all ratios check out, we have a Butterfly pattern
                pattern_type = "Bullish Butterfly" if d_type == "high" else "Bearish Butterfly"
                
                # Calculate completion probability based on how close ratios are to ideal
                ratio_scores = [
                    1 - abs(ab_retracement - 0.786) / 0.1,
                    1 - min(abs(bc_retracement - 0.382), abs(bc_retracement - 0.886)) / 0.1,
                    1 - min(abs(ad_extension - 1.272), abs(ad_extension - 1.618)) / 0.1,
                    1 - min(abs(cd_extension - 1.618), abs(cd_extension - 2.24)) / 0.1
                ]
                probability = max(0.5, min(0.95, sum(ratio_scores) / len(ratio_scores)))
                
                # Calculate potential reversal zone
                if d_type == "high":
                    # Bearish pattern - expect decline from D
                    target_1 = d_price - (d_price - a_price) * 0.382
                    target_2 = d_price - (d_price - a_price) * 0.618
                else:
                    # Bullish pattern - expect bounce from D
                    target_1 = d_price + (a_price - d_price) * 0.382
                    target_2 = d_price + (a_price - d_price) * 0.618
                
                pattern = HarmonicPattern(
                    name=pattern_type,
                    points={
                        "X": x_price,
                        "A": a_price,
                        "B": b_price,
                        "C": c_price,
                        "D": d_price
                    },
                    completion_price_range=(min(target_1, target_2), max(target_1, target_2)),
                    probability=probability,
                    educational_notes=f"Butterfly pattern with AB={ab_retracement:.3f}, BC={bc_retracement:.3f}, AD={ad_extension:.3f}, CD={cd_extension:.3f} ratios. This is an extension pattern where D extends beyond the initial XA move, often indicating exhaustion and potential reversal."
                )
                patterns.append(pattern)
                
            except (ZeroDivisionError, ValueError):
                # Skip if calculation fails
                continue
        
        return patterns

    def detect_bat_pattern(self) -> List[HarmonicPattern]:
        """Detects Bat patterns."""
        # Placeholder for Bat pattern logic
        # Key Fibonacci ratios:
        # B between 0.382 and 0.5 of XA
        # C between 0.382 and 0.886 of AB
        # D at 0.886 of XA
        # D between 1.618 and 2.618 of BC
        return []

    def detect_crab_pattern(self) -> List[HarmonicPattern]:
        """Detects Crab patterns."""
        # Placeholder for Crab pattern logic
        # Key Fibonacci ratios:
        # B between 0.382 and 0.618 of XA
        # C between 0.382 and 0.886 of AB
        # D at 1.618 of XA
        # D between 2.24 and 3.618 of BC
        return []

    def get_pattern_probabilities(self, pattern: HarmonicPattern) -> float:
        """
        Calculate the probability of a given pattern's completion.
        This uses the probability already calculated during pattern detection.
        """
        if pattern.probability is not None:
            return pattern.probability
        
        # Fallback calculation based on pattern type
        if "Gartley" in pattern.name:
            return 0.75  # Gartley patterns are generally reliable
        elif "Butterfly" in pattern.name:
            return 0.70  # Butterfly patterns are also quite reliable
        elif "Bat" in pattern.name:
            return 0.80  # Bat patterns have strict ratios, higher reliability
        elif "Crab" in pattern.name:
            return 0.65  # Crab patterns are extreme, moderate reliability
        
        return 0.60  # Default probability

    def get_educational_content(self, pattern_name: str) -> str:
        """Provides detailed educational content for specific patterns."""
        content = {
            "Gartley": """The Gartley pattern, discovered by H.M. Gartley in 1935, is one of the most reliable harmonic patterns. It's a retracement and continuation pattern that follows specific Fibonacci ratios:

Key Ratios:
- Point B: 61.8% retracement of XA
- Point C: 38.2% to 88.6% retracement of AB  
- Point D: 78.6% retracement of XA

Trading Strategy:
- Enter at point D (Potential Reversal Zone)
- Stop loss beyond D
- First target: 38.2% of AD
- Second target: 61.8% of AD

The pattern suggests that after the completion at point D, price should reverse in the direction of the original XA move.""",

            "Butterfly": """The Butterfly pattern is an extension pattern where point D extends beyond the initial XA leg, often indicating price exhaustion and potential reversal.

Key Ratios:
- Point B: 78.6% retracement of XA
- Point C: 38.2% to 88.6% retracement of AB
- Point D: 127.2% to 161.8% extension of XA
- Point D: 161.8% to 224% extension of BC

Trading Strategy:
- Enter at point D (Potential Reversal Zone)
- Stop loss beyond D
- Target the 38.2% to 61.8% retracement of the entire AD move

The Butterfly often occurs at significant support/resistance levels and can signal major trend reversals.""",

            "Bat": """The Bat pattern is a retracement pattern with tighter Fibonacci requirements, making it more precise but less frequent.

Key Ratios:
- Point B: 38.2% to 50% retracement of XA
- Point C: 38.2% to 88.6% retracement of AB
- Point D: 88.6% retracement of XA
- Point D: 161.8% to 261.8% extension of BC

The Bat pattern offers high probability trades due to its strict ratio requirements.""",

            "Crab": """The Crab pattern is the most extreme harmonic pattern, with point D extending significantly beyond the initial XA move.

Key Ratios:
- Point B: 38.2% to 61.8% retracement of XA
- Point C: 38.2% to 88.6% retracement of AB
- Point D: 161.8% extension of XA
- Point D: 224% to 361.8% extension of BC

The Crab pattern often marks major market turning points due to its extreme extension."""
        }
        return content.get(pattern_name, "Pattern information not available.")


class ElliottWaveAnalyzer:
    """
    Analyzes Elliott Wave patterns.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']

    def _find_significant_swings(self, threshold: float = 0.03) -> pd.DataFrame:
        """
        Identifies significant swing points for Elliott Wave counting.
        Uses the harmonic detector's method with a higher threshold for significance.
        """
        # Use the improved swing detection from harmonic detector
        detector = HarmonicPatternDetector(self.data)
        return detector._find_extreme_points(window=7, min_change_pct=threshold)

    def identify_impulse_waves(self) -> List[ElliottWaveSegment]:
        """
        Identifies potential 5-wave impulse sequences.
        Rules:
        - Wave 2 does not retrace more than 100% of Wave 1.
        - Wave 3 is often the longest and never the shortest of the impulse waves (1, 3, 5).
        - Wave 4 does not overlap with Wave 1 (except in diagonals).
        """
        impulse_waves = []
        swing_points = self._find_significant_swings()
        
        if len(swing_points) < 6:  # Need at least 6 points for 5 waves (0,1,2,3,4,5)
            return impulse_waves
        
        # Convert to list for easier indexing
        swings = [(idx, row['price'], row['type']) for idx, row in swing_points.iterrows()]
        
        # Look for 5-wave impulse patterns
        for i in range(len(swings) - 5):
            try:
                # Get 6 points for 5 waves (0,1,2,3,4,5)
                points = []
                for j in range(6):
                    time, price, swing_type = swings[i + j]
                    points.append((time, price, swing_type))
                
                # Check if we have alternating pattern (required for impulse)
                types = [p[2] for p in points]
                if not all(types[k] != types[k+1] for k in range(5)):
                    continue
                
                # Extract prices for wave analysis
                prices = [p[1] for p in points]
                
                # Determine if this is a bullish or bearish impulse
                if types[0] == "low" and types[1] == "high":  # Bullish impulse
                    # Wave 1: 0 to 1 (up)
                    # Wave 2: 1 to 2 (down) - should not retrace more than 100% of wave 1
                    # Wave 3: 2 to 3 (up) - should be longer than wave 1
                    # Wave 4: 3 to 4 (down) - should not overlap with wave 1
                    # Wave 5: 4 to 5 (up)
                    
                    wave1_size = prices[1] - prices[0]
                    wave2_retrace = prices[1] - prices[2]
                    wave3_size = prices[3] - prices[2]
                    wave4_retrace = prices[3] - prices[4]
                    wave5_size = prices[5] - prices[4]
                    
                    # Rule 1: Wave 2 does not retrace more than 100% of Wave 1
                    if wave2_retrace >= wave1_size:
                        continue
                    
                    # Rule 2: Wave 3 is never the shortest
                    if wave3_size <= wave1_size and wave3_size <= wave5_size:
                        continue
                    
                    # Rule 3: Wave 4 does not overlap with Wave 1 (price[4] should be above price[1])
                    if prices[4] <= prices[1]:
                        continue
                    
                    # If all rules pass, we have a potential bullish impulse
                    impulse_wave = ElliottWaveSegment(
                        wave_type="impulse",
                        start_price=prices[0],
                        end_price=prices[5],
                        sub_waves=[
                            ElliottWaveSegment("1", prices[0], prices[1]),
                            ElliottWaveSegment("2", prices[1], prices[2]),
                            ElliottWaveSegment("3", prices[2], prices[3]),
                            ElliottWaveSegment("4", prices[3], prices[4]),
                            ElliottWaveSegment("5", prices[4], prices[5])
                        ],
                        degree="Minor"
                    )
                    impulse_waves.append(impulse_wave)
                    
                elif types[0] == "high" and types[1] == "low":  # Bearish impulse
                    # Similar logic but inverted for bearish impulse
                    wave1_size = prices[0] - prices[1]
                    wave2_retrace = prices[2] - prices[1]
                    wave3_size = prices[2] - prices[3]
                    wave4_retrace = prices[4] - prices[3]
                    wave5_size = prices[4] - prices[5]
                    
                    # Rule 1: Wave 2 does not retrace more than 100% of Wave 1
                    if wave2_retrace >= wave1_size:
                        continue
                    
                    # Rule 2: Wave 3 is never the shortest
                    if wave3_size <= wave1_size and wave3_size <= wave5_size:
                        continue
                    
                    # Rule 3: Wave 4 does not overlap with Wave 1 (price[4] should be below price[1])
                    if prices[4] >= prices[1]:
                        continue
                    
                    # If all rules pass, we have a potential bearish impulse
                    impulse_wave = ElliottWaveSegment(
                        wave_type="impulse",
                        start_price=prices[0],
                        end_price=prices[5],
                        sub_waves=[
                            ElliottWaveSegment("1", prices[0], prices[1]),
                            ElliottWaveSegment("2", prices[1], prices[2]),
                            ElliottWaveSegment("3", prices[2], prices[3]),
                            ElliottWaveSegment("4", prices[3], prices[4]),
                            ElliottWaveSegment("5", prices[4], prices[5])
                        ],
                        degree="Minor"
                    )
                    impulse_waves.append(impulse_wave)
                    
            except (IndexError, ValueError, ZeroDivisionError):
                continue
        
        return impulse_waves

    def identify_corrective_waves(self) -> List[ElliottWaveSegment]:
        """
        Identifies potential 3-wave corrective sequences (e.g., ABC).
        Common corrections: Zigzag (5-3-5), Flat (3-3-5), Triangle (3-3-3-3-3).
        """
        corrective_waves = []
        swing_points = self._find_significant_swings()
        
        if len(swing_points) < 4:  # Need at least 4 points for 3 waves (A,B,C,end)
            return corrective_waves
        
        # Convert to list for easier indexing
        swings = [(idx, row['price'], row['type']) for idx, row in swing_points.iterrows()]
        
        # Look for 3-wave corrective patterns (ABC)
        for i in range(len(swings) - 3):
            try:
                # Get 4 points for 3 waves (A,B,C,end)
                a_time, a_price, a_type = swings[i]
                b_time, b_price, b_type = swings[i + 1]
                c_time, c_price, c_type = swings[i + 2]
                end_time, end_price, end_type = swings[i + 3]
                
                # Check if we have alternating pattern
                if not (a_type != b_type != c_type != end_type):
                    continue
                
                # Basic ABC correction pattern
                if a_type == "high" and b_type == "low" and c_type == "high":
                    # Bearish correction: A(high) -> B(low) -> C(high)
                    wave_a_size = a_price - b_price
                    wave_c_size = c_price - b_price
                    
                    # C should not exceed A significantly (for a typical correction)
                    if c_price > a_price * 1.1:  # Allow 10% overshoot
                        continue
                    
                    corrective_wave = ElliottWaveSegment(
                        wave_type="correction",
                        start_price=a_price,
                        end_price=c_price,
                        sub_waves=[
                            ElliottWaveSegment("A", a_price, b_price),
                            ElliottWaveSegment("B", b_price, c_price),
                            ElliottWaveSegment("C", c_price, end_price)
                        ],
                        degree="Minor"
                    )
                    corrective_waves.append(corrective_wave)
                    
                elif a_type == "low" and b_type == "high" and c_type == "low":
                    # Bullish correction: A(low) -> B(high) -> C(low)
                    wave_a_size = b_price - a_price
                    wave_c_size = b_price - c_price
                    
                    # C should not go below A significantly (for a typical correction)
                    if c_price < a_price * 0.9:  # Allow 10% undershoot
                        continue
                    
                    corrective_wave = ElliottWaveSegment(
                        wave_type="correction",
                        start_price=a_price,
                        end_price=c_price,
                        sub_waves=[
                            ElliottWaveSegment("A", a_price, b_price),
                            ElliottWaveSegment("B", b_price, c_price),
                            ElliottWaveSegment("C", c_price, end_price)
                        ],
                        degree="Minor"
                    )
                    corrective_waves.append(corrective_wave)
                    
            except (IndexError, ValueError, ZeroDivisionError):
                continue
        
        return corrective_waves

    def get_wave_probabilities(self, wave_segment: ElliottWaveSegment) -> float:
        """
        Calculate the probability of a given wave count being valid.
        """
        # Basic probability based on wave structure adherence
        if wave_segment.wave_type == "impulse":
            # Higher probability for impulse waves that follow strict rules
            return 0.75
        elif wave_segment.wave_type == "correction":
            # Moderate probability for corrective waves
            return 0.65
        return 0.5

    def get_educational_content(self, wave_type: str) -> str:
        """Provides educational content for Elliott Wave concepts."""
        content = {
            "ImpulseWave": "An impulse wave is composed of five sub-waves...",
            "CorrectiveWave": "A corrective wave is typically composed of three sub-waves...",
            "FibonacciRatios": "Fibonacci ratios are crucial in EW for projecting wave targets..."
        }
        return content.get(wave_type, "Elliott Wave information not available.")

# Main service class to integrate different analyzers
class PatternRecognitionService:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.harmonic_detector = HarmonicPatternDetector(data)
        self.elliott_wave_analyzer = ElliottWaveAnalyzer(data)

    def find_harmonic_patterns(self) -> List[HarmonicPattern]:
        """Finds all supported harmonic patterns."""
        patterns = []
        patterns.extend(self.harmonic_detector.detect_gartley_pattern())
        patterns.extend(self.harmonic_detector.detect_butterfly_pattern())
        patterns.extend(self.harmonic_detector.detect_bat_pattern())
        patterns.extend(self.harmonic_detector.detect_crab_pattern())
        
        for p in patterns:
            p.probability = self.harmonic_detector.get_pattern_probabilities(p)
            p.educational_notes = self.harmonic_detector.get_educational_content(p.name)
        return patterns

    def analyze_elliott_waves(self) -> Dict[str, List[ElliottWaveSegment]]:
        """Analyzes Elliott Wave structure."""
        impulse_waves = self.elliott_wave_analyzer.identify_impulse_waves()
        corrective_waves = self.elliott_wave_analyzer.identify_corrective_waves()
        
        # Attach probabilities and educational content if desired
        # for wave in impulse_waves + corrective_waves:
        #     wave.probability = self.elliott_wave_analyzer.get_wave_probabilities(wave)
            # wave.educational_notes = self.elliott_wave_analyzer.get_educational_content(wave.wave_type)

        return {
            "impulse_waves": impulse_waves,
            "corrective_waves": corrective_waves,
            "education": {
                 "general": self.elliott_wave_analyzer.get_educational_content("FibonacciRatios"),
                 "impulse": self.elliott_wave_analyzer.get_educational_content("ImpulseWave"),
                 "corrective": self.elliott_wave_analyzer.get_educational_content("CorrectiveWave"),
            }
        }

    def get_pattern_completion_probabilities(self, pattern: Any) -> float:
        """Generic method to get probability for any pattern type."""
        if isinstance(pattern, HarmonicPattern):
            return self.harmonic_detector.get_pattern_probabilities(pattern)
        elif isinstance(pattern, ElliottWaveSegment):
            return self.elliott_wave_analyzer.get_wave_probabilities(pattern)
        return 0.0

    def get_educational_material(self, item_name: str, item_type: str = "harmonic") -> str:
        """
        Get educational material for a pattern or wave type.
        item_type can be 'harmonic' or 'elliott'.
        """
        if item_type == "harmonic":
            return self.harmonic_detector.get_educational_content(item_name)
        elif item_type == "elliott":
            return self.elliott_wave_analyzer.get_educational_content(item_name)
        return "Educational content not found."

if __name__ == '__main__':
    # Example Usage (requires a DataFrame with 'high', 'low', 'close' columns)
    # Create a sample DataFrame for testing
    sample_data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                      '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                      '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                      '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
                                      '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25']),
        'open':  [100, 102, 101, 105, 103, 100,  98, 100, 102, 105, 103, 101, 99, 95, 98, 100, 97, 95, 98, 100, 102, 100, 98, 96, 99],
        'high':  [103, 105, 104, 108, 106, 102, 100, 103, 105, 108, 106, 103, 101, 99, 100, 102, 99, 98, 100, 103, 104, 102, 100, 98, 101],
        'low':   [99,  101, 100, 102, 101,  99,  97,  98, 100, 102, 101, 100, 98, 94, 96, 98, 96, 94, 96, 99, 100, 99, 97, 95, 97],
        'close': [102, 104, 102, 107, 104, 101,  99, 101, 104, 107, 105, 102, 100, 96, 99, 101, 98, 96, 99, 101, 103, 101, 99, 97, 100],
        'volume':[1000,1200,1100,1500,1300,1000,900,1100,1300,1600,1400,1200,1000,800,900,1100,950,850,950,1100,1200,1000,900,800,1000]
    }
    df = pd.DataFrame(sample_data)
    df.set_index('timestamp', inplace=True)

    service = PatternRecognitionService(df)
    
    print("--- Harmonic Patterns ---")
    harmonic_patterns = service.find_harmonic_patterns()
    if harmonic_patterns:
        for pattern in harmonic_patterns:
            print(f"Found: {pattern.name}")
            print(f"  Points: {pattern.points}")
            print(f"  Probability: {pattern.probability}")
            print(f"  Notes: {pattern.educational_notes}")
    else:
        print("No harmonic patterns found with current (placeholder) logic.")

    print("\n--- Elliott Wave Analysis ---")
    ew_analysis = service.analyze_elliott_waves()
    print(f"Impulse Waves Found: {len(ew_analysis['impulse_waves'])}")
    # for wave in ew_analysis['impulse_waves']: print(wave)
    print(f"Corrective Waves Found: {len(ew_analysis['corrective_waves'])}")
    # for wave in ew_analysis['corrective_waves']: print(wave)
    print(f"Educational Content (General EW): {ew_analysis['education']['general']}")

    print("\n--- Specific Educational Content ---")
    print(f"Gartley Info: {service.get_educational_material('Gartley', 'harmonic')}")
    print(f"Impulse Wave Info: {service.get_educational_material('ImpulseWave', 'elliott')}")

    # Example of finding swing points (for testing internal logic)
    # swings = service.harmonic_detector._find_extreme_points()
    # print("\n--- Swing Points ---")
    # print(swings) 