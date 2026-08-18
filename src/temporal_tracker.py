import time
from collections import deque

class TrafficLightTracker:
    """
    Tracks traffic light status over consecutive video frames,
    applies temporal smoothing to eliminate flicker noise, and estimates transitions.
    """
    def __init__(self, buffer_size=10):
        self.history = deque(maxlen=buffer_size)
        self.current_state = "UNKNOWN"
        self.confidence = 0.0
        self.state_start_time = time.time()
        self.last_countdown = None
        self.last_countdown_time = None

    def update(self, detected_state, conf, countdown_val=None):
        """
        Updates the tracker with new detection frame.
        detected_state: 'Red', 'Green', or None
        """
        if detected_state is not None:
            self.history.append((detected_state, conf))
        else:
            self.history.append(("UNKNOWN", 0.0))

        # Temporal Majority Voting
        valid_states = [s for s, c in self.history if s != "UNKNOWN"]
        if valid_states:
            # Count occurrences
            counts = {s: valid_states.count(s) for s in set(valid_states)}
            smoothed_state = max(counts, key=counts.get)
            avg_conf = sum(c for s, c in self.history if s == smoothed_state) / max(1, counts[smoothed_state])
            
            if smoothed_state != self.current_state:
                self.current_state = smoothed_state
                self.state_start_time = time.time()
                
            self.confidence = avg_conf
        else:
            self.current_state = "UNKNOWN"
            self.confidence = 0.0

        # Countdown update
        if countdown_val is not None and isinstance(countdown_val, int):
            self.last_countdown = countdown_val
            self.last_countdown_time = time.time()

        return self.get_summary()

    def get_summary(self):
        elapsed = time.time() - self.state_start_time
        remaining = None
        
        if self.last_countdown is not None and self.last_countdown_time is not None:
            time_since_read = time.time() - self.last_countdown_time
            estimated_rem = max(0, int(self.last_countdown - time_since_read))
            remaining = f"{estimated_rem}s (Verified Countdown)"
        else:
            remaining = f"Active: {int(elapsed)}s (Temporal Tracker)"

        return {
            "state": self.current_state,
            "confidence": round(self.confidence, 2),
            "remaining_time": remaining
        }
