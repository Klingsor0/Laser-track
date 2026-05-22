"""
Optical Experiment Data Classes
Pure data containers for angle measurement experiments - no UI dependencies
Framework-agnostic design for easy migration to PyQt6 later
"""

import numpy as np
from datetime import datetime
import uuid
import json


class CaptureSession:
    """
    Single acquisition session at one distance measurement
    Contains N frames captured at a specific distance and their extracted angles
    """
    
    def __init__(self, distance, unit='mm', description=''):
        """
        Initialize a new capture session
        
        Args:
            distance: Physical distance measurement (e.g., 10.0)
            unit: Unit of distance measurement (default: 'mm')
            description: Optional description of this session
        """
        self.distance = distance
        self.unit = unit
        self.description = description
        
        # Data storage
        self.frames = []           # List of numpy arrays (images)
        self.angles = []           # List of extracted angles (degrees)
        self.snake_results = []    # List of optimized snake coordinates (optional)
        
        # Metadata
        self.timestamp = datetime.now()
        self.session_id = str(uuid.uuid4())[:8]  # Short unique ID
        self.num_frames = 0
        
        # Processing parameters (stored for reproducibility)
        self.roi_params = None
        self.snake_params = None
        self.extraction_params = None
    
    def add_frame(self, frame, angle, snake_points=None):
        """
        Add a processed frame to the session
        
        Args:
            frame: Captured image (numpy array)
            angle: Extracted angle in degrees
            snake_points: Optimized snake coordinates (optional)
        """
        self.frames.append(frame)
        self.angles.append(angle)
        if snake_points is not None:
            self.snake_results.append(snake_points)
        self.num_frames += 1
    
    def get_statistics(self):
        """
        Calculate statistical measures of the angles
        
        Returns:
            dict: Statistics including mean, std, SEM, confidence intervals
        """
        if len(self.angles) == 0:
            return None
        
        angles_array = np.array(self.angles)
        n = len(angles_array)
        
        mean = np.mean(angles_array)
        std = np.std(angles_array, ddof=1)  # Sample std (N-1)
        sem = std / np.sqrt(n)  # Standard error of mean
        
        # 95% confidence interval (assuming normal distribution)
        # For small samples, should use t-distribution, but approximation for now
        ci_95 = 1.96 * sem
        
        return {
            'mean': mean,
            'std': std,
            'sem': sem,
            'ci_95_lower': mean - ci_95,
            'ci_95_upper': mean + ci_95,
            'min': np.min(angles_array),
            'max': np.max(angles_array),
            'median': np.median(angles_array),
            'n_frames': n
        }
    
    def get_angle_array(self):
        """Return angles as numpy array for easy manipulation"""
        return np.array(self.angles)
    
    def to_dict(self):
        """
        Export session data to dictionary (for JSON serialization)
        Note: Does not include frame images (too large)
        
        Returns:
            dict: Session metadata and angles
        """
        stats = self.get_statistics()
        
        return {
            'session_id': self.session_id,
            'distance': self.distance,
            'unit': self.unit,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'num_frames': self.num_frames,
            'angles': self.angles,
            'statistics': stats,
            'roi_params': self.roi_params,
            'snake_params': self.snake_params,
            'extraction_params': self.extraction_params
        }
    
    def save_to_json(self, filepath):
        """
        Save session data to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath):
        """
        Load session from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            CaptureSession: Reconstructed session (without frame images)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = cls(
            distance=data['distance'],
            unit=data['unit'],
            description=data['description']
        )
        
        session.session_id = data['session_id']
        session.timestamp = datetime.fromisoformat(data['timestamp'])
        session.angles = data['angles']
        session.num_frames = data['num_frames']
        session.roi_params = data.get('roi_params')
        session.snake_params = data.get('snake_params')
        session.extraction_params = data.get('extraction_params')
        
        return session
    
    def __repr__(self):
        return f"CaptureSession(id={self.session_id}, distance={self.distance}{self.unit}, frames={self.num_frames})"


class ExperimentData:
    """
    Container for entire experiment with multiple capture sessions
    Manages multiple distance measurements
    """
    
    def __init__(self, experiment_name=''):
        """
        Initialize experiment data container
        
        Args:
            experiment_name: Name/description of the experiment
        """
        self.experiment_name = experiment_name
        self.sessions = []  # List of CaptureSession objects
        self.created_at = datetime.now()
        self.experiment_id = str(uuid.uuid4())[:8]
        
        # Global parameters (applied to all sessions)
        self.global_roi_params = None
        self.global_snake_params = None
    
    def add_session(self, session):
        """
        Add a capture session to the experiment
        
        Args:
            session: CaptureSession object
        """
        self.sessions.append(session)
    
    def get_sessions_by_distance(self, distance, tolerance=1e-6):
        """
        Get all sessions at a specific distance
        
        Args:
            distance: Distance value to search for
            tolerance: Numerical tolerance for floating point comparison
            
        Returns:
            list: Sessions matching the distance
        """
        return [s for s in self.sessions if abs(s.distance - distance) < tolerance]
    
    def get_all_distances(self):
        """
        Get list of unique distances measured
        
        Returns:
            list: Sorted list of unique distances
        """
        distances = list(set(s.distance for s in self.sessions))
        return sorted(distances)
    
    def get_mean_angle_vs_distance(self):
        """
        Get mean angle for each distance (for plotting)
        
        Returns:
            tuple: (distances, mean_angles, std_errors)
        """
        distances = self.get_all_distances()
        mean_angles = []
        std_errors = []
        
        for d in distances:
            sessions = self.get_sessions_by_distance(d)
            # Combine all angles from all sessions at this distance
            all_angles = np.concatenate([s.get_angle_array() for s in sessions])
            mean_angles.append(np.mean(all_angles))
            std_errors.append(np.std(all_angles, ddof=1) / np.sqrt(len(all_angles)))
        
        return np.array(distances), np.array(mean_angles), np.array(std_errors)
    
    def to_dict(self):
        """
        Export experiment to dictionary
        
        Returns:
            dict: Complete experiment data
        """
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'created_at': self.created_at.isoformat(),
            'num_sessions': len(self.sessions),
            'distances': self.get_all_distances(),
            'global_roi_params': self.global_roi_params,
            'global_snake_params': self.global_snake_params,
            'sessions': [s.to_dict() for s in self.sessions]
        }
    
    def save_to_json(self, filepath):
        """Save entire experiment to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath):
        """Load experiment from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        experiment = cls(experiment_name=data['experiment_name'])
        experiment.experiment_id = data['experiment_id']
        experiment.created_at = datetime.fromisoformat(data['created_at'])
        experiment.global_roi_params = data.get('global_roi_params')
        experiment.global_snake_params = data.get('global_snake_params')
        
        # Reconstruct sessions
        for session_data in data['sessions']:
            # Use temporary file approach for reconstruction
            # (In practice, you'd pass the dict directly to a modified constructor)
            session = CaptureSession(
                distance=session_data['distance'],
                unit=session_data['unit'],
                description=session_data['description']
            )
            session.session_id = session_data['session_id']
            session.timestamp = datetime.fromisoformat(session_data['timestamp'])
            session.angles = session_data['angles']
            session.num_frames = session_data['num_frames']
            session.roi_params = session_data.get('roi_params')
            session.snake_params = session_data.get('snake_params')
            
            experiment.sessions.append(session)
        
        return experiment
    
    def __repr__(self):
        return f"ExperimentData(name='{self.experiment_name}', sessions={len(self.sessions)}, distances={self.get_all_distances()})"


class AngleExtractor:
    """
    Static utility methods for angle extraction from images
    These are helper functions used by the extraction pipeline
    """
    
    @staticmethod
    def find_brightest_points_in_margins(image, top_margin=10, bottom_margin=10):
        """
        Find brightest pixel in top and bottom margins of image
        Assumes line passes through entire image vertically
        
        Args:
            image: Grayscale image (2D numpy array)
            top_margin: Number of rows from top to search
            bottom_margin: Number of rows from bottom to search
            
        Returns:
            tuple: ((top_y, top_x), (bottom_y, bottom_x)) in pixel coordinates
        """
        h, w = image.shape
        
        # Top margin: find brightest pixel
        top_strip = image[:top_margin, :]
        top_idx = np.unravel_index(np.argmax(top_strip), top_strip.shape)
        top_point = (top_idx[0], top_idx[1])  # (y, x) or (row, col)
        
        # Bottom margin: find brightest pixel
        bottom_strip = image[-bottom_margin:, :]
        bottom_idx = np.unravel_index(np.argmax(bottom_strip), bottom_strip.shape)
        bottom_point = (h - bottom_margin + bottom_idx[0], bottom_idx[1])
        
        return top_point, bottom_point
    
    @staticmethod
    def points_to_angle(pt1, pt2):
        """
        Calculate angle from two points
        
        Args:
            pt1: (y, x) first point
            pt2: (y, x) second point
            
        Returns:
            float: Angle in degrees (from horizontal)
        """
        dy = pt2[0] - pt1[0]
        dx = pt2[1] - pt1[1]
        
        angle_rad = np.arctan2(dy, dx)
        angle_deg = angle_rad * 180 / np.pi
        
        return angle_deg
    
    @staticmethod
    def fit_line_to_points(points):
        """
        Fit a line to a set of points using least squares
        
        Args:
            points: Numpy array of shape (N, 2) with (x, y) coordinates
            
        Returns:
            tuple: (angle_deg, (vx, vy), (x0, y0))
                   angle: angle in degrees
                   (vx, vy): normalized direction vector
                   (x0, y0): point on the line
        """
        import cv2 as cv
        
        # OpenCV's fitLine uses robust fitting
        [vx, vy, x0, y0] = cv.fitLine(
            points.astype(np.float32),
            cv.DIST_L2,  # Least squares
            0,           # Not used for DIST_L2
            0.01,        # Accuracy for rho
            0.01         # Accuracy for theta
        )
        
        # Calculate angle from direction vector
        angle_rad = np.arctan2(vy[0], vx[0])
        angle_deg = angle_rad * 180 / np.pi
        
        return angle_deg, (vx[0], vy[0]), (x0[0], y0[0])
    
    @staticmethod
    def calculate_angle_statistics(angles):
        """
        Calculate statistics for a list of angles
        
        Args:
            angles: List or array of angles in degrees
            
        Returns:
            dict: Statistical measures
        """
        angles_array = np.array(angles)
        n = len(angles_array)
        
        if n == 0:
            return None
        
        mean = np.mean(angles_array)
        std = np.std(angles_array, ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        
        return {
            'mean': mean,
            'std': std,
            'sem': sem,
            'n': n,
            'min': np.min(angles_array),
            'max': np.max(angles_array),
            'range': np.max(angles_array) - np.min(angles_array)
        }
