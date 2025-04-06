# database.py
import sqlite3
import json
import os
from datetime import datetime

class CaptchaDatabase:
    def __init__(self, db_path='captcha_data.db'):
        """Initialize the database connection"""
        self.db_path = db_path
        self.conn = None
        self.initialize_db()
    
    def initialize_db(self):
        """Create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            user_agent TEXT,
            session_duration INTEGER,
            timestamp TEXT,
            prediction REAL,
            is_bot INTEGER,
            challenge_required INTEGER,
            challenge_passed INTEGER,
            raw_data TEXT
        )
        ''')
        
        # Create features table for storing extracted features
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            feature_name TEXT,
            feature_value REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        # Create labels table for manual labeling
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            label INTEGER,
            labeler TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        self.conn.commit()
    
    def store_session(self, data, prediction=None):
        """Store a session in the database"""
        if self.conn is None:
            self.initialize_db()
        
        cursor = self.conn.cursor()
        
        # Extract metadata
        metadata = data.get('metadata', {})
        ip_address = metadata.get('ip', 'unknown')
        user_agent = metadata.get('user_agent', 'unknown')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        
        # Session info
        session_info = data.get('sessionInfo', {})
        session_duration = session_info.get('duration', 0)
        
        # Prediction info (if available)
        is_bot = None
        challenge_required = None
        if prediction:
            is_bot = 1 if prediction.get('is_bot', False) else 0
            challenge_required = 1 if prediction.get('challenge_required', False) else 0
        
        # Store raw data as JSON
        raw_data = json.dumps(data)
        
        # Insert session record
        cursor.execute('''
        INSERT INTO sessions (
            ip_address, user_agent, session_duration, timestamp, 
            prediction, is_bot, challenge_required, challenge_passed, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ip_address, user_agent, session_duration, timestamp,
            prediction.get('confidence', None) if prediction else None,
            is_bot, challenge_required, None, raw_data
        ))
        
        session_id = cursor.lastrowid
        
        # Store extracted features
        features = data.get('extractedFeatures', {})
        if features:
            feature_list = []
            for name, value in features.items():
                if isinstance(value, (int, float)):
                    feature_list.append((session_id, name, value))
            
            cursor.executemany('''
            INSERT INTO features (session_id, feature_name, feature_value)
            VALUES (?, ?, ?)
            ''', feature_list)
        
        self.conn.commit()
        return session_id
    
    def update_challenge_result(self, session_id, passed):
        """Update whether a challenge was passed"""
        if self.conn is None:
            self.initialize_db()
            
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE sessions SET challenge_passed = ? WHERE id = ?
        ''', (1 if passed else 0, session_id))
        self.conn.commit()
    
    def add_label(self, session_id, label, labeler="manual"):
        """Add a manual label for a session"""
        if self.conn is None:
            self.initialize_db()
            
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO labels (session_id, label, labeler, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (session_id, 1 if label else 0, labeler, datetime.now().isoformat()))
        self.conn.commit()
    
    def export_dataset(self, output_path):
        """Export a labeled dataset for training"""
        if self.conn is None:
            self.initialize_db()
            
        cursor = self.conn.cursor()
        
        # Get sessions with labels
        cursor.execute('''
        SELECT s.id, s.raw_data, l.label
        FROM sessions s
        JOIN labels l ON s.id = l.session_id
        ''')
        
        labeled_sessions = cursor.fetchall()
        
        all_features = []
        for session_id, raw_data, label in labeled_sessions:
            # Parse raw data
            data = json.loads(raw_data)
            
            # Get features for this session
            cursor.execute('''
            SELECT feature_name, feature_value
            FROM features
            WHERE session_id = ?
            ''', (session_id,))
            
            features = dict(cursor.fetchall())
            features['label'] = label
            
            all_features.append(features)
        
        # Convert to DataFrame and save
        import pandas as pd
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
        
        return df
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None