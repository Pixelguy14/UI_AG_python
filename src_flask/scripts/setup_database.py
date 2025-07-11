"""
Optional: Setup script for database if you want to store session data persistently
"""
import sqlite3
import os

def create_database():
    """Create SQLite database for storing session data"""
    db_path = 'omics_sessions.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data_key TEXT,
            data_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database created at {db_path}")

if __name__ == "__main__":
    create_database()
