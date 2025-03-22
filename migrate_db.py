#!/usr/bin/env python3
"""
Database Migration Script for Document Retrieval App

This script helps migrate the database schema when changes are made.
It creates a backup of the current database and then applies migrations.
"""

import os
import sqlite3
import json
import shutil
import time
from datetime import datetime

DB_PATH = './threads.db'
BACKUP_DIR = './backups'

def backup_database():
    """Create a backup of the database file"""
    if os.path.exists(DB_PATH):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"threads_backup_{timestamp}.db")
        
        shutil.copy2(DB_PATH, backup_path)
        print(f"Database backup created at: {backup_path}")
        return backup_path
    return None

def migrate_database():
    """Apply migrations to the database"""
    if not os.path.exists(DB_PATH):
        print(f"Database file not found: {DB_PATH}")
        return False
    
    # Create a backup first
    backup_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current schema
    cursor.execute("PRAGMA table_info(threads)")
    columns = [column[1] for column in cursor.fetchall()]
    print(f"Current columns: {columns}")
    
    # Apply migrations
    migrations_applied = False
    
    # Migration 1: Add model column
    if 'model' not in columns:
        print("Applying migration: Adding 'model' column")
        cursor.execute('ALTER TABLE threads ADD COLUMN model TEXT')
        migrations_applied = True
    
    # Migration 2: Add documents column
    if 'documents' not in columns:
        print("Applying migration: Adding 'documents' column")
        cursor.execute('ALTER TABLE threads ADD COLUMN documents TEXT')
        migrations_applied = True
    
    # Migration 3: Convert messages from string to JSON
    if migrations_applied:
        print("Converting message format...")
        cursor.execute('SELECT id, messages FROM threads')
        rows = cursor.fetchall()
        
        for row_id, messages in rows:
            if messages and isinstance(messages, str):
                try:
                    # Try to parse as JSON first
                    json.loads(messages)
                    # If it's already valid JSON, no need to convert
                except json.JSONDecodeError:
                    try:
                        # Try to convert from Python string representation to JSON
                        messages_list = eval(messages)
                        json_messages = json.dumps(messages_list)
                        cursor.execute('UPDATE threads SET messages = ? WHERE id = ?', 
                                      (json_messages, row_id))
                        print(f"Converted messages for thread ID {row_id}")
                    except Exception as e:
                        print(f"Error converting messages for thread ID {row_id}: {str(e)}")
    
    conn.commit()
    conn.close()
    
    if migrations_applied:
        print("Database migration completed successfully")
    else:
        print("No migrations needed")
    
    return migrations_applied

if __name__ == "__main__":
    print("Starting database migration...")
    migrate_database()
    print("Migration process completed") 