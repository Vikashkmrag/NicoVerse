DB_PATH = './threads.db'
import sqlite3
from datetime import datetime
import json
import os
import shutil
import time
from modules.utils.debug import debug_print

def backup_database():
    """Create a backup of the database file"""
    if os.path.exists(DB_PATH):
        backup_dir = './backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"threads_backup_{timestamp}.db")
        
        shutil.copy2(DB_PATH, backup_path)
        print(f"Database backup created at: {backup_path}")
        return backup_path
    return None

def restore_database(backup_path):
    """Restore the database from a backup file"""
    if os.path.exists(backup_path):
        # Create a backup of the current database first
        if os.path.exists(DB_PATH):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            current_backup = os.path.join('./backups', f"threads_before_restore_{timestamp}.db")
            shutil.copy2(DB_PATH, current_backup)
        
        # Restore from backup
        shutil.copy2(backup_path, DB_PATH)
        print(f"Database restored from: {backup_path}")
        return True
    return False

# Initialize SQLite database for thread management
class ThreadDB:
    def __init__(self):
        self.init_db()
        self.migrate_db()

    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                messages TEXT NOT NULL
            )
        ''')
        
        # Create model_embeddings table to store embedding support information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_embeddings (
                model_name TEXT PRIMARY KEY,
                supports_embeddings INTEGER NOT NULL,
                last_checked TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def migrate_db(self):
        """Check if the database needs migration and update schema if needed"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if model column exists
        cursor.execute("PRAGMA table_info(threads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add missing columns if needed
        if 'model' not in columns:
            print("Migrating database: Adding 'model' column")
            cursor.execute('ALTER TABLE threads ADD COLUMN model TEXT')
        
        if 'documents' not in columns:
            print("Migrating database: Adding 'documents' column")
            cursor.execute('ALTER TABLE threads ADD COLUMN documents TEXT')
        
        # Check if model_embeddings table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_embeddings'")
        if not cursor.fetchone():
            print("Migrating database: Creating 'model_embeddings' table")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_embeddings (
                    model_name TEXT PRIMARY KEY,
                    supports_embeddings INTEGER NOT NULL,
                    last_checked TEXT NOT NULL
                )
            ''')
        
        conn.commit()
        conn.close()

    # Save a thread to the database
    def save_thread(self, name, messages, model=None, documents=None):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO threads (name, created_at, messages, model, documents) VALUES (?, ?, ?, ?, ?)',
                    (name, datetime.now().isoformat(), json.dumps(messages), model, json.dumps(documents) if documents else None))
        thread_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return thread_id

    # Update an existing thread in the database
    def update_thread(self, thread_id, name, messages, model=None, documents=None):
        debug_print("Updating thread: ID={}, Name={}", thread_id, name)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get current values
        cursor.execute('SELECT model, documents FROM threads WHERE id = ?', (thread_id,))
        result = cursor.fetchone()
        current_model, current_documents = result if result else (None, None)
        
        # Use new values or keep current ones
        model = model or current_model
        documents_json = json.dumps(documents) if documents else current_documents
        
        # Update the thread with the new name
        cursor.execute('UPDATE threads SET name = ?, messages = ?, model = ?, documents = ? WHERE id = ?',
                    (name, json.dumps(messages), model, documents_json, thread_id))
        conn.commit()
        conn.close()
        debug_print("Thread updated successfully: ID={}, Name={}", thread_id, name)
        return thread_id

    # Load all threads from the database
    def load_threads(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if model column exists
        cursor.execute("PRAGMA table_info(threads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'model' in columns:
            cursor.execute('SELECT id, name, created_at, model FROM threads ORDER BY created_at DESC')
        else:
            cursor.execute('SELECT id, name, created_at FROM threads ORDER BY created_at DESC')
            # Add None for missing model column in results
            threads = [(id, name, created_at, None) for id, name, created_at in cursor.fetchall()]
            conn.close()
            return threads
            
        threads = cursor.fetchall()
        conn.close()
        return threads

    # Load a specific thread's messages by ID
    def load_thread_messages(self, thread_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT messages FROM threads WHERE id = ?', (thread_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Handle both string and JSON formats for backward compatibility
            messages = result[0]
            if isinstance(messages, str):
                try:
                    return json.loads(messages)
                except json.JSONDecodeError:
                    # Handle old format (string representation of Python list)
                    try:
                        return eval(messages)
                    except:
                        return []
        return []
    
    # Load a specific thread's model by ID
    def load_thread_model(self, thread_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if model column exists
        cursor.execute("PRAGMA table_info(threads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'model' in columns:
            cursor.execute('SELECT model FROM threads WHERE id = ?', (thread_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result and result[0] else None
        else:
            conn.close()
            return None
    
    # Load a specific thread's documents by ID
    def load_thread_documents(self, thread_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if documents column exists
        cursor.execute("PRAGMA table_info(threads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'documents' in columns:
            cursor.execute('SELECT documents FROM threads WHERE id = ?', (thread_id,))
            result = cursor.fetchone()
            conn.close()
            return json.loads(result[0]) if result and result[0] else []
        else:
            conn.close()
            return []

    # Save or update embedding support information for a model
    def save_model_embedding_support(self, model_name, supports_embeddings):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if model already exists in the table
        cursor.execute('SELECT model_name FROM model_embeddings WHERE model_name = ?', (model_name,))
        result = cursor.fetchone()
        
        if result:
            # Update existing record
            cursor.execute(
                'UPDATE model_embeddings SET supports_embeddings = ?, last_checked = ? WHERE model_name = ?',
                (1 if supports_embeddings else 0, datetime.now().isoformat(), model_name)
            )
        else:
            # Insert new record
            cursor.execute(
                'INSERT INTO model_embeddings (model_name, supports_embeddings, last_checked) VALUES (?, ?, ?)',
                (model_name, 1 if supports_embeddings else 0, datetime.now().isoformat())
            )
        
        conn.commit()
        conn.close()
        return True
    
    # Get embedding support information for a model
    def get_model_embedding_support(self, model_name):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT supports_embeddings, last_checked FROM model_embeddings WHERE model_name = ?', (model_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'supports_embeddings': bool(result[0]),
                'last_checked': result[1]
            }
        return None
    
    # Get all models with embedding support information
    def get_all_model_embedding_support(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT model_name, supports_embeddings, last_checked FROM model_embeddings')
        results = cursor.fetchall()
        conn.close()
        
        models_info = {}
        for row in results:
            models_info[row[0]] = {
                'supports_embeddings': bool(row[1]),
                'last_checked': row[2]
            }
        
        return models_info
        
    # Get all threads from the database
    def get_all_threads(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, created_at FROM threads ORDER BY created_at DESC')
        thread_rows = cursor.fetchall()
        conn.close()
        
        threads = []
        for row in thread_rows:
            threads.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2]
            })
        
        return threads
    
    # Get a specific thread by ID
    def get_thread(self, thread_id):
        debug_print("Getting thread with ID: {}", thread_id)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, name, messages, model, documents FROM threads WHERE id = ?', (thread_id,))
            result = cursor.fetchone()
            
            if not result:
                debug_print("No thread found with ID: {}", thread_id)
                return None
                
            debug_print("Thread found: ID={}, Name={}", result[0], result[1])
            
            # Parse messages
            messages = []
            if result[2]:
                try:
                    messages = json.loads(result[2])
                    debug_print("Successfully parsed {} messages", len(messages))
                except json.JSONDecodeError as e:
                    debug_print("Error parsing messages JSON: {}", str(e))
                    messages = []
            
            thread = {
                'id': result[0],
                'name': result[1],
                'messages': messages,
                'model': result[3]
            }
            
            # Handle documents if available
            if result[4]:
                try:
                    thread['documents'] = json.loads(result[4])
                    debug_print("Successfully parsed {} documents", len(thread['documents']))
                except json.JSONDecodeError as e:
                    debug_print("Error parsing documents JSON: {}", str(e))
                    thread['documents'] = []
            else:
                thread['documents'] = []
                
            return thread
            
        except Exception as e:
            debug_print("Error getting thread: {}", str(e))
            return None
        finally:
            conn.close()
    
    # Delete a thread by ID
    def delete_thread(self, thread_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM threads WHERE id = ?', (thread_id,))
        conn.commit()
        conn.close()
        
        return True
