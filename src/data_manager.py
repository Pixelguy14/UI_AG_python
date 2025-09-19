import os
import uuid
import pandas as pd
from flask import session, current_app
import shutil
import threading

# Global lock for all data processing operations
_processing_lock = threading.RLock()

# Dictionary to hold a lock for each file path, plus a lock for the dictionary itself
_file_locks = {}
_lock_for_locks = threading.Lock()

def processing_lock():
    """A context manager for the global data processing lock."""
    return _processing_lock

def _get_lock(file_path):
    """Gets the lock for a specific file path, creating one if it doesn't exist."""
    with _lock_for_locks:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]

def _get_data_folder():
    """Gets the absolute path to the data storage folder for the current session."""
    session_id = session.get('session_id')
    if not session_id:
        raise RuntimeError("Session ID not found. Ensure session is initialized.")
    return os.path.abspath(os.path.join(current_app.config['UPLOAD_FOLDER'], session_id))

def _generate_unique_filename(prefix='data', suffix='.h5'):
    """Generates a unique filename to prevent collisions."""
    return f"{prefix}_{uuid.uuid4()}{suffix}"

def save_dataframe(df, session_key, filename_prefix='df'):
    """
    Saves a Pandas DataFrame to an HDF5 file and stores the path in the session.
    This function is thread-safe.
    """
    data_folder = _get_data_folder()
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # If a file is already associated with this session key, delete it first.
    if session.get(session_key):
        delete_dataframe(session[session_key]) # Use the locking delete function

    filename = _generate_unique_filename(prefix=filename_prefix)
    file_path = os.path.join(data_folder, filename)
    file_lock = _get_lock(file_path)

    with file_lock:
        try:
            df.to_hdf(file_path, key='df', mode='w')
            session[session_key] = file_path
        except Exception as e:
            print(f"Error saving dataframe to {file_path}: {e}")
            if session.get(session_key):
                del session[session_key]

def load_dataframe(session_key):
    """
    Loads a Pandas DataFrame from the path stored in the session.
    This function is thread-safe.
    """
    file_path = session.get(session_key)
    if not file_path:
        return None
    
    file_lock = _get_lock(file_path)
    with file_lock:
        try:
            df = pd.read_hdf(file_path, 'df')
            return df
        except FileNotFoundError:
            print(f"Data file not found: {file_path}")
            if session_key in session:
                del session[session_key]
            return None
        except Exception as e:
            print(f"Error loading dataframe from {file_path}: {e}")
            return None

def delete_dataframe(session_key_or_path):
    """
    Deletes the data file associated with a session key or a direct path.
    This function is thread-safe.
    """
    if os.path.isabs(session_key_or_path):
        file_path = session_key_or_path
        session_key_to_clear = None
        for key, value in session.items():
            if isinstance(value, str) and value == file_path:
                session_key_to_clear = key
                break
    else:
        file_path = session.get(session_key_or_path)
        session_key_to_clear = session_key_or_path

    if file_path:
        file_lock = _get_lock(file_path)
        with file_lock:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except (FileNotFoundError, OSError) as e:
                print(f"Error deleting data file {file_path}: {e}")
            finally:
                if session_key_to_clear and session_key_to_clear in session:
                    del session[session_key_to_clear]
                # Clean up the lock from the dictionary
                with _lock_for_locks:
                    if file_path in _file_locks:
                        del _file_locks[file_path]

def delete_all_session_dataframes():
    """
    Deletes all data files stored in the current session and clears their keys.
    Also deletes the session's dedicated folder.
    """
    dataframe_keys = [
        'df_main_path', 
        'df_metadata_path', 
        'df_meta_thd_path',
        'differential_analysis_results_path'
    ]
    
    for key in dataframe_keys:
        delete_dataframe(key)

    history_keys = session.get('df_history_paths', [])
    if history_keys:
        # Create a copy for iteration as delete_dataframe will modify the session
        for key in list(history_keys):
            # history_keys contains session keys, not paths
            delete_dataframe(key)
    
    if 'df_history_paths' in session:
        del session['df_history_paths']

    try:
        session_folder = _get_data_folder()
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)
            print(f"Deleted session folder: {session_folder}")
    except Exception as e:
        print(f"Error deleting session folder: {e}")
