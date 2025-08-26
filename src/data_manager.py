import os
import uuid
import pandas as pd
from flask import session, current_app
import shutil

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

    Args:
        df (pd.DataFrame): The DataFrame to save.
        session_key (str): The session key to store the file path (e.g., 'df_main_path').
        filename_prefix (str): A prefix for the generated filename.
    """
    data_folder = _get_data_folder()
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # If a file is already associated with this session key, delete it first.
    if session.get(session_key):
        try:
            os.remove(session[session_key])
        except (FileNotFoundError, OSError):
            pass  # Ignore if the file doesn't exist or other OS error

    filename = _generate_unique_filename(prefix=filename_prefix)
    file_path = os.path.join(data_folder, filename)
    
    try:
        df.to_hdf(file_path, key='df', mode='w')
        session[session_key] = file_path
    except Exception as e:
        # Handle potential errors during file save (e.g., disk full)
        print(f"Error saving dataframe to {file_path}: {e}")
        if session.get(session_key):
            del session[session_key]


def load_dataframe(session_key):
    """
    Loads a Pandas DataFrame from the path stored in the session.

    Args:
        session_key (str): The session key where the file path is stored.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if the path is not found
                               or the file doesn't exist.
    """
    file_path = session.get(session_key)
    if not file_path:
        return None
    
    try:
        df = pd.read_hdf(file_path, 'df')
        return df
    except FileNotFoundError:
        print(f"Data file not found: {file_path}")
        # Clean up the stale session key
        del session[session_key]
        return None
    except Exception as e:
        print(f"Error loading dataframe from {file_path}: {e}")
        return None

def delete_dataframe(session_key):
    """
    Deletes the data file associated with a session key and removes the key.
    """
    file_path = session.get(session_key)
    if file_path:
        try:
            os.remove(file_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Error deleting data file {file_path}: {e}")
        finally:
            if session_key in session:
                del session[session_key]


def delete_all_session_dataframes():
    """
    Deletes all data files stored in the current session and clears their keys.
    Also deletes the session's dedicated folder.
    """
    # Delete individual dataframe files and clear session keys
    dataframe_keys = [
        'df_main_path', 
        'df_metadata_path', 
        'df_meta_thd_path',
        'differential_analysis_results_path'
    ]
    
    for key in dataframe_keys:
        delete_dataframe(key)

    # Special handling for df_history which is a list of keys
    history_keys = session.get('df_history_paths', [])
    if history_keys:
        for key in history_keys:
            delete_dataframe(key)
    
    if 'df_history_paths' in session:
        del session['df_history_paths']

    # Finally, delete the session's folder
    session_folder = _get_data_folder()
    if os.path.exists(session_folder):
        try:
            shutil.rmtree(session_folder)
            print(f"Deleted session folder: {session_folder}")
        except OSError as e:
            print(f"Error deleting session folder {session_folder}: {e}")