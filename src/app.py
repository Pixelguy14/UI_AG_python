import os
import sys
import uuid
from flask import Flask, session, g
from flask_session import Session # type: ignore
from config import Config
# Import and register blueprints
from .routes import core, processing, analysis, api # type: ignore

def create_app():
    # Adjust template and static folder paths to be relative to the `src` directory
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(Config)

    # Initialize extensions
    Session(app)

    # Ensure instance folders exist
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

    app.register_blueprint(core.core_bp)
    app.register_blueprint(processing.processing_bp)
    app.register_blueprint(analysis.analysis_bp)
    app.register_blueprint(api.api_bp, url_prefix='/api')

    # This decorator will apply to all requests for the app
    @app.before_request
    def before_request():
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())

        # Calculate session size
        session_size = 0
        g.session_items = {}
        for key, value in session.items():
            item_size = sys.getsizeof(value)
            session_size += item_size
            g.session_items[key] = f'{item_size / 1024:.2f} KB'
        g.session_size_mb = session_size / (1024 * 1024)

        # Ensure session variables are initialized
        if 'df_main_path' not in session:
            session['df_main_path'] = None
        if 'df_metadata_path' not in session:
            session['df_metadata_path'] = None
        if 'df_meta_thd_path' not in session:
            session['df_meta_thd_path'] = None
        if 'df_history_paths' not in session:
            session['df_history_paths'] = []
        if 'imputed_mask' not in session:
            session['imputed_mask'] = None
        if 'imputation_performed' not in session:
            session['imputation_performed'] = False
        if 'group_assignments' not in session:
            session['group_assignments'] = {}
        if 'group_names' not in session:
            session['group_names'] = {}
        if 'n_groups' not in session:
            session['n_groups'] = 0
        if 'group_vector' not in session:
            session['group_vector'] = {}
        if 'processing_steps' not in session:
            session['processing_steps'] = []
        if 'step_transformation' not in session:
            session['step_transformation'] = []
        if 'step_scaling' not in session:
            session['step_scaling'] = []
        if 'step_normalization' not in session:
            session['step_normalization'] = []
        if 'differential_analysis_results_path' not in session:
            session['differential_analysis_results_path'] = None
        if 'latest_differential_analysis_method' not in session:
            session['latest_differential_analysis_method'] = None
        if 'group_regexes' not in session:
            session['group_regexes'] = {}

    return app
