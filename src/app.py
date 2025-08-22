import os
from flask import Flask, session
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
        # Ensure session variables are initialized
        if 'df_main' not in session:
            session['df_main'] = None
        if 'df_metadata' not in session:
            session['df_metadata'] = None
        if 'df_meta_thd' not in session:
            session['df_meta_thd'] = None
        if 'df_history' not in session:
            session['df_history'] = []
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
        if 'differential_analysis_results' not in session:
            session['differential_analysis_results'] = None
        if 'latest_differential_analysis_method' not in session:
            session['latest_differential_analysis_method'] = None
        if 'group_regexes' not in session:
            session['group_regexes'] = {}

    return app
