from src.app import create_app

app = create_app()

if __name__ == '__main__':
    # The host is set to '0.0.0.0' to be accessible from outside the container
    # in a Dockerized environment. For local development, '127.0.0.1' is also fine.
    app.run(host='0.0.0.0', port=5000, debug=True)