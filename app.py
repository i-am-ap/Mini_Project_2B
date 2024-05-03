from flask import Flask
from views import views
import subprocess

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")

# @app.route("/")
# def Home():
#     return "Hello, World!"

@app.route('/run_Aircanvas')
def run_Aircanvas():
    # Run the Python script for Project 1
    subprocess.Popen(['python', 'Aircanvas.py'])
    return 'Project 1 is running.'

@app.route('/run_letter')
def run_letter():
    # Run the Python script for Project 1
    subprocess.Popen(['python', 'app copy.py'])
    return 'Project 2 is running.'

@app.route('/run_drawing')
def run_drawing():
    # Run the Python script for Project 1
    subprocess.Popen(['python', 'camera_app.py'])
    return 'Project 3 is running.'

if __name__ == '__main__':
    app.run(debug=True, port=8000)