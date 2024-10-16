from flask import Flask
from flask_session import Session
from controllers.main_controller import main_bp
from controllers.recommendation_controller import recommendation_bp
from controllers.feedback_controller import feedback_bp
from datetime import timedelta
import redis
import os
from rq import Queue

'''
class BodyShape(Enum):
    HOURGLASS=0
    TRAPEZOID=1
    ROUND=2
    RECTANGLE=3
    INVERTED_TRIANGLE=4
    TRIANGLE=5

age: 20,30,40,50,60
gender: man, woman

faceshape = ['heart','oblong','oval','round','square']
color = ['spring', 'summer', 'autumn', 'winter']
'''

app = Flask(__name__)
app.secret_key = 'asdf92($(*()))8u983ij9s8eduf98s'
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=10)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'flask_session')
app.config['SESSION_PERMANENT'] = False  # Set to True if you want persistent sessions
app.config['SESSION_USE_SIGNER'] = True  # Adds an extra layer of security

redis_conn = redis.Redis()
task_queue = Queue(connection=redis_conn)

app.register_blueprint(main_bp)
app.register_blueprint(recommendation_bp)
app.register_blueprint(feedback_bp)

Session(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
