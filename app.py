from flask import Flask
from controllers.main_controller import main_bp
from controllers.recommendation_controller import recommendation_bp
from controllers.feedback_controller import feedback_bp
from datetime import timedelta
import redis
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

redis_conn = redis.Redis()
task_queue = Queue(connection=redis_conn)

app.register_blueprint(main_bp)
app.register_blueprint(recommendation_bp)
app.register_blueprint(feedback_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
