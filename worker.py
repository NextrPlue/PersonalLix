import redis
from rq import Worker, Queue, Connection
from models.recommender import update_model

redis_conn = redis.Redis()

def update_data(data):
    update_model(
        data['gender'],
        data['age'],
        data['color'],
        data['faceshape'],
        data['bodyshape'],
        data['clothes'],
        int(data['rating'])
    )

if __name__ == '__main__':
    with Connection(redis_conn):
        q = Queue()
        worker = Worker(q)
        worker.work()
