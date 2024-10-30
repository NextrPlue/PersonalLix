from rq import Worker, Queue
from rq.connections import Connection
from models.recommender import update_model
from extensions import redis_conn

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
