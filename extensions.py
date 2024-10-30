from rq import Queue
import redis

redis_conn = redis.Redis()
task_queue = Queue(connection=redis_conn)
