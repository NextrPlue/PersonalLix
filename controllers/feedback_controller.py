from flask import Blueprint, request, jsonify, make_response
from worker import update_data
from extensions import task_queue  # extensions 모듈에서 task_queue 임포트

feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/feedback', methods=['POST'])
def updade_model_req():
    required_fields = ['gender', 'age', 'color', 'faceshape', 'bodyshape', 'clothes', 'rating']
    data = request.json
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return make_response(jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400)

    # 작업 큐에 업데이트 작업 추가
    job = task_queue.enqueue(update_data, data)  # app.task_queue 대신 task_queue 사용
    return make_response(jsonify({"info": f"Job submitted: {job.get_id()}"}), 200)
