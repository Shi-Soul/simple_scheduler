from flask import Blueprint, request, jsonify, render_template
from .scheduler import Scheduler
from .utils import format_tasks

main = Blueprint('main', __name__)
scheduler = Scheduler(servers=["yq-3080-1", "yq-3080-2"], gpus=8)

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/status', methods=['GET'])
def status():
    task_status = format_tasks(scheduler.get_tasks())
    gpu_status = scheduler.get_gpu_status()
    return jsonify({
        'tasks': task_status,
        'gpu_status': gpu_status
    })


@main.route('/submit', methods=['POST'])
def submit_task():
    data = request.json
    command = data.get('command')
    gpu_request = data.get('gpu_request')

    if not command or not gpu_request:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        gpu_request = float(gpu_request)
        assert 0.0 <= gpu_request and gpu_request<=1.0
    except ValueError:
        return jsonify({'error': 'Invalid GPU request format. It should be a number.'}), 400
    except AssertionError:
        return jsonify({'error': 'GPU request must be between 0.0 and 1.0.'}), 400

    task_id = scheduler.add_task(command, gpu_request)
    return jsonify({'id': task_id}), 201


@main.route('/cancel', methods=['POST'])
def cancel_task():
    task_id = request.json.get('id')
    try:
        task_id = int(task_id)
    except ValueError:
        return jsonify({'error': 'Invalid task ID format. It should be an integer.'}), 400
    
    if scheduler.cancel_task(task_id):
        return jsonify({'success': True, 'id':task_id}), 200
    else:
        return jsonify({'error': 'Task not found'}), 404
