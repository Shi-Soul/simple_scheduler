import threading
import subprocess
from queue import Queue

class Scheduler:
    def __init__(self, servers, gpus):
        self.servers = servers
        self.gpus = gpus
        self.tasks = []
        self.task_queue = Queue()
        self.task_counter = 0
        self.lock = threading.Lock()
        self.gpu_usage = {server: [0.0] * gpus for server in servers}  # Fractional usage

        # Start a thread to monitor task completion
        self.monitor_thread = threading.Thread(target=self.monitor_tasks)
        self.monitor_thread.start()

    def add_task(self, command, gpu_request):
        with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
            task = {
                'id': task_id,
                'command': command,
                'gpu_request': float(gpu_request),  # Convert to float
                'status': 'queued',
                'server': None,
                'gpu': None
            }
            self.tasks.append(task)
            self.task_queue.put(task)
            return task_id

    def cancel_task(self, task_id):
        with self.lock:
            for task in self.tasks:
                if task['id'] == task_id and task['status'] == 'running':
                    # Terminate the process if running
                    # Code to terminate the process
                    task['status'] = 'cancelled'
                    return True
                elif task['id'] == task_id and task['status'] == 'queued':
                    task['status'] = 'cancelled'
                    return True
        return False

    def monitor_tasks(self):
        while True:
            task = self.task_queue.get()
            if task['status'] == 'queued':
                self.run_task(task)
            self.task_queue.task_done()

    def run_task(self, task):
        with self.lock:
            # Find an available server and GPU
            for server in self.servers:
                for gpu in range(self.gpus):
                    if self.gpu_usage[server][gpu] + task['gpu_request'] <= 1.0:
                        self.gpu_usage[server][gpu] += task['gpu_request']
                        task['status'] = 'running'
                        task['server'] = server
                        task['gpu'] = gpu
                        break
                if task['status'] == 'running':
                    break

        if task['status'] == 'running':
            threading.Thread(target=self.execute_task, args=(task,)).start()
        else:
            # Requeue the task if no GPU is available
            task['status'] = 'queued'
            self.task_queue.put(task)

    def execute_task(self, task):
        command = f"ssh {task['server']} 'CUDA_VISIBLE_DEVICES={task['gpu']} {task['command']}'"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        with self.lock:
            task['status'] = 'completed'
            self.gpu_usage[task['server']][task['gpu']] -= task['gpu_request']

    def get_tasks(self):
        with self.lock:
            return self.tasks
