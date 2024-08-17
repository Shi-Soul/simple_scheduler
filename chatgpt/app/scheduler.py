import threading
import subprocess
import time
from queue import Queue, Empty

class Scheduler:
    def __init__(self, servers, gpus):
        self.servers = servers
        self.gpus = gpus
        self.tasks = []
        self.task_queue = Queue()
        self.task_counter = 0
        self.lock = threading.Lock()
        # Use floating-point values to track GPU usage
        self.gpu_usage = {server: [0.0] * gpus for server in servers}
        self.processes = {}  # To keep track of running processes

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
                'gpu_request': gpu_request,
                'status': 'queued',
                'server': None,
                'gpu': None,
                'process': None
            }
            self.tasks.append(task)
            self.task_queue.put(task)
            return task_id

    def cancel_task(self, task_id):
        print("Cancel Task  ", task_id)
        task_id = int(task_id)
        with self.lock:
            for task in self.tasks:
                if task['id'] == task_id:
                    if task['status'] == 'running':
                        # Attempt to terminate the running process
                        process = self.processes.get(task_id)
                        if process:
                            process.terminate()
                            process.wait()  # Ensure the process is terminated
                            self.processes.pop(task_id, None)
                            task['status'] = 'cancelled'
                            return True
                    elif task['status'] == 'queued':
                        task['status'] = 'cancelled'
                        # Remove the task from the queue
                        self.remove_task_from_queue(task_id)
                        return True
        return False

    def remove_task_from_queue(self, task_id):
        temp_queue = Queue()
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task['id'] != task_id:
                temp_queue.put(task)
        self.task_queue = temp_queue


    def monitor_tasks(self):
        while True:
            try:
                task = self.task_queue.get(timeout=1)
            except Empty:
                continue
            if task['status'] == 'queued':
                self.run_task(task)
            self.task_queue.task_done()

    def run_task(self, task):
        with self.lock:
            # Find an available server and GPU with enough capacity
            for server in self.servers:
                for gpu in range(self.gpus):
                    if self.gpu_usage[server][gpu] + task['gpu_request'] <= 1.0:  # 1.0 means 100% of the GPU
                        self.gpu_usage[server][gpu] += task['gpu_request']
                        task['status'] = 'running'
                        task['server'] = server
                        task['gpu'] = gpu
                        break
                if task['status'] == 'running':
                    break

        if task['status'] == 'running':
            # Run the task on the selected server and GPU
            task['process'] = threading.Thread(target=self.execute_task, args=(task,))
            task['process'].start()
        else:
            # Requeue the task if no GPU is available
            task['status'] = 'queued'
            self.task_queue.put(task)

    def execute_task(self, task):
        # command = f" {task['command']}"
        command = f"ssh {task['server']} 'export CUDA_VISIBLE_DEVICES={task['gpu']}; {task['command']}' &> /dev/null"
        print(f"DEBUG: scheduler execute_task {command=}")
        process = subprocess.Popen(command, shell=True)
        
        # Store the process in the task for potential cancellation
        with self.lock:
            self.processes[task['id']] = process

        process.wait()

        # Get the task completion status
        ret = process.returncode

        with self.lock:
            if task['status'] != 'cancelled':
                if ret != 0:
                    task['status'] = 'failed'
                else:
                    task['status'] = 'completed'
                
            # Free up GPU resources
            self.gpu_usage[task['server']][task['gpu']] -= task['gpu_request']
            self.processes.pop(task['id'], None)

    def get_tasks(self):
        with self.lock:
            return self.tasks

    def get_gpu_status(self):
        with self.lock:
            status = {}
            for server in self.servers:
                status[server] = []
                for gpu in range(self.gpus):
                    usage = self.gpu_usage[server][gpu]
                    status[server].append({
                        'gpu': gpu,
                        'allocated': usage,
                        'free': usage==0.0
                    })
            return status