function submitTask() {
    const command = document.getElementById('command').value;
    const gpu_request = document.getElementById('gpu_request').value;

    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ command, gpu_request }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.id !== undefined) {
            alert(`Task submitted with ID: ${data.id}`);
            refreshStatus();
        } else {
            alert('Failed to submit task: ' + data.error);
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function cancelTask() {
    const taskId = document.getElementById('cancel_id').value;

    fetch('/cancel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id: taskId }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Task with ID ${data.id} has been canceled.`);
            refreshStatus();
        } else {
            alert('Failed to cancel task');
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function refreshStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        const statusElement = document.getElementById('status');
        statusElement.innerHTML = data;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
