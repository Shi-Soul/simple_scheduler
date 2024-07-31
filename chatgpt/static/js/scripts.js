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
            alert('Failed to cancel task: '+ data.error);
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
        const gpuStatusElement = document.getElementById('gpu-status');

        // Display task status
        statusElement.innerHTML = data.tasks;

        // Format GPU status
        const gpuStatusHtml = Object.keys(data.gpu_status).map(server => {
            const serverStatus = data.gpu_status[server].map(gpu => {
                // Define the padding length for each column
                const gpuNumber = `GPU ${gpu.gpu}`.padEnd(8); // Adjust as needed
                const allocated = `Allocated ${(gpu.allocated * 100)}%`.padEnd(24); // Adjust as needed
                const free = `Free ${gpu.free}`.padEnd(12); // Adjust as needed

                return `${gpuNumber} ${allocated} ${free}`;
            }).join('\n');

            return `<h3>${server}</h3><pre style="font-family: monospace; white-space: pre;">${serverStatus}</pre>`;
        }).join('');

        // Display GPU status
        gpuStatusElement.innerHTML = gpuStatusHtml;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
