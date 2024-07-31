from typing import List, Dict

def format_tasks(tasks: List[Dict]) -> str:
    tasks = sorted(tasks, key=lambda x: x['id'], reverse=True)

    # Create HTML table
    html = '''
    <table border="1" cellpadding="5" cellspacing="0" style="width:100%; border-collapse:collapse;">
        <thead>
            <tr>
                <th>ID</th>
                <th>Status</th>
                <th>Command</th>
                <th>GPU Request</th>
                <th>Server</th>
                <th>GPU</th>
            </tr>
        </thead>
        <tbody>
    '''

    # Add rows to the table with colored status
    for task in tasks:
        status_color = {
            "running": "blue",
            "completed": "green",
            "cancelled": "red",
            "queued": "purple",
        }.get(task["status"], "orange")

        html += f'''
        <tr>
            <td>{task["id"]}</td>
            <td style="color:{status_color};">{task["status"]}</td>
            <td>{task["command"]}</td>
            <td>{task["gpu_request"]}</td>
            <td>{task["server"]}</td>
            <td>{task["gpu"]}</td>
        </tr>
        '''

    html += '''
        </tbody>
    </table>
    '''
    
    return html
