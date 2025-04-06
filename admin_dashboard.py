from flask import Flask, jsonify, render_template
from database import CaptchaDatabase
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
db = CaptchaDatabase()

@app.route('/admin/sessions', methods=['GET'])
def get_sessions():
    """Fetch user sessions from the database."""
    sessions = db.get_all_sessions()  # Assuming this method exists
    return render_template('admin_sessions.html', sessions=sessions)

@app.route('/admin/visualization', methods=['GET'])
def visualize_data():
    """Generate a simple visualization of user behavior."""
    # Example data
    labels = ['Bots', 'Humans']
    sizes = [db.get_bot_count(), db.get_human_count()]  # Assuming these methods exist

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)

    return render_template('visualization.html', image=image_base64)

@app.route('/admin/logs', methods=['GET'])
def get_logs():
    """Fetch and display logs."""
    log_file_path = 'path/to/your/logfile.log'  # Update with the actual log file path
    with open(log_file_path, 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)
