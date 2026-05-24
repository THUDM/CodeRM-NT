import docker
import time
import traceback
import threading
import argparse
from flask import Flask, request, jsonify

client = docker.from_env()
image_name = 'python:3.11'

app = Flask(__name__)
semaphore = None
# semaphore = threading.Semaphore(4)

@app.route('/', methods=['POST'])
def generate_code():
    if not semaphore.acquire(blocking=False):
        return jsonify({"error": "Server is busy. Please try again later."}), 503
    
    data = request.get_json()
    code = data.get("code")
    timeout = data.get("timeout", 60)
    print(len(code))
    container = None
    try:
        container = client.containers.run(image_name, command=['python', '-c', code], detach=True)
        time_exceed = False
        start_time = time.time()
        while container.status != 'exited':
            container.reload()
            if time.time() - start_time > timeout:
                container.kill()
                time_exceed = True
                break
            time.sleep(timeout / 5)
        exit_status = container.wait()
        stdout = container.logs(stderr=False).decode()
        stderr = container.logs(stdout=False).decode()
        log = container.logs().decode()
        if time_exceed:
            stderr = stderr + '\nTimeout' if stderr else 'Timeout'
            log = log + '\nTimeout' if log else 'Timeout'
        print(log)
        return jsonify({
            "stdout": stdout,
            "stderr": stderr,
            "log": log,
            "exit_status": exit_status,
        }), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 503
    finally:
        semaphore.release()
        if container:
            container.remove()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    semaphore = threading.Semaphore(args.num_workers)
    app.run(host='0.0.0.0', port=args.port)
