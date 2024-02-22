from flask import Flask, render_template, request
import subprocess
from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('result.html', result="No image uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', result="No image selected.")

    file_path = 'static/' + file.filename
    file.save(file_path)
    result = run_test_script(file_path)
    return render_template('result.html', result=result)

def run_test_script(image_path):
    # Run test.py script with the image path as argument
    process = subprocess.Popen(['python', 'test.py', image_path], stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)