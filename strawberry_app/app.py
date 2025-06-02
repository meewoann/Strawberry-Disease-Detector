from flask import Flask, render_template, request
import cv2
import numpy as np
from fmodule.fruit_detector import FruitRipenessDetector
from fmodule.strawberry_disease_detector import StrawberryDiseaseDetector

app = Flask(__name__)
fruit_detector = FruitRipenessDetector()
disease_detector = StrawberryDiseaseDetector()

@app.route('/', methods=['GET', 'POST'])
def index():
    image_result = None
    crops_fruit = []
    crops_disease = []

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


            image_result, crops_fruit = fruit_detector.process_image(image)
            _, crops_disease = disease_detector.process_image(image)

    return render_template('index.html', image_result=image_result,
                           crops_fruit=crops_fruit, crops_disease=crops_disease)

if __name__ == '__main__':
    app.run(debug=True)
