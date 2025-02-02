from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

def find_camera(id):
    cameras = [0,2]
    return cameras[int(id)]
 

def gen_frames(camera_id):
     
    cam = find_camera(camera_id)
    cap=  cv2.VideoCapture(cam)
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_camera_ids', methods=['GET'])
def get_camera():
    # Logic to find available cameras
    # index = 0
    available_cameras = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if not cap.read()[0]:
                continue
            else:
                available_cameras.append(f"Camera {i}")
            cap.release()
        except:
            pass
    
    cameras = {"cameras": available_cameras}
    return jsonify(cameras)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
