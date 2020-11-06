import sys
import cv2, pickle
import numpy as np
from google.cloud import automl_v1beta1
#from google.cloud.automl_v1beta1.proto import service_pb2
import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/purvajay/Downloads/fast-ability-294222-ee8724eb3793.json"

image_x, image_y = 50, 50


# 'content' is base-64-encoded image data.
def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}

  requestObj = automl_v1beta1.PredictRequest(
    name=name,
    payload=payload
)

  request = prediction_client.predict(requestObj)
  return request  # waits till request is returned

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300


def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def predictImage(file_path):

  with open(file_path, 'rb') as ff:
    content = ff.read()

  response = get_prediction(content, '814473355030' , 'ICN7193781324119801856')

  print("Prediction results:")
  print(response)
  for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))
    return result.display_name
  return ""


def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0
    total_pics = 100
    flag_start_capturing = False
    pic_no = 0
    frames = 0
    create_folder("ABCD")
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh,thresh,thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        old_text = text


        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 30:

                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                if w1 > h1:

                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:

                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("ABCD/"+str(pic_no)+".jpg", save_img)
                text = predictImage("ABCD/"+str(pic_no)+".jpg")
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 2:
                    word = word + text

            elif cv2.contourArea(contour) < 1000:

                text = ""
                word = ""
        else:
            text = ""
            word = ""

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word , (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))

        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)

        frames += 1
        keypress = cv2.waitKey(1)
        if keypress == ord('v') and is_voice_on:
            is_voice_on = False
        elif keypress == ord('v') and not is_voice_on:
            is_voice_on = True
        print("Pic no ", pic_no)


def recognize():
        cam = cv2.VideoCapture(1)

        if cam.read()[0]==False:
            cam = cv2.VideoCapture(0)

if __name__ == '__main__':


  recognize()







