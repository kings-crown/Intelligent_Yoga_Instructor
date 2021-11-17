import cv2
import time
import pyttsx3
import numpy as np
import tensorflow as tf
from pygame import mixer
from playsound import playsound
from tf_pose.estimator import TfPoseEstimator
from object_detection.utils import label_map_util
from tf_pose.networks import get_graph_path, model_wh
from object_detection.utils import visualization_utils as vis_util

resize = '320x240'
model = 'mobilenet_v2_large' # mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small

mountain = [{'id': 1, 'name': 'mountain'}]
lion = [{'id': 2, 'name': 'lion'}]
triangle = [{'id': 3, 'name': 'triangle'}]
tree = [{'id': 4, 'name': 'tree'}]
warrior = [{'id': 5, 'name': 'warrior'}]

def draw():
    
    PATH_TO_FROZEN_GRAPH = '/inference_final/frozen_inference_graph.pb'
    PATH_TO_LABEL_MAP = '/Pose_estimation/label_final.pbtxt'
    NUM_CLASSES = 5

    label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')        
    sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i=0
    j=0
    mixer.init()
    
    if __name__ == '__main__':
        
        w, h = model_wh(resize)
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        cam = cv2.VideoCapture(0)
        ret_val, image = cam.read()
        
        fps_time = 0
        #i=0
        
        while True:
            
            ret_val, image_np = cam.read()
            humans = e.inference(image_np, resize_to_default=(w > 0 and h > 0), upsample_size= 4.0)
            image_np = TfPoseEstimator.draw_humans(image_np, humans, imgcopy=False)


            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor:image_np_expanded})
            
           
            if (i==0):
                mixer.music.load('1.mp3')
                mixer.music.play()
                time.sleep(30)
                i=1
                continue   
                
            if (i==1):
                mixer.music.load('2.mp3')
                mixer.music.play()
                i=2
                continue 
           
            if (i==3):
                mixer.music.load('4.mp3')
                mixer.music.play()
                i=4
                continue 
                
            if (i==5):
                mixer.music.load('6.mp3')
                mixer.music.play()
                i=6
                continue  

            if (i==7):
                mixer.music.load('8.mp3')
                mixer.music.play()
                i=8
                continue 
            
            if (i==9):
                mixer.music.load('10.mp3')
                mixer.music.play()
                i=10
                continue 
                
                
            predict = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.97]
            predict1 = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.95]
            predict2= [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.985]
                       
            
            if (predict1 == tree) and (i==2):
                print(predict1)
                mixer.music.load('3.mp3')
                mixer.music.play()
                time.sleep(10)
                j=0
                i=3
                
                continue                   

            if (predict1 == warrior) and (i==4):
                print(predict1)
                mixer.music.load('5.mp3')
                mixer.music.play()
                time.sleep(10)
                j=0
                i=5
                continue
                
                
            if (predict1 == triangle)and (i==6):
                print(predict1)
                mixer.music.load('7.mp3')
                mixer.music.play()
                time.sleep(10)
                j=0
                i=7
                continue
                
            if (predict== mountain)and (i==8):
                print(predict)
                mixer.music.load('9.mp3')
                mixer.music.play()
                time.sleep(10)
                j=0
                i=9
                continue
                
            if (predict2==lion)and (i==10):
                print(predict2)
                mixer.music.load('11.mp3')
                mixer.music.play()
                time.sleep(10)
                j=0
                i=1
                
                continue    
                
            #continue
            if (j>120):
                i=i-1
                j=0
            cv2.putText(image_np,"FPS: %f" % (1.0 / (time.time() - fps_time))
                    ,(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)            
            
            cv2.imshow('tf-pose-estimation result', image_np)
            fps_time = time.time()

            j=j+1
            print(j)
            if cv2.waitKey(1) == 27:
                break
                
#cv2.destroyAllWindows()
draw()




















