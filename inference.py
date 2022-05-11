from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
# config = 'configs/resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py'
# base_cfg_path base_checkpoint
base_cfg_path = "./configs/resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py"
base_checkpoint = "./strikes_back/latest.pth"
from mmcv import Config
cfg = Config.fromfile(base_cfg_path)

cfg.model.roi_head.bbox_head.num_classes = 3
# Setup a checkpoint file to load
# checkpoint = 'new_dataset/latest.pth'
# initialize the detector
model = init_detector(cfg, base_checkpoint, device='cuda:0')

import sys
sys.path.insert(0, './KalmanFilter')

import cv2
import numpy as np
import random
FONT = cv2.FONT_HERSHEY_SIMPLEX
from mmdet.apis import inference_detector


from KalmanFilter.kalmanFilter import KalmanFilter


model.cfg = cfg
TTL = 60
STABLED = 55

def threshold(x):
    return x[4] >= 0.5

def draw_text(img, text,
          font=FONT,
          pos=(0, 0),
          font_scale=0.4,
          font_thickness=1,
          text_color=(220,220,220),
          text_color_bg=(0, 0, 0)
          ):
    pos = pos[0] + 2, pos[1] + 2
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 4, y + text_h + 4), text_color_bg, -1)
    cv2.putText(img, text, (x + 2, int(y + text_h + font_scale + 1)), font, font_scale, text_color, font_thickness)

    return text_size

def rectangle_area(tl_x, tl_y, br_x, br_y):
    return (br_x - tl_x) * (br_y - tl_y)

def iou(box1_tl, box1_br, box2_tl, box2_br):
    
    # print(box1_tl, box1_br)
    # print(box2_tl, box2_br)
    
    tl_x = max(box1_tl[0], box2_tl[0])
    tl_y = max(box1_tl[1], box2_tl[1])
    br_x = min(box1_br[0], box2_br[0])
    br_y = min(box1_br[1], box2_br[1])
    # print(tl_x, tl_y, br_x, br_y)
    
    
    # print(box1_area, box2_area, intersect_area)
    if tl_x < br_x and tl_y < br_y:
        intersect_area = rectangle_area(tl_x, tl_y, br_x, br_y)
    
        # box1_area = (box1_br[0] - box1_tl[0]) * (box1_br[1] - box1_tl[1])
        box1_area = rectangle_area(*box1_br, *box1_tl)
        # box2_area = (box2_br[0] - box2_tl[0]) * (box2_br[1] - box2_tl[1])
        box2_area = rectangle_area(*box2_br, *box2_tl)
        return intersect_area / (box1_area + box2_area - intersect_area)
    else:
        return 0
    # print(iou, intersect_area)
    # box1 = torch.tensor([[box1_tl[0], box1_tl[1], box2_tl[0], box2_br[1]]], dtype=torch.float)
    # box2 = torch.tensor([[544, 59, 610, 94]], dtype=torch.float)
    # iou = bops.box_iou(box1, box2)
    
    # return iou
    
    
OBJECT_ID = 0
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
# RANGES = ((150, 220), (10, 100), (10, 100)), ((100, 150), (100, 150), (100, 150)), ((10, 100), (150, 200), (10, 150))
# RANGES = ((239, 220), (10, 90), (10, 100)), ((100, 170), (100, 190), (100, 190)), ((10, 80), (110, 200), (10, 100))
# RANGES = ((83, 196), (83, 191), (238, 238)), ((238, 238), (114, 212), (83, 83)), ((83, 129), (238, 238), (83, 217))
RANGES = ((238, 239), (83, 192), (83, 197)), ((83, 84), (114, 213), (238, 239)), ((83, 218), (238, 239), (83, 130))

# POINT_COLORS = ((0, 255, 0), (255, 0, 0), (0, 0, 255))

# def new_object(colors, mid_pt):
def new_object(colors):
    global OBJECT_ID
    OBJECT_ID += 1
    kf = KalmanFilter(0, method="Accerelation")
    # print(kf)
    kf.predict()
    # kf.correct(np.array(mid_pt))
    return (random.randrange(*colors[0]), random.randrange(*colors[1]), random.randrange(*colors[2])), OBJECT_ID, kf
    
def mid_point(x, y):
    return (x+y)/2

def traslate(old_tl, old_br, predict_mid):
    
    # print(old_tl, old_br, predict_mid)
    width = old_br[0] - old_tl[0]
    width/=2
    height = old_br[1] - old_tl[1]
    height/=2
    x, y = predict_mid
    return (int(x - width), int(y - height)), (int(x + width), int(y + height))
    
        
def draw_rectangles(frame, bboxes, i, previous_bboxes, colors):
    class_name = CLASSES[i]
    
    new_bboxes = []
    centers = []
    
    visited_previous_index_set = set()
    prev2new = {}
    matched_box = [[] for _ in range(len(previous_bboxes))]
    mutilple_matched_index = []
    
    predicted_bboxes = [traslate(pb[0], pb[1], pb[5].predict()) for pb in previous_bboxes]
    
    bboxes_no = 0
    
    for bboxes_index, (bbox_x1, bbox_y1, bbox_x2, bbox_y2, score) in enumerate(bboxes):
        bboxes_no += 1
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
        
        matched_bbox_color, matched_id, matched_index = None, None, None
        threshold = 0.5
        for previous_index, p_box in enumerate(previous_bboxes):
            # current_iou = iou((bbox_x1, bbox_y1), (bbox_x2, bbox_y2), p_box[0], p_box[1])
            if p_box[6] < STABLED:
                pred_bboxes = predicted_bboxes[previous_index]
                current_iou = iou((bbox_x1, bbox_y1), (bbox_x2, bbox_y2), pred_bboxes[0], pred_bboxes[1])
                if current_iou > threshold:
                    threshold = current_iou
                    matched_bbox_color = p_box[3]
                    matched_id = p_box[4]
                    matched_index = previous_index
                    kf = p_box[5]
                
        if threshold == 0.5:
            for previous_index, p_box in enumerate(previous_bboxes):
                current_iou = iou((bbox_x1, bbox_y1), (bbox_x2, bbox_y2), p_box[0], p_box[1])

                # pred_bboxes = predicted_bboxes[previous_index]
                if current_iou > threshold:
                    threshold = current_iou
                    matched_bbox_color = p_box[3]
                    matched_id = p_box[4]
                    matched_index = previous_index
                    kf = p_box[5]
        
        if matched_bbox_color:
            color = matched_bbox_color
            obj_id = matched_id
            if len(matched_box[matched_index]) == 1:
                mutilple_matched_index.append(matched_index)
                print(matched_box[matched_index], bboxes_index)
            matched_box[matched_index].append(bboxes_index)
            visited_previous_index_set.add(matched_index)
            prev2new[matched_index] = len(new_bboxes)  # bboxes_index
        else:
            # color, obj_id, kf = new_object(colors, [mid_point(bbox_x1, bbox_x2), mid_point(bbox_y1, bbox_y2)])
            color, obj_id, kf = new_object(colors)
        new_bboxes.append([(bbox_x1, bbox_y1), (bbox_x2, bbox_y2), score, color, obj_id, kf])
        # centers.append([mid_point(bbox_x1, bbox_x2), mid_point(bbox_y1, bbox_y2)])
        

    for target_previous_index in mutilple_matched_index:
        print("Have multiple")
        multiple_bboxes = matched_box[target_previous_index]
        best_match_index, smallest_area_ratio_diff = None, None
        
        bbox_in_pre_frame = previous_bboxes[target_previous_index]
        p_box = predicted_bboxes[target_previous_index]
        pre_frame_area = rectangle_area(*p_box[0], *p_box[1])
        
        for i, new_bboxes_index in enumerate(multiple_bboxes):
            this_box = new_bboxes[new_bboxes_index]
            this_area = rectangle_area(*this_box[0], *this_box[1])
            area_ratio = this_area/ pre_frame_area
            this_diff = abs(1 - area_ratio)
            if smallest_area_ratio_diff is None or this_diff < smallest_area_ratio_diff:
                smallest_area_ratio_diff = this_diff
                best_match_index = i
        print(multiple_bboxes)
        print("Best", best_match_index)
        for i, new_bboxes_index in enumerate(multiple_bboxes):
            if i != best_match_index:
                print(i)
                box_to_be_corrected = new_bboxes[new_bboxes_index]
                # new_color, new_obj_id, kf = new_object(colors, [mid_point(*box_to_be_corrected[0]), mid_point(*box_to_be_corrected[1])])
                new_color, new_obj_id, kf = new_object(colors)
                print(new_bboxes[new_bboxes_index][4])
                box_to_be_corrected[3] = new_color
                box_to_be_corrected[4] = new_obj_id
                print(new_bboxes[new_bboxes_index][4])
                box_to_be_corrected[5] = kf
            else:
                prev2new[target_previous_index] = new_bboxes_index
    # visited_previous_index_set = set()
    # prev2new = {}
    for i in range(len(previous_bboxes)):
        if i in visited_previous_index_set:
            (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), score, color, obj_id, kf = new_bboxes[prev2new[i]]
            
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color , 2)
            # cv2.rectangle(frame, p_bbox[0], p_bbox[1], p_bbox[3] , 2)
            score = format(score, '.2f')
            draw_text(frame, f'{obj_id} {class_name} {score}', pos=(bbox_x1, bbox_y1))
            new_bboxes[prev2new[i]].append(TTL + 1)
        else:
            # print(new_bboxes[i])
            # new_bboxes[i].append(previous_bboxes[i][6] - 1)
            # previous_bboxes[i][6] -= 1
            
            # update true bboxes
            p_bbox = previous_bboxes[i]
            if p_bbox[6] < STABLED:
                p_bbox[:2] = traslate(p_bbox[0], p_bbox[1], p_bbox[5].predict())
            new_bboxes.append(p_bbox)
            
    for i in set(range(bboxes_no)) - set(prev2new[i] for i in prev2new):
        (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), score, color, obj_id, kf = new_bboxes[i]
        cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color , 2)
        score = format(score, '.2f')
        draw_text(frame, f'{obj_id} {class_name} {score}', pos=(bbox_x1, bbox_y1))
        new_bboxes[i].append(TTL)
    
    # print(new_bboxes)
    
    remove_list = []
    for i, ((bbox_x1, bbox_y1), (bbox_x2, bbox_y2), score, color, obj_id, kf, ttl) in enumerate(new_bboxes):
        # cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color , 2)
        kf.correct(np.array([[mid_point(bbox_x1, bbox_x2)], [mid_point(bbox_y1, bbox_y2)]]))
        predicted = kf.predict()
        # cv2.rectangle(frame, *traslate((bbox_x1, bbox_y1), (bbox_x2, bbox_y2), predicted), (0, 0, 0) , 2)
        
        # draw_text(frame, f'{obj_id} {class_name} {score}', pos=(bbox_x1, bbox_y1))
        
        if new_bboxes[i][6] == 0:
            remove_list.insert(0, i)
        else:
            new_bboxes[i][6] -= 1
    
    for i in remove_list:
        del new_bboxes[i]
    
    return new_bboxes

FRAME = 0
v_no = "Driving Downtown - New York City 4K - USA"
video = f'demo/youtube/{v_no}.mp4'
mid_file = f'demo/mid/{v_no}.txt'
vid_capture = cv2.VideoCapture(video)
# KALMAN_TRACKER_LIST = []
# for i in range(3):
#     KALMAN_TRACKER_LIST.append(Tracker(150, 30, 3))
if (vid_capture.isOpened() == False):
    print("Error opening the video file")
else:
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    frame_width = vid_capture.get(3)
    frame_height = vid_capture.get(4)
    output_vid = cv2.VideoWriter(f'demo/latest{v_no}.mp4', 0x7634706d , fps, (int(frame_width), int(frame_height)))
    c = 0
    previous_frame = [[], [], []]
    while True:
        
        ret, frame = vid_capture.read()
        if ret == True:
            # cv2.imwrite(f"demo/out/{i}.jpg", frame)
            result = inference_detector(model, frame)
            # print(result)
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None

            
            # print(previous_frame)
            for i in range(3):
                previous_frame[i] = draw_rectangles(frame, filter(threshold, bbox_result[i]), i, previous_frame[i], RANGES[i])
            
            # print(previous_frame)
            # cv2.imwrite(f"demo/out1/3c{c}.jpg", frame)
            output_vid.write(frame)
            
            c += 1
            
        else:
            break
    vid_capture.release()
    output_vid.release()
print("Ended")