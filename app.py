# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import cv2, os
import time

from PIL import Image
from util import *

import base64

st.beta_set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def set_png_as_sidebar_bg(png_file):
    img = cv2.imread(png_file)
    img = cv2.resize(img, (333, 888), cv2.INTER_LINEAR)
    cv2.imwrite(png_file, img)
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .sidebar .sidebar-content {
        background-image: url("data:image/png;base64,%s");
        background-size: fill;
        color: black;
    }
    </style>
    ''' % bin_str
    
    st.sidebar.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')
set_png_as_sidebar_bg('sidebar.png')


def main():
    rebuild_weights()
    coll, colb, colr = st.beta_columns((1, 4, 1))

    with open("instructions.md", "r", encoding='utf-8') as fmd:
        readme_text = colb.markdown(fmd.read())

    st.sidebar.title("功能菜单")
    app_mode = st.sidebar.selectbox("选择应用模式", ["应用介绍", "单图检测", "测试展示"])
    if app_mode == "应用介绍":
        st.sidebar.success('选择"单图检测"进行卷烟目标识别')
    elif app_mode == "单图检测":
        readme_text.empty()
        run_the_app()
    elif app_mode == "测试展示":
        readme_text.empty()
        display_data_test()
    # elif app_mode == "测试记录":
    #     readme_text.empty()
    #     show_database()


def run_the_app():

<<<<<<< HEAD
    coll, colb, colr = st.beta_columns((1, 4, 1))
    
    with open("run_the_app.md", "r", encoding='utf-8') as fmd:
        run_the_app_text = colb.markdown(fmd.read()) 
=======
    with open("run_the_app.md", "r", encoding='utf-8') as fmd:
        run_the_app_text = st.markdown(fmd.read()) 
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19

    confidence_threshold, overlap_threshold = object_detector_ui()

    st.set_option('deprecation.showfileUploaderEncoding', False)
<<<<<<< HEAD
    file_up = colb.file_uploader(label="上传测试图片", type=None)

    col1, colb, col2 = st.beta_columns((5, 1, 5))
=======
    file_up = st.file_uploader(label="上传测试图片", type=None)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19

    if file_up is not None:
        run_the_app_text.empty()
        image = Image.open(file_up)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
<<<<<<< HEAD
        img =  cv2.resize(img, (416, 416), cv2.INTER_LINEAR)
        
        col1.markdown('### 测试图片:')
        image = image.resize((512, 320))
        col1.image(image, use_column_width=True)
=======
        img =   cv2.resize(img, (416, 416), cv2.INTER_LINEAR)
        
        st.markdown('### 测试图片:')
        image = image.resize((512, 320))
        st.image(image, use_column_width=True)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
        
        curTime = get_cur_time()
        classes, confidences, boxes, costTime = yolov4(img, confidence_threshold, overlap_threshold)

<<<<<<< HEAD
        bimg = cv2.imread('VS.jpg')
        bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
        colb.markdown('#  ')
        colb.markdown('#  ')
        colb.markdown('#  ')
        colb.markdown('#  ')
        colb.markdown('#  ')
        colb.image(bimg, use_column_width=True)

        col2.markdown('### 检测结果:')
=======
        st.markdown('### 检测结果:')
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
        if type(classes) == tuple:
            st.warning("未发现卷烟目标或者算法检测失败！！ ╮(╯▽╰)╭")
        else:
            st.sidebar.warning("发现疑似目标--香烟，警告!!")
            show_detect_result(curTime, classes, confidences, boxes, costTime)
            st.balloons()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
            draw_image_with_predict_boxes(img, classes, confidences, boxes, col2, None, False)


def display_data_test():
    st.markdown('# 测试展示')
    st.markdown('# ')
    selected_image_index, selected_image, selected_image_boxes = image_selector_ui()
    confidence_threshold, overlap_threshold = object_detector_ui()

    col1, colb, col2 = st.beta_columns((5, 1, 5))

    bimg = cv2.imread('VS.jpg')
    bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
    colb.markdown('#  ')
    colb.markdown('#  ')
    colb.markdown('#  ')
    colb.markdown('#  ')
    colb.markdown('#  ')
    colb.markdown('#  ')
    colb.image(bimg, use_column_width=True)

=======
            draw_image_with_predict_boxes(img, classes, confidences, boxes, None, False)


def display_data_test():
    selected_image_index, selected_image, selected_image_boxes = image_selector_ui()
    confidence_threshold, overlap_threshold = object_detector_ui()

>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
    curTime = get_cur_time()
    origin_image = selected_image.copy()
    predict_image = selected_image.copy()
    classes, confidences, boxes, costTime = yolov4(predict_image, confidence_threshold, overlap_threshold)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
    draw_image_with_real_boxes(origin_image, selected_image_boxes, col1, "**Human-annotated data** (image `%i`)" % selected_image_index)
=======
    draw_image_with_real_boxes(origin_image, selected_image_boxes, "**Human-annotated data** (image `%i`)" % selected_image_index)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
    if type(classes) == tuple:
        st.warning("未发现卷烟目标或者算法检测失败！！ ╮(╯▽╰)╭")
    else:
        show_detect_result(curTime, classes, confidences, boxes, costTime)
        predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
        draw_image_with_predict_boxes(predict_image, classes, confidences, boxes, col2, "**YOLO v4 Model** (overlap `%3.2f`) (confidence `%3.2f`)" % (overlap_threshold, confidence_threshold))
=======
        draw_image_with_predict_boxes(predict_image, classes, confidences, boxes, "**YOLO v4 Model** (overlap `%3.2f`) (confidence `%3.2f`)" % (overlap_threshold, confidence_threshold))
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19


def image_selector_ui():

    @st.cache
    def load_data(image_root_path, test_data_path, boxes_root_path):
        with open(test_data_path, 'r') as f:
            data = [os.path.join(image_root_path, line.strip()+'.jpg') for line in f.readlines()]
        boxes = []
        for item in data:
            boxes_path = os.path.join(boxes_root_path, item.split('/')[-1].split('.')[0] + '.txt')
            with open(boxes_path, 'r') as f:
                boxes.append([line.strip().split(' ')[1:] for line in f.readlines()])
            for i in range(0, len(boxes[-1])):
                for j in range(0, len(boxes[-1][i])):
                    boxes[-1][i][j] = int(float(boxes[-1][i][j])*416)
                boxes[-1][i][0] = int(boxes[-1][i][0] - boxes[-1][i][2]/2)
                boxes[-1][i][1] = int(boxes[-1][i][1] - boxes[-1][i][3]/2)
        return data, boxes

    image_root_path = "./data/VOCdevkit/VOC2020/JPEGImages/"
    test_data_path = "./data/VOCdevkit/VOC2020/ImageSets/Main/test.txt"
    boxes_root_path = "./data/VOCdevkit/VOC2020/labels/"
    data, boxes = load_data(image_root_path, test_data_path, boxes_root_path)

    st.sidebar.markdown("# 测试集图片")
    selected_image_index = st.sidebar.slider("选择图片 (index)", 0, len(data)-1, 0)
    selected_image = load_image(data[selected_image_index])
    selected_image_boxes = boxes[selected_image_index]
    # st.write(data[selected_image_index].split('/')[-1])

    return selected_image_index, selected_image, selected_image_boxes
    

def object_detector_ui():
    st.sidebar.markdown("## 模型参数设置:")
<<<<<<< HEAD
    confidence_threshold = st.sidebar.slider("置信度阈值(0.25)：", 0.0, 1.0, 0.25, 0.01)
    overlap_threshold = st.sidebar.slider("IOU 阈值(0.45)：", 0.0, 1.0, 0.45, 0.01)
=======
    confidence_threshold = st.sidebar.slider("置信度阈值：", 0.0, 1.0, 0.25, 0.01)
    overlap_threshold = st.sidebar.slider("IOU 阈值：", 0.0, 1.0, 0.5, 0.01)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
    return confidence_threshold, overlap_threshold


@st.cache
def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (416, 416), cv2.INTER_LINEAR)
    return image


@st.cache
def load_names(names_path):
    with open(names_path, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    return names


<<<<<<< HEAD
def yolov4(image, confidence_threshold=0.25, overlap_threshold=0.45):
=======
def yolov4(image, confidence_threshold=0.25, overlap_threshold=0.50):
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19

    @st.cache(allow_output_mutation=True)
    def load_net(cfg, weights):
        net = cv2.dnn_DetectionModel(cfg, weights)
        net.setInputSize(416, 416)
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)
        return net

    net = load_net('./yolov4-obj.cfg', './yolov4-obj_final.weights')

    startTime = time.time()
<<<<<<< HEAD
    classes, confidences, boxes = net.detect(image, confThreshold=confidence_threshold, nmsThreshold=overlap_threshold)
=======
    classes, confidences, boxes = net.detect(image, confThreshold=0.25, nmsThreshold=0.45)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
    endTime = time.time()
    costTime = endTime - startTime

    return classes, confidences, boxes, costTime


<<<<<<< HEAD
def draw_image_with_real_boxes(image, boxes, col1, description, show_info=True):
=======
def draw_image_with_real_boxes(image, boxes, description, show_info=True):
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19
    for box in boxes:
        label = '%s: %s' % ('cigarette', 'ground truth')
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        left, top, width, height = box
        top = max(top, labelSize[1])
        cv2.rectangle(image, box, color=(255, 0, 0), thickness=3)
        cv2.rectangle(image, (left, top-labelSize[1]), (left+labelSize[0], top+baseLine), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    if show_info:
<<<<<<< HEAD
        col1.header(description)
        col1.markdown('# ')
    image = cv2.resize(image, (512, 320), cv2.INTER_LINEAR)
    col1.image(image.astype(np.uint8), use_column_width=True)


def draw_image_with_predict_boxes(image, classes, confidences, boxes, col2, description, show_info=True):
=======
        st.markdown(description)
    image = cv2.resize(image, (512, 320), cv2.INTER_LINEAR)
    st.image(image.astype(np.uint8), use_column_width=True)


def draw_image_with_predict_boxes(image, classes, confidences, boxes, description, show_info=True):
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19

    names = load_names('obj.names')

    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
	    label = '%.2f' % confidence
	    label = '%s: %s' % (names[classId], label)
	    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	    left, top, width, height = box
	    top = max(top, labelSize[1])
	    cv2.rectangle(image, box, color=(255, 0, 0), thickness=3)
	    cv2.rectangle(image, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
	    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    if show_info:
<<<<<<< HEAD
        col2.header(description)
        col2.markdown('# ')
    image = cv2.resize(image, (512, 320), cv2.INTER_LINEAR)
    col2.image(image.astype(np.uint8), use_column_width=True)
=======
        st.markdown(description)
    image = cv2.resize(image, (512, 320), cv2.INTER_LINEAR)
    st.image(image.astype(np.uint8), use_column_width=True)
>>>>>>> 0498c8e94b733b66095a48d60e40c12f2fb74a19


def show_detect_result(curTime, classes, confidences, boxes, costTime):
    
    names = load_names('obj.names')

    detect_result = '## 测试结果:\n'
    detect_result += '| 测试时间 | {} |\n'.format(curTime)
    detect_result += '| :--: | :--: |\n'
    detect_result += '| 检测时长 | {:.15f}(s) |\n'.format(costTime)
    for clss, conf in zip(classes, confidences):
        detect_result += '| 目标类别 | {} |\n'.format(names[clss[0]])
        detect_result += '| 类置信度 | {} |\n'.format(conf[0])
    
    st.sidebar.markdown(detect_result)


if __name__ == '__main__':
    main()
