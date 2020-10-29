# -*- coding:utf-8 -*-
import os, time


def rebuild_weights():
    if not os.path.exists('./yolov4-obj_final.weights'):
        weights = b''
        for idx in range(0, 10):
            file = './weights/yolov4-obj_final.weights' + '.part' + str(idx)
            with open(file, 'rb') as f:
                weights += f.read()
        with open('./yolov4-obj_final.weights', 'wb') as f:
            f.write(weights)


def get_cur_time():
    time_tup = time.localtime(time.time())
    format_time = '%Y-%m-%d_%H-%M-%S'
    cur_time = time.strftime(format_time, time_tup)
    return cur_time


# class ImageData(object):
#     def __init__(self, image, detection, test_time):
#         self.image = image
#         self.detection = detection
#         self.test_time = test_time

#     def __str__(self):
#         return 'test_time: {}'.format(self.test_time)


# def show_database():

#     with open("show_database_table.md", "r", encoding='utf-8') as fmd:
#         st.markdown(fmd.read()) 
    
#     st.write(query_sqlite())


# def store_into_sqlite(image, detection, test_time):

#     image_data = ImageData(image, detection, test_time)

#     con = sqlite3.connect('data.db')
#     cur = con.cursor()
#     cur.execute("insert into pickled(data) values (?)", (sqlite3.Binary(pickle.dumps(image_data, protocol=2)),))
#     cur.execute("select data from pickled")
#     con.commit()
#     con.close()
#     print('database write done')

#     return True


# def query_sqlite():

#     data_dict = {
#         'test time' : [],
#         'class' : [],
#         'confidence' : [],
#     }

#     con = sqlite3.connect('data.db')
#     cur = con.cursor()  
#     cur.execute("select data from pickled")
#     con.commit()
#     for row in cur:
#         serialized_data = row[0]
#         image_data = pickle.loads(serialized_data)
#         data_dict['test time'].append(image_data.test_time)
#         data_dict['class'].append(image_data.detection['classes'])
#         data_dict['confidence'].append(image_data.detection['confidences'])
#     con.close()
#     print('query finished!')
    
#     return pd.DataFrame(data_dict)