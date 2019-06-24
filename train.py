import os
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import fire
from models import SurvivalModel

def gen_dataset(data_dir):
    mask_dir = data_dir + '_mask'
    img_ori_dir = data_dir + '_ori'
    datasets = []
    datasets_name = []
    f_list = os.listdir(img_ori_dir)

    #train_max = 0
    #train_min = 0
    #test_max = 0
    #test_min = 0

    for index, i in enumerate(f_list):
        name, os2, event = os.path.splitext(i)[0].split('_')

        X = np.load(os.path.join(img_ori_dir, i))
        m = np.load(os.path.join(mask_dir, i))
        X0 = X.copy()
        m0 = m.copy()
        #m = (m*255).astype(np.uint8).copy()
        #print (np.max(m), np.min(m), np.unique(m))
        #_, cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print (cnts)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            img_max = np.max(X[X<175])
            img_min = np.min(X[X>-75])
            X[X>175] = img_max
            X[X<-75] = img_min
        elif data_dir.split('/')[-1] == 'test_single': 
            img_max = np.max(X[X<125])
            img_min = np.min(X[X>-125])
            X[X>125] = img_max
            X[X<-125] = img_min
        '''

        img_max = np.max(X[X<125])
        img_min = np.min(X[X>-125])
        X[X>125] = img_max
        X[X<-125] = img_min

        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        X = clahe.apply(X)
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        
        #single_img_save_path = data_dir.rstrip('/') + '_img'
        #os.makedirs(single_img_save_path, exist_ok=True)
        #io.imsave(os.path.join(single_img_save_path, f'{name}_{index}.jpg'), X)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            train_max = max(train_max, X[m>0].max())
            train_min = min(train_min, X[m>0].min())
        elif data_dir.split('/')[-1] == 'test_single': 
            test_max = max(test_max, X[m>0].max())
            test_min = min(test_min, X[m>0].min())

        continue
        '''
        x, y = np.where(m>0)

        #x_mean, y_mean = np.mean(x), np.mean(y)
        #print (x_mean, y_mean)
        w0, h0 = m.shape
        x_min = max(0, int(np.min(x)-5))
        x_max = min(w0, int(np.max(x)+5))
        y_min = max(0, int(np.min(y)-5))
        y_max = min(h0, int(np.max(y)+5))

        #print (x_min, x_max, y_min, y_max)
        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max] 

        X_m_1 = X.copy() 
        #X_m_1[m<1.0] = 0


        #X_m_1 = ((X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0])))*0.9+0.1
        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X_m_1[m==0] = 0
        #print (np.unique(X_m_1))
        #raise


        X_m_2 = X.copy() 
        X_m_2[m>0] = 0


        #print (X_m_1.max(), X_m_1.min(), np.unique(X_m_1))
        #print (X_m_1[X_m_1>0].max(), X_m_1[X_m_1>0].min(), np.unique(X_m_1[X_m_1>0]))
        #plt.imshow(X_m_1)
        #plt.show()
        #raise        

        h, w = X_m_1.shape
        #print (w, h)

        if h < w:
            pad_1 = (w - h)//2
            pad_2 = w - pad_1 - h
            X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2),(0,0)), 'constant', constant_values=(0, 0))
            m = np.lib.pad(m, ((pad_1, pad_2),(0,0)), 'constant', constant_values=(0, 0))
        elif h >= w:
            pad_1 = (h - w)//2
            pad_2 = h - pad_1 - w
            X_m_1 = np.lib.pad(X_m_1, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0, 0))
            m = np.lib.pad(m, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0, 0))

        #print (X_m_1.min(), X_m_1.max())

        if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:
            #X = cv2.resize(X, (96, 96), interpolation=cv2.INTER_CUBIC)
            #m = cv2.resize(m, (96, 96), interpolation=cv2.INTER_CUBIC)
            #X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_NEAREST)
            X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)
            #X_m_2 = cv2.resize(X_m_2, (96, 96), interpolation=cv2.INTER_CUBIC)

        #X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        #X_m_1[m==0] = 0

        #print (X_m_1.max(), X_m_1.min(), X_m_1.shape)
        #print (m.max(), m.min(), m.shape)
        #raise

        if m0.shape[0] != 160 or m0.shape[1] != 160:
            m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

        #print (X_m_1.min(), X_m_1.max())
        #raise

        #single_img_save_path = data_dir.rstrip('/') + '_img_cut'
        #os.makedirs(single_img_save_path, exist_ok=True)
        #io.imsave(os.path.join(single_img_save_path, f'{name}_{os2}_{event}.jpg'), X_m_1)

        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X_m_1[m<=0] = 0
        #print (X.shape, np.max(X_m_1), np.min(X_m_1))
        #raise

        X_m_1 = np.expand_dims(X_m_1, axis=2)
        m = np.expand_dims(m, axis=2)
        m0 = np.expand_dims(m0, axis=2)

        #X_m_2 = np.expand_dims(X_m_2, axis=2)

        XX = np.concatenate((X_m_1, X_m_1, X_m_1), axis=-1) 
        #XX = X_m_1

        datasets.append((XX[None,...], np.array([float(os2)]), np.array([int(event)])))
        datasets_name.append(name)


    set_name = data_dir.split('/')[-1]
    print (f'{set_name}: {len(datasets)}')
    print (f'{set_name}: {len(datasets_name)}')
    return datasets, datasets_name


def load_dataset(data_path):

    train_dir = os.path.join(data_path, 'test_single')
    val_dir = os.path.join(data_path, 'val_single')
    test_dir = os.path.join(data_path, 'train_single')

    return gen_dataset(train_dir), gen_dataset(val_dir), gen_dataset(test_dir)

def main(data_path='data'):

    [datasets_train, train_name], [datasets_val, val_name], [datasets_test, test_name] = load_dataset(data_path)
    survival_model = SurvivalModel()
    survival_model.fit(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, loss_func='cox', epochs=1000, lr=0.0001, mode='merge', batch_size = 32)

    #survival_model.fit(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, loss_func='cox', epochs=1000, lr=0.0001, mode='infer', batch_size = 32)

    #survival_model.fit(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, loss_func='cox', epochs=1000, lr=0.0001, mode='vis', batch_size = 32)

    #survival_model.fit(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, loss_func='cox', epochs=1000, lr=0.0001, mode='vis_cam', batch_size = 32)

if __name__ == '__main__':
    fire.Fire(main)


