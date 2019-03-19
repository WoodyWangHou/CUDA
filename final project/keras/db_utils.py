import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

# used to downsize the image
def compress_images():
    directory = '../datasets/images/'
    list_direct = '../datasets/lists/'
    train_list = sio.loadmat(list_direct + 'train_list.mat')
    test_list = sio.loadmat(list_direct + 'test_list.mat')
    X_test_path = test_list['file_list']
    X_train_path = train_list['file_list']
    
    for index in range(X_train_path.shape[0]):
        img = Image.open(directory + str(X_train_path[index][0][0]))
        img = img.convert("RGB")
        img = img.resize((100, 114));
        img.save(directory + str(X_train_path[index][0][0]),quality=95)
        img.close()
    
    for index in range(X_test_path.shape[0]):
        img = Image.open(directory + str(X_test_path[index][0][0]))
        img = img.convert("RGB")
        img = img.resize((100, 114));
        img.save(directory + str(X_test_path[index][0][0]),quality=95)
        img.close()
    
def load_datasets(numOfBreeds=15):
    directory = '../datasets/images/'
    list_direct = '../datasets/lists/'
    train_list = sio.loadmat(list_direct + 'train_list.mat')
    
    X_train_path = train_list['file_list']
    X_train_path = X_train_path[:numOfBreeds * 100]
    X_train_orig = np.array([mpimg.imread(directory + str(X_train_path[index][0][0])) for index in range(X_train_path.shape[0])])
    
    Y_train_orig = train_list['labels']
    Y_train_orig = Y_train_orig[:numOfBreeds * 100]

    test_list = sio.loadmat(list_direct + 'test_list.mat')
    X_test_path = test_list['file_list']
    X_test_path = X_test_path[:1306]
    X_test_orig = np.array([mpimg.imread(directory + str(X_test_path[index][0][0])) for index in range(X_test_path.shape[0])])
    
    Y_test_orig = test_list['labels']
    Y_test_orig = Y_test_orig[:1306] # for first 15 breeds
    
    return X_train_orig, Y_train_orig, X_test_orig, Y_test_orig