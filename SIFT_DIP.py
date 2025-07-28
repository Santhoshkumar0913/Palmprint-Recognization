
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def get_train_and_test_img_features():
    train_path = r"C:\Users\sandy\Desktop\prog\New folder\Palmprint_Recognition-master\Palmprint\training"
    test_path = r"C:\Users\sandy\Desktop\prog\New folder\Palmprint_Recognition-master\Palmprint\testing"

    train_dataset = []
    test_dataset = []
    
    train_img_list = os.listdir(train_path)
    test_img_list = os.listdir(test_path)

    for train_img in train_img_list:
        img_path = os.path.join(train_path, train_img)
        print(f"Reading {img_path}...") 
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue  # Skip processing if the image cannot be read
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        kp_query, des_query = get_sift_features(equalize)
        train_dataset.append(des_query)

    for test_img in test_img_list:
        img_path = os.path.join(test_path, test_img)
        print(f"Reading {img_path}...")  # Debugging line
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue  # Skip processing if the image cannot be read
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        kp_query, des_query = get_sift_features(equalize)
        test_dataset.append(des_query)

    return train_dataset, test_dataset

def get_sift_features(img, dect_type='sift'):
    if dect_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
    elif dect_type == 'surf':
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None) 
    return kp, des


def sift_detect_match_num(des_query, des_train, ratio=0.70):
    # step3.KNN is used to calculate the number of matching points between the query image and the training image, and k (k=2) nearest neighbor matching is adopted.A successful match is considered when the ratio of the nearest point distance to the second nearest point distance is less than the threshold ratio.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def get_one_palm_match_num(des_query, index, train_dataset, ratio=0.70):
    # Get the number and number of matching points between the query image and the three images belonging to each group in the training image
    match_num_sum = 0
    for i in range(index, index + 3):
        match_num_sum += sift_detect_match_num(des_query, train_dataset[i], ratio=ratio)
    return match_num_sum


def get_match_result(des_query, train_dataset, ratio=0.70):
    # step4.According to the maximum number of matching points, determine the category of the query image
    index = 0
    train_length = len(train_dataset)
    result = np.zeros(train_length // 3, dtype=np.int32)
    while index < train_length:
        result[index // 3] = get_one_palm_match_num(des_query, index, train_dataset, ratio=ratio)
        index += 3
    return result.argmax()


def predict(train_features, test_features, ratio=0.70):
    predict_true = 0
    for i, feature in enumerate(test_features):
        print('Processing image', i + 1,'...')
        # Predict the category of each test image
        category = get_match_result(feature, train_features, ratio=ratio)
        if category == i // 3:
            predict_true += 1
        print('Predict result:', category + 1, 'Groud truth:', i // 3 + 1)
    print('Predict the correct number of pictures:', predict_true, 'Accuracy:', predict_true / len(test_features), 'ratio:', ratio)
    return predict_true / len(test_features)


def show_plot(ratio, acc, name, title):
    # Accuracy change chart
    plt.plot(ratio, acc)
    plt.title(title)
    if not os.path.exists('Image_result'):
        os.makedirs('Image_result')
    plt.savefig(os.path.join('Image_result', name))


def main():
    train_sift_features, test_sift_features = get_train_and_test_img_features()  
    ratio = 0.65
    best_acc = 0
    best_ratio = 0
    ratio_list = []
    acc_list = []
    max_ratio = 0.85

    while ratio <= max_ratio:  # Debugging to check loop
        print(f"Checking ratio: {ratio}")  # Debugging line
        acc = predict(train_sift_features, test_sift_features, ratio)
        acc_list.append(acc)
        ratio_list.append(ratio)
        if acc > best_acc:
            best_acc = acc
            best_ratio = ratio
        ratio += 0.03 # Ensure this actually increases the ratio

    title = f'Best ratio: {best_ratio} | Best accuracy: {best_acc:.4f}'
    plt_name = f"SIFT_{str(max_ratio).split('.')[-1]}"
    show_plot(ratio_list, acc_list, plt_name, title)
    print(title)


if __name__ == '__main__':
    main()
