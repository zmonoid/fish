import mxnet as mx
import cv2
import numpy as np
from skimage import transform
import glob

model = mx.model.FeedForward.load('checkpoints/inception-bn', 10)
model.ctx = [mx.gpu(0)]
mean_rgb = np.array([123.68, 116.779, 103.939])
mean_rgb = mean_rgb.reshape((3, 1, 1))

def get_labels():
    f = open('data/labels.txt', 'r')
    content = f.readlines()
    lines = [line.strip() for line in content]
    lines = lines[1:]
    return lines

def PreprocessImage(img):
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    resized_img = transform.resize(crop_img, (224, 224))
    sample = np.asarray(resized_img) * 256.0
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    normed_img = sample - mean_rgb
    normed_img = normed_img.reshape((1, 3, 224, 224))
    return normed_img

def test_camera():
    cap = cv2.VideoCapture(0)
    labels = get_labels()
    while True:
        ret, frame = cap.read()
        #frame=cv2.flip(frame, 0)
        #frame=cv2.flip(frame, 1)
        if ret == True:
            cv2.imshow('frame', frame)
        frame = np.asarray(frame)
        frame = PreprocessImage(frame)
        prob = model.predict(frame)
        label = np.argmax(prob[0])
        print label, labels[label]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_dataset():

    filenames = glob.glob('./data/train/space/*.jpg')
    labels = get_labels()
    acc = 0.0
    for filename in filenames:
        #ret, frame = cap.read()
        #frame=cv2.flip(frame, 0)
        #frame=cv2.flip(frame, 1)
        frame = cv2.imread(filename)
        cv2.imshow('frame', frame)
        frame = np.asarray(frame)
        frame = PreprocessImage(frame)
        prob = model.predict(frame)
        label = np.argmax(prob[0])
        if labels[label] == 'space':
            acc += 1
        print label, labels[label]
        #while not (cv2.waitKey(1) & 0xFF == ord('n')):
        #    continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print 'accuray:'
    print acc / len(filenames)

def main():
    test_dataset()

if __name__ == "__main__":
    main()
