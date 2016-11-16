import cv2
import os

def record_cam(cap, name):
    count = 1
    record = False
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            #frame = cv2.flip(frame, 0)
            cv2.imshow('camera', frame)
            #cv2.imwrite('test.jpg', frame)
            if record:
                cv2.imwrite("./train/%s/img_%d.jpg" % (name, count), frame)
                count += 1
            if cv2.waitKey(1) & 0xFF == ord('s'):
                record = not record
                if record:
                    print "Record started, press s to stop"
                else:
                    print "Record ended..."
                    return
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()



def main():
    cap = cv2.VideoCapture(0)
    names = raw_input("Type the sequence of the object name seperated by comma, e.g: legs, wires, papers, floormat\n")
    names = names.split(',')
    #os.system('rm -rf train/*')
    for name in names:
        os.system('cd train;rm -rf %s;cd ..' % name)
        name = name.strip()
        print "Taking video of object %s" % name
        print "Switch to video window and press s to start..."
        os.system('mkdir train/%s' % name)
        record_cam(cap, name)

if __name__ == "__main__":
    main()
