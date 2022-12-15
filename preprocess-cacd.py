import dlib
import cv2
import os
import numpy as np
from PIL import Image

print(f'DLIB: {dlib.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')

# Last tested with
# DLIB: 19.22.0
# NumPy: 1.20.2
# OpenCV: 4.5.2

pic_conut = 0
root_path = 'C:/vscode/age_estimation_forlab/'

orig_path = os.path.join(root_path, 'CACD2000')
out_path = os.path.join(root_path, 'CACD2000-centered')

if not os.path.exists(orig_path):
    raise ValueError(f'Original image path {orig_path} does not exist.')

if not os.path.exists(out_path):
    os.mkdir(out_path)

detector = dlib.get_frontal_face_detector()
keep_picture = []
#path = '18yrs.jpg'

for picture_name in os.listdir(orig_path):
    pic_conut+=1
    print(picture_name,pic_conut)
    path = os.path.join(orig_path, picture_name)
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    #img = cv2.imread(os.path.join(orig_path, picture_name))

    detected = detector(img, 1)
    print(detected,len(detected))

    if len(detected) == 0:  # skip if there are 0 or more than 1 face
        print("detected = 0 !!")
        try:
            img = np.array(Image.fromarray(np.uint8(img)).resize((120, 120), Image.ANTIALIAS))
            output_path = os.path.join(out_path, picture_name)
            print(output_path)
            cv2.imencode('.jpg',img)[1].tofile(output_path)
            #cv2.imwrite(os.path.join(out_path, picture_name), tmp)
            print(f'Wrote {picture_name}')
            keep_picture.append(picture_name)
        except ValueError:
            print(f'Failed {picture_name}')
            pass
        

    for idx, face in enumerate(detected):
        width = face.right() - face.left()
        height = face.bottom() - face.top()
        tol = 15
        up_down = 5
        diff = height-width
        #print("diff :",diff,width,height)
        #print("face data ",face.top(),face.bottom(),face.left(),face.right())

        if(diff > 0):
            if not diff % 2:  # symmetric
                #print("1")
                if diff == 0:
                    diff = -1
                if face.left()-tol-int(diff/2) < 0:
                    tol = face.left() - int(diff/2)
                if face.top()-tol-up_down < 0:
                    tol = face.top()
                    up_down = 0

                tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                          (face.left()-tol-int(diff/2)):(face.right()+tol+int(diff/2)),
                          :]
            else:
                #print("2")
                if diff == -1 or diff == 1:
                    diff = 0
                if face.left()-tol-int((diff-1)/2) < 0:
                    tol = face.left() - int((diff-1)/2)
                if face.top()-tol-up_down < 0:
                    tol = face.top()
                    up_down = 0
                tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                          (face.left()-tol-int((diff-1)/2)):(face.right()+tol+int((diff+1)/2)),
                          :]
        if(diff <= 0):
            if not diff % 2:  # symmetric
                #print("3")
                if diff ==0:
                    diff = -1
                if face.left()-tol < 0:
                    tol = face.left()
                if face.top()-tol-int(diff/2)-up_down <0:
                    up_down = 0
                    tol =face.top() - int(diff/2)

                tmp = img[(face.top()-tol-int(diff/2)-up_down):(face.bottom()+tol+int(diff/2)-up_down),
                          (face.left()-tol):(face.right()+tol),
                          :]
                      
                #print(tmp)
            else:
                #print("4")
                if diff == 1 or diff == -1 :
                    diff = 0
                if face.left()-tol < 0:
                    tol = face.left()
                if face.top()-tol-int((diff-1)/2)-up_down <0:
                    up_down = 0
                    tol = face.top() - int((diff-1)/2)


                tmp = img[(face.top()-tol-int((diff-1)/2)-up_down):(face.bottom()+tol+int((diff+1)/2)-up_down),
                          (face.left()-tol):(face.right()+tol),
                          :]
        if len(detected) > 0:  # skip if there are 0 or more than 1 face
            print("detected > 0 !!")
            try:
                tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))
                output_path = os.path.join(out_path, picture_name)
                print(output_path)
                cv2.imencode('.jpg',tmp)[1].tofile(output_path)
                #cv2.imwrite(os.path.join(out_path, picture_name), tmp)
                print(f'Wrote {picture_name}')
                keep_picture.append(picture_name)
            except ValueError:
                print(f'Failed {picture_name}')
                pass
        elif len(detected)==1:            
            try:
                print("detected = 1 !!")
                tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))
                print(os.path.join(out_path, picture_name))
                output_path = os.path.join(out_path, picture_name)
                print(output_path)
                cv2.imencode('.jpg',tmp)[1].tofile(output_path)
                #cv2.imwrite(output_path.encode(), tmp)
                print(f'Wrote {picture_name}')
                keep_picture.append(picture_name)
            except TypeError:#ValueError:
                print(f'Failed {picture_name}')
                pass

        
