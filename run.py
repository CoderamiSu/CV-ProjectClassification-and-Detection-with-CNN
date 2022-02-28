import cv2
import numpy as np
from tensorflow import keras

def NMS(bboxes, limit, scores):

    boxes = []
    keep=[]

    area = bboxes[:,2]*bboxes[:,3]
    idx = list(np.argsort(scores))

    while len(idx)>0:
        i=idx.pop()
        box=bboxes[i]
        boxes.append(box)
        keep.append(i)
        
        x1=box[0]
        y1=box[1]
        x2=box[0]+box[2]-1
        y2=box[1]+box[3]-1
        a=area[i]


        drop=[]
        for j in idx:
            box_=bboxes[j]
            x1_=box_[0]
            y1_=box_[1]
            x2_=box_[0]+box_[2]-1
            y2_=box_[1]+box_[3]-1
            h=max(0,min(x2,x2_)-max(x1,x1_)+1)
            w=max(0,min(y2,y2_)-max(y1,y1_)+1)
            overlap=(h*w)/a
            
            if overlap>limit:
                drop.append(j)

        idx=[i for i in idx if i not in drop]

    return boxes

def predicted(img, bboxes):
    crops=np.zeros((len(bboxes),32,32,3)).astype(np.uint8)
    for i in range(len(bboxes)):
        b=bboxes[i]
        x1=max(0,b[1]-1)
        x2=min(img.shape[0],b[1]+b[3]+1)
        y1=max(0,b[0]-1)
        y2=min(img.shape[1],b[0]+b[2]+1)
        crop=img[x1:x2,y1:y2]
        crop=cv2.resize(crop,(32,32),interpolation = cv2.INTER_AREA)
        crops[i]=crop

    model=keras.models.load_model("model.h5")
    prob=model.predict(crops)
    return prob


def run(in_img, out_img):

    img=cv2.imread(in_img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()

    blur=cv2.GaussianBlur(gray,(3,3),15)
    mser_out = mser.detectRegions(blur)

    regions=[]
    bboxes=[]

    img_area=img.shape[0]*img.shape[1]

    for i in range(len(mser_out[0])):
        r=mser_out[0][i]
        b=mser_out[1][i]
        ar=mser_out[1][i][2]/mser_out[1][i][3]

        if b[2]*b[3]>img_area*0.001 and b[2]*b[3]<img_area*0.3 and \
            b[2]/b[3]<1 and b[2]/b[3]>0.2:

            regions.append(mser_out[0][i])
            bboxes.append(mser_out[1][i])

    prob=predicted(img, bboxes)

    scores=1-prob[:,0]

    bboxes=np.array(bboxes)
    bboxes=NMS(bboxes, 0.3, scores)

    prob=predicted(img, bboxes)

    pred=np.argmax(prob,axis=1)

    out=img.copy()
    for i in range(len(bboxes)):
        if pred[i]>0:
            if pred[i]==10:
                n=0
            else:
                n=pred[i]
            b=bboxes[i]
            s=int(img.shape[0]/25)
            x=max(0,b[0]-s)
            y=max(0,b[1]-s)
            size=int((img.shape[0]>500)+1)
            cv2.putText(out, str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 
                        size, (255, 0, 255), 2, cv2.LINE_AA)

            p1=(b[0],b[1])
            p2=(b[0]+b[2],b[1]+b[3])
            cv2.rectangle(out,p1,p2,(0, 255, 0), 2)

    cv2.imwrite(out_img, out)

for i in range(1,6):
    in_img="./input_images/{}.png".format(i)
    out_img="./graded_images/{}.png".format(i)
    run(in_img,out_img)
