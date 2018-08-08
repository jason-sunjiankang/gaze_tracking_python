import cv2
import numpy as np

def rotate_bound(image, angle):
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))



if __name__ == '__main__':
    calibration_data = []
    cap = cv2.VideoCapture("D:/sunjiankang/项目/eyeTracking_video/信利_实测2.avi")
    success, frame=cap.read()

    need_solve_model = False
    need_compute_gaze = False

    gaze_x=0.0
    gaze_y=0.0 

    while success:
        success, frame = cap.read() 
        #frame = cv2.imread("1.jpg")
#conver image to gray  
        if frame.shape[2]==1:
            image_gray=frame
        elif frame.shape[2]==3:
            image_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        elif frame.shape[2]==4:
            image_gray=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
        else:
            print("unsupported image channels")
            break       
#image rotation
        #image_gray = flip(image_gray)
       

        image_gray = rotate_bound(image_gray,-90)  
        pupil_roi_gray = image_gray[0:200, 100:600]#[rows, cols]
        image_gauss = cv2.GaussianBlur(pupil_roi_gray, (9,9),0)
        thresh, image_binary = cv2.threshold(image_gauss, 100, 255, cv2.THRESH_BINARY_INV)
        #find contours & ellipse fitting
        image_contours, contours, hierarchy =cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        cv2.drawContours(pupil_roi_gray,contours,-1,(0,0,255),1)
        area = []
        for i in range(0,len(contours)):
             area.append(cv2.contourArea(contours[i]))
             x, y, w, h = cv2.boundingRect(contours[i])
             cv2.rectangle(pupil_roi_gray, (x,y), (x+w,y+h), (255,255,0), 1)
        max_contour_id =np.argmax(area)
        ellipse=[]
        key = cv2.waitKey(1)
        key = key -48
        if(len(contours[max_contour_id])>5):
            ellipse = cv2.fitEllipse(contours[max_contour_id])#return (center,axis,angle)
            cv2.ellipse(pupil_roi_gray, ellipse, (255,255,255), 2)
            cv2.circle(pupil_roi_gray, (int(ellipse[0][0]), int(ellipse[0][1])), 1, (255,255,255), 1)
            #get calibration data
            pupil_center = ellipse[0]
            if(key == 1):
                calibration_data.append(pupil_center)
            elif(key == 2):
                calibration_data.append(pupil_center)
            elif(key == 3):
                calibration_data.append(pupil_center)
            elif(key == 4):
                calibration_data.append(pupil_center)
            elif(key == 5):
                calibration_data.append(pupil_center)
            elif(key == 6):
                calibration_data.append(pupil_center)
            elif(key == 7):
                calibration_data.append(pupil_center)
            elif(key == 8):
                calibration_data.append(pupil_center)
            elif(key == 9):
                calibration_data.append(pupil_center)
                need_solve_model =True
            
            if(need_solve_model):
                if(len(calibration_data)==9):
                    A=np.array([[1, calibration_data[0][0], calibration_data[0][1], calibration_data[0][0]*calibration_data[0][1], calibration_data[0][0]*calibration_data[0][0], calibration_data[0][1]*calibration_data[0][1]],
                            [1, calibration_data[1][0], calibration_data[1][1], calibration_data[1][0]*calibration_data[1][1], calibration_data[1][0]*calibration_data[1][0], calibration_data[1][1]*calibration_data[1][1]],
                            [1, calibration_data[2][0], calibration_data[2][1], calibration_data[2][0]*calibration_data[2][1], calibration_data[2][0]*calibration_data[2][0], calibration_data[2][1]*calibration_data[2][1]],
                            [1, calibration_data[3][0], calibration_data[3][1], calibration_data[3][0]*calibration_data[3][1], calibration_data[3][0]*calibration_data[3][0], calibration_data[3][1]*calibration_data[3][1]],
                            [1, calibration_data[4][0], calibration_data[4][1], calibration_data[4][0]*calibration_data[4][1], calibration_data[4][0]*calibration_data[4][0], calibration_data[4][1]*calibration_data[4][1]],
                            [1, calibration_data[5][0], calibration_data[5][1], calibration_data[5][0]*calibration_data[5][1], calibration_data[5][0]*calibration_data[5][0], calibration_data[5][1]*calibration_data[5][1]],
                            [1, calibration_data[6][0], calibration_data[6][1], calibration_data[6][0]*calibration_data[6][1], calibration_data[6][0]*calibration_data[6][0], calibration_data[6][1]*calibration_data[6][1]],
                            [1, calibration_data[7][0], calibration_data[7][1], calibration_data[7][0]*calibration_data[7][1], calibration_data[7][0]*calibration_data[7][0], calibration_data[7][1]*calibration_data[7][1]],
                            [1, calibration_data[8][0], calibration_data[8][1], calibration_data[8][0]*calibration_data[8][1], calibration_data[8][0]*calibration_data[8][0], calibration_data[8][1]*calibration_data[8][1]]])
                    b=np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0])
                    c=np.array([1.2, 1.2, 0.0, -1.2, -1.2, -1.2, 0.0, 1.2, 0.0])
                    a0, a1, a2, a3, a4, a5 = np.linalg.lstsq(A, b, rcond=None)[0]
                    b0, b1, b2, b3, b4, b5 = np.linalg.lstsq(A, c, rcond=None)[0]
                    need_compute_gaze = True
              
            if(need_compute_gaze):
                gaze_x = a0 + a1*pupil_center[0] + a2*pupil_center[1] + a3*pupil_center[0]*pupil_center[1] + a4*pupil_center[0]*pupil_center[0] + a5*pupil_center[1]*pupil_center[1]
                gaze_y = b0 + b1*pupil_center[0] + b2*pupil_center[1] + b3*pupil_center[0]*pupil_center[1] + b4*pupil_center[0]*pupil_center[0] + b5*pupil_center[1]*pupil_center[1]

            cv2.imshow("image_binary", image_binary)
            cv2.imshow("pupil_roi_gray", pupil_roi_gray)
            cv2.imshow("image_contours", image_contours)
            cv2.imshow("image_gauss", image_gauss)
            
            image_test = np.zeros((500,500,3), dtype=np.uint8)
            image_test =image_test +255
            
        
            calibration_point=[]
            calibration_point.append((3.0*100,1.8*100))
            calibration_point.append((4.0*100,1.8*100))
            calibration_point.append((4.0*100,3.0*100))
            calibration_point.append((4.0*100,4.2*100))
            calibration_point.append((3.0*100,4.2*100))
            calibration_point.append((2.0*100,4.2*100))
            calibration_point.append((2.0*100,3.0*100))
            calibration_point.append((2.0*100,1.8*100))
            calibration_point.append((3.0*100,3.0*100))
            for i in range (0,len(calibration_point)):
                cv2.circle(image_test,(int(calibration_point[i][0]),int(calibration_point[i][1])),3,(0,0,255),1)
            cv2.circle(image_test,(int((gaze_x+3)*100),int((gaze_y+3)*100)),3,(0,0,255),1)   
            cv2.imshow("image_test", image_test)


            
        
        
        
       
        if key==-21:
            cv2.destroyAllWindows()
            break
        

    cap.release()
