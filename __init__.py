import cv2
import numpy as np
cap = cv2.VideoCapture('D:/Do an 3/Test/video4.mp4')
"""ret,last_frame=cap.read()
if last_frame is None:
    exit()"""
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FILE_OUTPUT = 'output4.avi'
out=cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))
while (cap.isOpened()):
    ret, image = cap.read()
    if image is None:
        break

    #image = cv2.imread("D:/Do an 3/Test/A2.jpg")
    #cv2.imshow("Hinh", image)
    # RGB
    """
    Whitemin=np.uint8([200,200,200])
    Whitemax=np.uint8([255,255,255])ds
    white=cv2.inRange(image,Whitemin,Whitemax)
    cv2.imshow('Mn',white)
    Yemin=np.uint8([100,190,190])
    Yemax = np.uint8([255,225,255])
    Ye = cv2.inRange(image,Yemin,Yemax)
    cv2.imshow('Mn1',Ye)
    mask = cv2.bitwise_or(white,Ye)
    masked = cv2.bitwise_and(image,image,mask=mask)
    cv2.imshow('A',masked)"""
    # HSV"
    # """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    """cv2.imwrite("HSV.jpg",image1)
    cv2.imwrite("HLS.jpg",image2)"""
    Whitemin = np.uint8([0, 200, 0])
    Whitemax = np.uint8([255, 255, 255])
    whitemask = cv2.inRange(image2, Whitemin, Whitemax)
    # cv2.imshow ("W",whitemask)
    Yemin = np.uint8([10, 10, 150])
    Yemax = np.uint8([160, 255, 255])
    yemask = cv2.inRange(image2, Yemin, Yemax)
    # cv2.imshow("Y",yemask)
    mask = cv2.bitwise_or(whitemask, yemask)
    mask1 = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow ('mask',mask1)
    # cv2.imwrite("mask.jpg",mask1)
    Gray = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
    Blur = cv2.GaussianBlur(Gray, (15, 15), 0)
    # cv2.imshow("Gray",Gray)
    # cv2.imshow('Blur',Blur)
    low_threshold = 50
    high_threshold = 150
    canny_edge = cv2.Canny(Blur, low_threshold, high_threshold)
    # cv2.imshow ('Edge',canny_edge)
    # cv2.imwrite("Edge.jpg",canny_edge)
    # ROI
    r, c = canny_edge.shape[:2]
    bottom_left = [c * 0.1, r * 0.95]
    top_left = [c * 0.4, r * 0.6]
    bottom_r = [c * 0.9, r * 0.95]
    top_r = [c * 0.6, r * 0.6]
    vert = np.array([[bottom_left, top_left, top_r, bottom_r]], dtype=np.int32)
    # pts=vert.reshape((-1,1,2))
    # img=cv2.polylines(image,[vert],True,(0,255,255),3)
    # cv2.imshow("Img",img)
    mask = np.zeros_like(canny_edge)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vert, 255)
    else:
        cv2.fillPoly(mask, vert, (255,) * mask.shape[2])
    ROI = cv2.bitwise_and(canny_edge, mask)
    # cv2.imwrite ("ROI.jpg",ROI)
    # cv2.imshow("ROI",ROI)
    """lines=cv2.HoughLines(ROI,rho=1,theta=np.pi/180,threshold=50)
    for line in lines:
        rho, theta=line[0]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))
        y1=int(y0+1000*a)
        x2=int(x0-1000*(-b))
        y2=int(y0-1000*(a))
        L=cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("Line",L)"""
    """

    lines=cv2.HoughLinesP(ROI,1,np.pi/180,20,minLineLength=20,maxLineGap=300)
    for line in lines:
        x1,y1,x2,y2=line[0]
        L=cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("L",L)"""
    lines = cv2.HoughLinesP(ROI, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=300)

    """left_L=[]
    left_W=[]
    right_L=[]
    right_W=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2==x1:
                continue
            slope=(y2-y1)/(x2-x1)
            intercept=y1-slope*x1
            length=np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope<0:
                left_L.append((slope,intercept))
                left_W.append((length))
            else:
                right_L.append((slope,intercept))
                right_W.append((length))
    left_lane=np.dot(left_W,left_L)/np.sum(left_W) if len(left_W)>0 else None
    right_lane=np.dot(right_W,right_L)/np.sum(right_W) if len(right_W)>0 else None"""
    left_L = []
    left_W = []
    right_L = []
    right_W = []


    def slope_intercept(lines):
        left_L = []
        left_W = []
        right_L = []
        right_W = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < 0:
                    left_L.append((slope, intercept))
                    left_W.append((length))
                else:
                    right_L.append((slope, intercept))
                    right_W.append((length))
        left_lane = np.dot(left_W, left_L) / np.sum(left_W) if len(left_W) > 0 else None
        right_lane = np.dot(right_W, right_L) / np.sum(right_W) if len(right_W) > 0 else None
        return left_lane, right_lane


    def make_line(y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))


    def lane_line(image, lines):
        left_lane, right_lane = slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = make_line(y1, y2, left_lane)
        right_line = make_line(y1, y2, right_lane)
        return left_line, right_line


    def draw_line(image, lines, color=[255, 0, 0], thickness=20):
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


    left_line, right_line = lane_line(image, lines)
    A = draw_line(image, (left_line, right_line))
    cv2.imshow("A", A)
    if ret == True:

        frame = cv2.flip(A, 1)

        out.write(frame)
    # cv2.imwrite("KQ.jpg",A)
    # lane_image=[]
    # lane_image.append(draw_line(image,lane_image(image,lines)))
    if cv2.waitKey(33) >= 0:
        break
cap.release()
out.release()
cv2.destroyAllWindows()