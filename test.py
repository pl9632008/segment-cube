import cv2 
import numpy as np


thres_names = ["qianhong",
               "qianlv",
               "shenhong",
               "huang",
               "zong",
               "shenlv",
               "lan"
               ]

#@param   image       : 三通道图片
#@return  ans         : 所有分割后的三通道方块图片的列表
#         final_names : 与ans对应的分割后的颜色  
#         min_boxes   ：与ans对应的最小外接矩形
#         centers     : 与ans对应的最小外接矩形的中心点坐标
def findBox(image):

    lower_qianhong = np.array([17, 16, 107])  # 下界阈值
    upper_qianhong = np.array([32, 25, 144])  # 上界阈值

    lower_qianlv = np.array([88, 92, 18])  # 下界阈值
    upper_qianlv = np.array([130, 130, 40])  # 上界阈值

    lower_shenhong = np.array([0, 0, 73])  # 下界阈值
    upper_shenhong = np.array([40, 19, 111])  # 上界阈值

    lower_huang = np.array([1, 123, 149])  # 下界阈值
    upper_huang = np.array([18, 150, 190])  # 上界阈值

    lower_zong = np.array([69, 99, 101])  # 下界阈值
    upper_zong = np.array([88, 122, 152])  # 上界阈值

    lower_shenlv = np.array([36, 73, 19])  # 下界阈值
    upper_shenlv = np.array([64, 90, 45])  # 上界阈值

    lower_lan = np.array([63, 39, 24])  # 下界阈值
    upper_lan = np.array([90, 69, 39])  # 上界阈值

    thresholds = [[lower_qianhong,upper_qianhong ],
                    [lower_qianlv,upper_qianlv],
                    [lower_shenhong,upper_shenhong],
                    [lower_huang,upper_huang],
                    [lower_zong,upper_zong],
                    [lower_shenlv,upper_shenlv],
                    [lower_lan,upper_lan]
                ]
   

    img_canny = cv2.Canny(image,100,200)

    contours, hierarchy = cv2.findContours(img_canny , cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    contours.sort(key=cv2.contourArea, reverse=True)
    masks = []

    for k in range(len(contours)):
        area = cv2.contourArea(contours[k])
        if area < 5:
            continue
        rect = cv2.minAreaRect(contours[k])
        box =  np.int0(cv2.boxPoints(rect))

        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask,[box],255)

        masked = cv2.bitwise_and(image, image, mask=mask)
        masks.append(masked)

    result = []
    ans = []
    names = []
    final_names = []
    min_boxes = []
    centers = []

    for mask in masks:
        for thres in range(len(thresholds)):
            mask_new = cv2.inRange(mask, thresholds[thres][0],thresholds[thres][1])
            cnt = np.count_nonzero(mask_new)
            if cnt > 100:
                res = cv2.bitwise_and(mask, mask, mask=mask_new)
                result.append(res)   
                names.append(thres)
                break
    
    idx = np.zeros([len(result),len(result)])
    
    for i in range(idx.shape[0]):
        for j in range(i+1,idx.shape[1]):
            temp = cv2.bitwise_and(result[i],result[j])
            cnt = np.count_nonzero(temp)          
            if cnt > 1 :
                idx[i][j] = 1

    used_flag =  [ 0 for i in range(len(result))]

    for i in range(idx.shape[0]):
        temp = result[i]
        for j in range(i+1,idx.shape[1]):
            if idx[i][j] == 1:
                temp =  cv2.bitwise_or(temp,result[j])
                used_flag[j] = 1
        if used_flag[i] == 0:
            ans.append(temp)
            final_names.append(names[i])
    
    for i in range(len(ans)) :
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        ans[i] = cv2.dilate(ans[i], kernel, iterations = 2)
        ans[i] = cv2.erode (ans[i], kernel, iterations = 1)

        gray = cv2.cvtColor(ans[i], cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray , cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)

        min_rect = []
        center = []
        for k in range(len(contours)):
            area = cv2.contourArea(contours[k])
            if area < 100:
                continue
            rect = cv2.minAreaRect(contours[k])
            box =  np.int0(cv2.boxPoints(rect))
        
            min_rect.append(box)
            center.append(np.int0(rect[0]))
        centers.append(center)
        min_boxes.append(min_rect)


    return ans,final_names,min_boxes,centers


if __name__ == "__main__":

    image = cv2.imread("./test.jpg")
    
    ans,name_index,min_boxes,centers = findBox(image)


#---------可视化---------
    for i in min_boxes:
            cv2.polylines(image, i, isClosed=True, color=(255, 125, 125), thickness=2)
    
    for i in centers:
        for j in i :
            cv2.circle(image,tuple(j),radius = 2,color = (255,255,3),thickness = 2)

    cv2.imwrite("out.jpg",image)

    for i in range(len(ans)):
        name = "test_"+str(i)+".jpg"
        print(name + " value = ",thres_names[ name_index[i] ])
        cv2.imwrite(name,ans[i])




