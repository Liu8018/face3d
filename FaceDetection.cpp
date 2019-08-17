#include "FaceDetection.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

#define DETECT_BUFFER_SIZE 0x20000

const float FACEDT_CONF_THRESHOLD = 90;

FaceDetection g_faceDT;

FaceDetection::FaceDetection()
{
    pResults = nullptr; 
    pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        exit(0);
    }

    resizeWidth = 160;//最小84
    resizeRatio = -1;
}

void FaceDetection::detect(const cv::Mat &img, std::vector<cv::Rect> &boxes)
{
    resizeRatio = resizeWidth/(float)img.cols;
        
    //std::cout<<"src.size:"<<src.size<<std::endl;
    cv::Mat image;
    cv::resize(img,image,cv::Size(),resizeRatio,resizeRatio);
    
    //std::cout<<"image.size:"<<image.size<<std::endl;
    
    pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
    
    boxes.clear();
    for(int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short * p = ((short*)(pResults+1))+142*i;
        int x = p[0];
        int y = p[1];
        int w = p[2];
        int h = p[3];
        int confidence = p[4];
        //int angle = p[5];

        x /= resizeRatio;
        y /= resizeRatio;
        w /= resizeRatio;
        h /= resizeRatio;

        //printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x,y,w,h,confidence, angle);

        if(confidence > FACEDT_CONF_THRESHOLD)
        {
            cv::Rect faceRect = cv::Rect(x, y, w, h);
            if(faceRect.br().x > img.cols-1)
                faceRect.width = img.cols-1-faceRect.x;
            if(faceRect.br().y > img.rows-1)
                faceRect.height = img.rows-1-faceRect.y;
            if(faceRect.x < 0)
                faceRect.x = 0;
            if(faceRect.y < 0)
                faceRect.y = 0;
            
            boxes.push_back(faceRect);
        }
    }
    /*
    if(m_net.empty())
        m_net = cv::dnn::readNetFromTensorflow(FACEDT_MODEL_PATH,FACEDT_MODELCONF_PATH);
    
    boxes.clear();
    
    cv::Mat blob = cv::dnn::blobFromImage(img,1.0,cv::Size(300,300),cv::Scalar(104,117,123),true,false);
    
    m_net.setInput(blob);
    cv::Mat detectionMat = m_net.forward();
    cv::Mat detections(detectionMat.size[2], detectionMat.size[3], CV_32F, detectionMat.ptr<float>());
    
    for (int i = 0; i < detections.rows; i++)
    {
        float confidence = detections.at<float>(i, 2);

        if (confidence > FACEDT_CONF_THRESHOLD)
        {
            int xLeftBottom = static_cast<int>(detections.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detections.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detections.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detections.at<float>(i, 6) * img.rows);

            cv::Rect rect(xLeftBottom, yLeftBottom,
                (xRightTop - xLeftBottom),
                (yRightTop - yLeftBottom));
            
            boxes.push_back(rect);
        }
    }
    */
}
