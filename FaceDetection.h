#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "/home/liu/codes/项目/facerec-gui3/include/facedetectcnn.h"
#include <opencv2/core.hpp>

class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &img, std::vector<cv::Rect> &boxes);
    
private:
    int * pResults;
    unsigned char * pBuffer;
    
    int resizeWidth;
    float resizeRatio;
};

extern FaceDetection g_faceDT;

#endif // FACEDETECTOR_H
