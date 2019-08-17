TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        FaceDetection.cpp \
        main.cpp

HEADERS += \
    FaceDetection.h \
    helpers.hpp

INCLUDEPATH += $$PWD/include

LIBS += -L/usr/lib/x86_64-linux-gnu/ -lpthread

INCLUDEPATH += usr/local/include\
               usr/local/include/opencv \
               usr/local/include/opencv2

LIBS += /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_imgcodecs.so \
        /usr/local/lib/libopencv_videoio.so

LIBS += /home/liu/codes/项目/facerec-gui3/libs/libfacedetection.a
