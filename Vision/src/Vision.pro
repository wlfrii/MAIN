#-------------------------------------------------
#
# Project created by QtCreator 2020-07-14T14:45:14
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Vision
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++17

HEADERS += \
    def/define.h \
    def/micro_define.h \
    def/mtimer.h \
    def/ring_buffer.h \
    def/triple_buffer.h \
    frame_reader.h \
    frame_displayer.h \
    camera_handle.h \
    camera.h \
    image_processor.h \
    camera_parameters.h \
    map_calculator.h \
    terminal.h

    

SOURCES += \
    main.cpp \
    def/mtimer.cpp \
    frame_reader.cpp \
    frame_displayer.cpp \
    camera_handle.cpp \
    camera.cpp \
    image_processor.cpp \
    camera_parameters.cpp \
    map_calculator.cpp \
    terminal.cpp


FORMS +=

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


# ============================== Linux ===============================
# Include the library path, for linux
unix{
    HEADERS += \
        unix/mjpeg2jpeg.h \
        unix/v4l2_capture.h

    SOURCES += \
        unix/mjpeg2jpeg.cpp \
        unix/v4l2_capture.cpp

    INCLUDEPATH += \
        /usr/local/include/opencv4 \
        /usr/local/include

    LIBS += -L"/usr/local/lib" \
        -lturbojpeg \
        -lopencv_world
} # unix end


# ============================== Windows ===============================
# Include the library path, for windows
win32{
    INCLUDEPATH += \
        "D:/opencv_cmake/install/include/"

    CONFIG(debug, debug|release) {
        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
            -lopencv_world420d
    }
    else:CONFIG(release, debug|release){
        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
            -lopencv_world420
    }
} # win32 end
# =====================================================================
