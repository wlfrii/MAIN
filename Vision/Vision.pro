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
CONFIG(debug, debug|release){
    DESTDIR = $$PWD/../build/bin/debug
}else{
    DESTDIR = $$PWD/../build/bin/release
}


HEADERS += \
    $$files("./src/*.h", false) \
    $$files("./src/def/*.h", false) \
    $$files("./src/ui/*.h", false)

SOURCES += \
    $$files("./src/*.cpp", false) \
    $$files("./src/def/*.cpp", false) \
    $$files("./src/ui/*.cpp", false)


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


# ============================== Linux ===============================
# Include the library path, for linux
unix{
    HEADERS += $$files("./src/unix/*.h", true) \
                $$files("./src/usb/*.h", false)

    SOURCES += $$files("./src/unix/*.cpp", true) \
                $$files("./src/usb/*.cpp", false)

    INCLUDEPATH += \
        /usr/local/include/opencv4 \
        /usr/local/include

    LIBS += -L"/usr/local/lib" \
        -lturbojpeg \
        -lopencv_world \
        -luvc

    # Include the lib_gpu_vision
    INCLUDEPATH += \
        "/media/wanglf/DATA/MyProjects/lib_vision_gpu/src" \
        "$$PWD/../lib_utility/src"
    CONFIG(debug, debug|release){
        LIBS += -L"../../lib/debug" \
            -llib_vision_gpu \
            -llib_utility
    }else{
        LIBS += -L"../../lib/release" \
            -llib_vision_gpu \
            -llib_utility
    }

    # --------------------- CUDA -----------------------
    CUDA_DIR = "/usr/local/cuda-11.0"   # Path to cuda toolkit install

    QMAKE_LIBDIR += $$CUDA_DIR/lib64

    INCLUDEPATH += $$CUDA_DIR/include

    LIBS += -L $$CUDA_DIR/lib64 -lcuda -lcudart

    # There is no need to join() a NVCC_LIBS, use CUDA_LIBS is enough.
    CUDA_LIBS = -lcuda -lcudart

    # The first argument of join() must include all the path that CUDA related,
    # such as OpenCV. Otherwise, nvcc cannot find the dependent library.
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')


    CONFIG(debug, debug|release){
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc \
            -D_DEBUG $$NVCC_OPTIONS \
            $$CUDA_INC \
            $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME} \
            2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2 # Used to output errors
        cuda_d.dependency_type = TYPE_C

        QMAKE_EXTRA_COMPILERS += cuda_d
    }
    else{
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc \
            $$NVCC_OPTIONS \
            $$CUDA_INC \
            $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }
} # unix end


# ============================== Windows ===============================
# Include the library path, for windows
#win32{
#    INCLUDEPATH += \
#        "D:/opencv_cmake/install/include/"

#    CONFIG(debug, debug|release) {
#        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
#            -lopencv_world420d
#    }
#    else:CONFIG(release, debug|release){
#        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
#            -lopencv_world420
#    }
#} # win32 end
# =====================================================================
