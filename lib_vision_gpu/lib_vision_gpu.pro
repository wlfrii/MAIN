CONFIG -= qt

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++17

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += $$files("./src/*.cpp", true)
#message("The sources: $$SOURCES")

HEADERS += $$files("./src/*.h", true)
#message("The headers: $$HEADERS")

# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target


# =================  CUDA General  =================
OTHER_FILES += $$files("./src/*.cu", true)
#message("The headers: $$OTHER_FILES")

CUDA_SOURCES+= $$OTHER_FILES

# GPU architecture
SYSTEM_TYPE  = 64
CUDA_ARCH    = sm_75

# -O2 mean to optimization, note its a O not 0
NVCC_OPTIONS = --use_fast_math -O2

# Mandatory flags for stepping through the code
debug{
    NVCC_OPTIONS += -g -G
}

CUDA_OBJECTS_DIR = ./


# ============================== Linux ===============================
# Include the library path, for linux
unix{
    INCLUDEPATH += \
        /usr/local/include/opencv4 \
        /usr/local/include

    LIBS += -L"/usr/local/lib" \
        -lopencv_world


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
win32{
    # ------------------- CUDA --------------------
    CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/"
    CUDA_INC = $$join($$CUDA_DIR/include,'" -I"','-I"','"')

    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64

    INCLUDEPATH += \
        "D:/opencv_cmake/install/include/" \
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include" \
        "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/inc"

    LIBS += -L$$CUDA_DIR/lib/x64 \
        -lcudadevrt \
        -lcudart    \
        -lcudart_static

    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    MSVCRT_LINK_FLAG_DEBUG = "/MDd"  # refers to VS
    MSVCRT_LINK_FLAG_RELEASE = "/MD" # refers to VS

    CONFIG(debug, debug|release) {
        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
            -lopencv_world420d \

        # -- Config complier for CUDA in Debug mode --
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}algorithm.obj # temp file in windows
        cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe \
            -D_DEBUG $$NVCC_OPTIONS \
            $$CUDA_INC \
            $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME} \
            -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
    }
    else:CONFIG(release, debug|release){
        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
         -lopencv_world420 \

        # -- Config complier for CUDA in Release mode --
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}algorithm.obj
        cuda.commands = $$CUDA_DIR/bin/nvcc.exe \
            $$NVCC_OPTIONS \
            $$CUDA_INC $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME} \
            -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }
} # win32 end
