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


# =======================================================================
# =                            CUDA General                             =
# =======================================================================

# Set CUDA sources
OTHER_FILES += $$files("./src/*.cu", true)
CUDA_SOURCES+= $$OTHER_FILES
#message("CUDA_SOURCESs: $$CUDA_SOURCES")

# GPU architecture
SYSTEM_TYPE  = 64
CUDA_ARCH    = sm_75
#CUDA_CODE    = compute_75

# -O2 mean to optimization, note its a O not 0
NVCC_OPTIONS = --use_fast_math -O2


## Mandatory flags for stepping through the code
debug{
    NVCC_OPTIONS += -g -G
}


CUDA_OBJECTS_DIR = ./cuda_obj

unix:message("unix: Target compilation platform: $$QMAKE_HOST.arch")
unix:INCLUDEPATH += /usr/local/include/opencv4
unix:LIBS += -L"/usr/local/lib" -lopencv_world

# Path to cuda SDK install
unix:CUDA_DIR = "/usr/local/cuda-11.0"

# CUDA include paths
INCLUDEPATH += $$CUDA_DIR/include
# The first argument of join() must include all the path that CUDA related,
# such as OpenCV. Otherwise, nvcc cannot find the dependent library.
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')


# CUDA libs
unix:QMAKE_LIBDIR += $$CUDA_DIR/lib64
unix:LIBS += -L $$CUDA_DIR/lib64 -lcuda -lcudart
# CUDA compile
unix{
    CONFIG(debug, debug|release){
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc \
            -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME} \
            2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2 # Used to output errors
        cuda_d.dependency_type = TYPE_C

        QMAKE_EXTRA_COMPILERS += cuda_d
    }else{
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc \
            $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS \
            --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH \
            -c -o ${QMAKE_FILE_OUT} \
            ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }
} # unix end

###################################################################################################
#GENCODE = arch=compute_75,code=sm_75

## Path to CUDA SDK intalled
#win32:CUDA_DIR = "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"
## Path to CUDA toolkit installed
#win32:CUDA_SDK = "c:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0"

## CUDA include paths
#INCLUDEPATH += $$CUDA_DIR/include
#win32:INCLUDEPATH += $$CUDA_SDK/common/inc
#win32:INCLUDEPATH += "D:/opencv_cmake/install/include"

## CUDA libs
#win32:QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
#win32:QMAKE_LIBDIR += $$CUDA_SDK/common/lib/x64

##Join the includes in a line
#CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

## NVCC flags (ptxas option verbose is always useful),   -O2 mean to optimization, note its a O not 0
#NVCC_OPTIONS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math -O2

## On windows we must define if we are in debug mode or not
#CONFIG(debug, debug|release) {
#    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
#    win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
#    win32:NVCC_OPTIONS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
#}
#else{
#    win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
#    win32:NVCC_OPTIONS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
#}

#message("CUDA_DIR: $$CUDA_DIR")
#message("CUDA_SDK: $$CUDA_SDK")
#message("CUDA_INC: $$CUDA_INC")
#message("QMAKE_LIBDIR: $$QMAKE_LIBDIR")

## Prepare intermediate cuda compiler
#cuda_intr.input = CUDA_SOURCES
#cuda_intr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
## So in windows object files have to be named with the .obj suffix instead of just .o
## God I hate you windows!!
#win32:cuda_intr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

### Tweak arch according to your hw's compute capability
#cuda_intr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc \
#                    $$NVCC_OPTIONS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

## Set our variable out. These obj files need to be used to create the link obj file
## and used in our final gcc compilation
#cuda_intr.variable_out = CUDA_OBJ
#cuda_intr.variable_out += OBJECTS
#cuda_intr.clean = cuda_intr_obj/*.o
#win32:cuda_intr.clean = cuda_intr_obj/*.obj

#QMAKE_EXTRA_UNIX_COMPILERS += cuda_intr

## Prepare the linking compiler step
#cuda.input = CUDA_OBJ
#cuda.output = ${QMAKE_FILE_BASE}_link.o
#win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj

## Tweak arch according to your hw's compute capability
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCC_OPTIONS  ${QMAKE_FILE_NAME}
## Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_UNIX_COMPILERS += cuda
###################################################################################################

## ============================== Windows ===============================
## Include the library path, for windows
#win32{
#    win32:DEFINES += _CRT_SECURE_NO_WARNINGS

#    message("win32: Target compilation platform: $$QMAKE_HOST.arch")

#    # ------------------- CUDA --------------------
#    CUDA_DIR = "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"
##    message("CUDA_DIR:  $$CUDA_DIR")
#    CUDA_SDK = "d:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0"

#    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64

#    win32:INCLUDEPATH += \
#        "d:/opencv_cmake/install/include" \
#        "$$CUDA_DIR/include" \
#        "d:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/inc"

#    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#    message("CUDA_INC: $$CUDA_INC")

#    LIBS += -L$$CUDA_DIR/lib/x64 \
#        -lcudadevrt \
#        -lcudart    \
#        -lcudart_static

#    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
#    MSVCRT_LINK_FLAG_DEBUG = /NODEFAULTLIB:msvcrt.lib #"/MDd"  # refers to VS
#    MSVCRT_LINK_FLAG_RELEASE = /NODEFAULTLIB:msvcrtd.lib #"/MD" # refers to VS

#    win32:CONFIG(debug, debug|release) {
#        LIBS += -L"D:/opencv_cmake/install/x64/vc15/lib" \
#            -lopencv_world420d

#        #CUDA_LIBS += opencv_world420d.lib cudart.lib cudadevrt.lib

#        message("CUDA_LIBS: $$CUDA_LIBS")
#        # -- Config complier for CUDA in Debug mode --
#        cuda_d.input = CUDA_SOURCES
#        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}algorithm.obj # temp file in windows
#        cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe \
#            -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS \
#            --machine $$SYSTEM_TYPE \
#            -arch=$$CUDA_ARCH \
#            -c -o ${QMAKE_FILE_OUT} \
#            ${QMAKE_FILE_NAME} \
#            -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
#        cuda_d.dependency_type = TYPE_C
#        QMAKE_EXTRA_COMPILERS += cuda_d
#    }
#    else:CONFIG(release, debug|release){
#        LIBS += -L"d:/opencv_cmake/install/x64/vc15/lib" \
#            -lopencv_world420

#        CUDA_LIBS +=$$LIBS
#        message("CUDA_LIBS: $$CUDA_LIBS")
#        # -- Config complier for CUDA in Release mode --
#        cuda.input = CUDA_SOURCES
#        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}algorithm.obj
#        cuda.commands = $$CUDA_DIR/bin/nvcc.exe \
#            $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS \
#            --machine $$SYSTEM_TYPE \
#            -arch=$$CUDA_ARCH \
#            -c -o ${QMAKE_FILE_OUT} \
#            ${QMAKE_FILE_NAME} \
#            -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
#        cuda.dependency_type = TYPE_C
#        QMAKE_EXTRA_COMPILERS += cuda
#    }
#} # win32 end
