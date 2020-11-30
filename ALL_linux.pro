TEMPLATE = subdirs

#CONFIG(debug, debug|release){
#    OBJECTS_DIR = ./build/qqDebug
#}else{
#    OBJECTS_DIR = ./build/qqRelease
#}

SUBDIRS += \
    Vision \
    lib_vision_gpu \
    lib_utility \
    lib_math
