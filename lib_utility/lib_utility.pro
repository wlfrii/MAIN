TEMPLATE = lib
TARGET = lib_utility

CONFIG -= qt
CONFIG += staticlib
CONFIG += c++17
CONFIG(debug, debug|release){
    DESTDIR = $$PWD/../build/lib/debug
}else{
    DESTDIR = $$PWD/../build/lib/release
}


SOURCES += $$files("./src/*.cpp", true)
#message("The sources: $$SOURCES")

HEADERS += $$files("./src/*.h", true)
#message("The headers: $$HEADERS")
