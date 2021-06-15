#ifndef MICRO_DEFINE_H
#define MICRO_DEFINE_H


// Switch to select compilation system
#define LINUX                   0   

// Judge whether use Qt GUI
#define WITH_QT                 1   

// Processing image on GPU if 1, otherwise on CPU 
#define WITH_CUDA				0   



#define VIEDO_RECORDER          0   // Video recorder, 1->enable, 0-> disable
#define SHOW_DISPARITY          0   // 1->show disparity in 3d screen, 0-> hide
#define DEBUG_TIME              0   // Show the time-consuming when debug each stage
#define DEBUG_MSG               1   // Shot the debug message like printf, qDebug().

#endif
