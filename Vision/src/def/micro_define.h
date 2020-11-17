#ifndef MICRO_DEFINE_H
#define MICRO_DEFINE_H

#define LINUX                   1   // Switch to select compilation system
#define WITH_QT                 1   // Flag for whether there is a Qt, so that UI could be supported

#define VIEDO_RECORDER          0   // Video recorder, 1->enable, 0-> disable
#define SHOW_DISPARITY          0   // 1->show disparity in 3d screen, 0-> hide
#define DEBUG_TIME              0   // Show the time-consuming when debug each stage
#define DEBUG_MSG               1   // Shot the debug message like printf, qDebug().

/* Delete a pointer */
#define DELETE_PIONTER(ptr)	\
	if(ptr != nullptr) {	\
		delete ptr;			\
		ptr = nullptr;		\
	}
/* Delete an array of pointer */
#define DELETE_ARRAY_PIONTER(ptr)	\
    if(ptr != nullptr) {            \
        delete[] ptr;               \
        ptr = nullptr;              \
    }



#endif
