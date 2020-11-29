#include "frame_displayer.h"
#include "def/micro.h"
#include "def/ptr_define.h"
#include <libutility/timer/mtimer.h>
#include <thread>
#include <sstream>
#if LINUX
#include <stdio.h>  ////@func int access(const char *_Filename, int _AccessMode)
#include <unistd.h>
#include <sys/stat.h> ////@func int mkdir(const char *pathname, mode_t mode), int rmdir(const char *_Path)
#else
#include <io.h>     ////@func int access(const char *_Filename, int _AccessMode)
#include <direct.h> ////@func int mkdir(const char *_Path), int rmdir(const char *_Path)
#endif
#include "./ui/cmd.h"

namespace
{
    const std::string win_name = "Display Window";
    int win_width = vision::SCREEN_WIDTH;
    int win_height = vision::SCREEN_HEIGHT / 2;

    void init_screen_show()
    {
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, win_width, win_height);
        cv::moveWindow(win_name, 0, 0);
        //cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        //cv::setWindowProperty(win_name, cv::WND_PROP_AUTOSIZE, cv::WINDOW_AUTOSIZE);
    }

    auto time_point = mtimer::getDurationSinceEpoch();
    void checkFPS(long long image_tag, uchar id)
    {
        static std::set<long long> frame[2];

        auto now = mtimer::getDurationSinceEpoch();
        if ((now - time_point).count() >= 1e6)
        {
            //printf("camera %d ---> fps: %ld\n", id, frame[id].size());
            time_point = now;
            frame[id].clear();
        }
        else
            if (image_tag > 0)
                frame[id].insert(image_tag);
    }

    int fps = 0;
    void checkFPS(long long image_tag)
    {
        static std::set<long long> frame;

        auto now = mtimer::getDurationSinceEpoch();
        if ((now - time_point).count() >= 1e6)
        {
            //printf("camera ---> fps: %ld\n", frame.size());
            time_point = now;
            fps = frame.size();
            frame.clear();
        }
        else
            if (image_tag > 0)
                frame.insert(image_tag);
    }

}

FrameDisplayer* FrameDisplayer::instance = new FrameDisplayer();

FrameDisplayer::~FrameDisplayer()
{
    DELETE_PIONTER(instance);
}

FrameDisplayer* FrameDisplayer::getInstance()
{
    return instance;
}


void FrameDisplayer::updateFrame(cv::Mat &image, uchar cam_id)
{
    if (cam_id < vision::MAX_CAMERA_NUMBER)
    {
        image.copyTo(images[cam_id]);
//        printf("UpdateFrame\n");
    }
}

void FrameDisplayer::showFrame()
{
    // Because the time interval has been controled in ControlPanel's timer.
    auto start_point = mtimer::getCurrentTimePoint();

    int width = vision::ImageSize::width;
    // Each image processed by GPU will convert to 4 channel !!
    cv::Mat tmp(vision::ImageSize::height, width * 2, CV_8UC4);

    static bool is_start = false;
    if(!is_start)
    {
        is_start = true;
        init_screen_show();
    }

    for(uchar i = 0; i < vision::MAX_CAMERA_NUMBER; i++)
    {
        if(!images[i].empty()){
            images[i].copyTo(tmp.colRange(i*width, (i + 1)*width));
//            cv::imshow("test", images[i]);
        }
    }
    if(!tmp.empty())
    {
        if(CMD::capture.is_take_photo)
            saveImage(tmp);

        cv::Mat out;
        cv::resize(tmp, out, cv::Size(win_width, win_height));   

        if(CMD::is_show_fps)
        {
            // check the fps
            static long long img_count = 0;
            img_count++;
            checkFPS(img_count);
            // put the text on displayed images
            std::stringstream ss;
            ss << ::fps;
            std::string txt = "FPS->" + ss.str();
            cv::putText(out, txt, cv::Point(10, 50)
                        , cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0,0,255,1));
        }

        cv::imshow(win_name, out);
        cv::waitKey(1);
    }

    auto time = mtimer::getDurationSince(start_point);
    if(time < vision::FRAME_UPDATE_INTERVAL_MS)
    {
        auto ms = vision::FRAME_UPDATE_INTERVAL_MS - time;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}

/* Make a dir if the dir is not exist*/
#if LINUX
#define MAKE_DIR(folder) { \
    if(access(folder.c_str(), 0) == -1){ \
        mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); }}
#else
#define MAKE_DIR(folder) { \
    if(_access(folder.c_str(), 0) == -1){ \
        _mkdir(folder.c_str()); }}
#endif

void FrameDisplayer::saveImage(const cv::Mat &img)
{
    auto left_folder = CMD::capture.save_path + "/left/";
    auto right_folder = CMD::capture.save_path + "/right/";
    MAKE_DIR(CMD::capture.save_path);
    MAKE_DIR(left_folder);
    MAKE_DIR(right_folder);

	auto tmp = mtimer::getCurrentTimeStr();

	static uint8_t count = 1;
	char suffix[4];
	if (CMD::capture.save_num > 1) {
		sprintf(suffix, "%d", count);
	}
	int cols = img.cols;
	auto L_path = left_folder + "L_" + CMD::capture.save_name + "-" + std::string(suffix) + ".bmp";
	auto R_path = right_folder + "R_" + CMD::capture.save_name + "-" + std::string(suffix) + ".bmp";
    cv::imwrite(L_path, img.colRange(1, cols/2));
    cv::imwrite(R_path, img.colRange(cols/2, cols));

	if (count >= CMD::capture.save_num) {
		CMD::capture.is_take_photo = false;
		count = 1;
		CMD::capture.finished = true;
	}
	else {
		count++;
	}
}
