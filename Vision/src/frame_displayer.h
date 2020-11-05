#ifndef FRAME_DISPLAYER_H
#define FRAME_DISPLAYER_H
#include "def/define.h"
#include <array>
#include "terminal.h"
#include <string>

/** @brief This class is designed to display the binocular view.
 * Since there needs only one window to show the frames even there is more than one
 * camera, so we designate this class via a single instance mode.
 */
class FrameDisplayer
{
protected:
    FrameDisplayer() {
    }
public:
    friend Terminal;
    ~FrameDisplayer();

    static FrameDisplayer* getInstance();

    /** @brief The interface called by FrameReader, to update frame.
     * @param cam_id  The id of the camera.
     * @param img  The new frame read by FrameReader.
     */
    void updateFrame(cv::Mat &image, uchar cam_id);

    /** @brief The interface called by Main Thread to show the frame.
     */
    void showFrame();

private:
    static FrameDisplayer* instance;

    std::array<cv::Mat, vision::MAX_CAMERA_NUMBER> images;


    /* Terminal operation */
    struct CMD
    {
        // display fps on displayed images
        bool is_show_fps = true;

        // save images
        std::string pictures_save_path = "./capture";
        uint picture_save_num = 1;
        bool is_take_photos = false;

    }cmd;

    void saveImage(const cv::Mat &img);
};

#endif // FRAME_DISPLAYER_H
