#ifndef FRAME_DISPLAYER_H
#define FRAME_DISPLAYER_H
#include "def/define.h"
#include <array>
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
    void saveImage(const cv::Mat &img);

private:
    std::array<cv::Mat, vision::MAX_CAMERA_NUMBER> images;
};

#endif // FRAME_DISPLAYER_H
