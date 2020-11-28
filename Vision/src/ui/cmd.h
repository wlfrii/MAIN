#ifndef CMD_H
#define CMD_H
#include <string>


class CMD
{
public:
    // display fps on displayed images
    static bool is_show_fps;

    // save images
    static bool is_take_photo;
    static std::string pictures_save_path;

    // start/close image enhancement
    static bool is_enhance;
};

#endif // CMD_H
