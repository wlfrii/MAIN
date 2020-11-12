#ifndef CMD_H
#define CMD_H
#include <string>

struct CMD
{
    // display fps on displayed images
    bool is_show_fps = true;

    // save images
    bool is_take_photo = false;
    std::string pictures_save_path = "./capture";

    // start/close image enhancement
    bool is_enhance = false;


}cmd;

#endif // CMD_H
