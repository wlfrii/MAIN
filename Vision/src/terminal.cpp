#include "terminal.h"
#include <iostream>
#include "frame_displayer.h"

Terminal::Terminal()
{

}

void Terminal::run()
{
    thread = std::thread(&Terminal::waitInput, this);
    thread.detach();
}


void Terminal::waitInput()
{
    std::string in;
    while(true)
    {
        std::cout << "wanglf@Vision:> ";
        getline(std::cin, in);
        processInput(in);
    }
}

void Terminal::processInput(std::string &in)
{
    if(in == "fps on")
        FrameDisplayer::getInstance()->cmd.is_show_fps = true;
    else if(in == "fps off")
        FrameDisplayer::getInstance()->cmd.is_show_fps = false;
    else if(in == "take a photo" || in == " ")
        FrameDisplayer::getInstance()->cmd.is_take_photos = true;
    else {
        //printf("Wrong command.\n");
        return;
    }
}
