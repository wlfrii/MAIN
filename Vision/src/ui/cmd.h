#ifndef CMD_H
#define CMD_H
#include <string>
#include <cstdint>

class CMD
{
	struct Capture {
		bool        is_take_photo;
		uint8_t     save_num;
		std::string save_name;
		std::string save_path;
		
		bool		finished; // check whether saved all the images
	};

public:
    // display fps on displayed images
    static bool is_show_fps;

    // save images
	static Capture capture;

    // start/close image enhancement
    static bool is_enhance;
};

#endif // CMD_H
