#ifndef TERMINAL_H
#define TERMINAL_H
#include <thread>
#include <string>


class Terminal
{
public:
    Terminal();

    void run();

private:
    void waitInput();
    void processInput(std::string &in);

private:
    std::thread     thread;
};


#endif // TERMINAL_H
