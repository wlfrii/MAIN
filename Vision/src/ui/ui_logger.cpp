#include "ui_logger.h"
#include "../def/micro_define.h"

UILogger::UILogger()
    : txt_browser(new QTextBrowser())
{

}


UILogger::~UILogger()
{
    //DELETE_PIONTER(txt_browser);
}


UILogger *UILogger::getInstance()
{
    static UILogger logger;
    return &logger;
}


QTextBrowser *UILogger::getTxtBrowser()
{
    return txt_browser;
}


void UILogger::log(const QString &qstr)
{
    static unsigned int count = 0;
    QString tmp = QString("#%1. ").arg(++count) + qstr;
    txt_browser->append(tmp);
}

