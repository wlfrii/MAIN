#ifndef LOGER_H
#define LOGER_H
#include "../def/micro_define.h"
#if LINUX && WITH_QT
#include <QWidget>
#include <QString>
#include <QTextBrowser>

class UILogger : public QWidget
{
protected:
    UILogger();

public:
    ~UILogger();
    static UILogger *getInstance();

    QTextBrowser* getTxtBrowser();

    void log(const QString &qstr);

private:
    QTextBrowser *txt_browser;
};

#endif
#endif // LOGER_H
