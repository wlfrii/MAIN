#ifndef UI_LOGER_H
#define UI_LOGER_H
#include <QWidget>
#include <QString>
#include <QTextBrowser>

class UILogger : public QWidget
{
	Q_OBJECT
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

#endif // UI_LOGER_H
