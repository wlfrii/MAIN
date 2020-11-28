#include "ui_logger.h"
#include <QFont>

UILogger::UILogger()
{

}


UILogger::~UILogger()
{
}


UILogger *UILogger::getInstance()
{
    static UILogger logger;
    return &logger;
}


QTextBrowser *UILogger::getTxtBrowser()
{
	if (!txt_browser)
	{
		txt_browser = new QTextBrowser();

		/** Set the font style **/
		QFont font;
		font.setFamily(QStringLiteral("Arial"));
		font.setPointSize(9);
		font.setItalic(true);
		txt_browser->setFont(font);
		txt_browser->setStyleSheet(QString("color:rgb(255,85,0);background-color:rgb(220,220,220)"));
	}
    return txt_browser;
}


void UILogger::log(const QString &qstr)
{
    static unsigned int count = 0;
    QString tmp = QString("#%1. ").arg(++count) + qstr;
    txt_browser->append(tmp);
}

