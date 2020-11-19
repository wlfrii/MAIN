#include "control_panel.h"
#include "cmd.h"
#include "ui_logger.h"
#include "ui_capture.h"
#include "ui_control_group.h"
#include "ui_enhance_group.h"
#include "../def/micro_define.h"
#include <QFont>

ControlPanel::ControlPanel(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}


ControlPanel::~ControlPanel()
{
}

void ControlPanel::setupUI()
{
    /** Set the size the widget **/
    this->setWindowTitle("Vision Control Panel");
    this->resize(800, 500);
    /** Set the font style **/
    QFont font;
    font.setFamily(QStringLiteral("Arial"));
    font.setPointSize(9);
    this->setFont(font);
    /** Initialize others **/
    QBoxLayout* hlayout = new QHBoxLayout(this);
    QBoxLayout* vlayout = new QVBoxLayout(this);

    // LOG: Text Browser
    hlayout->addWidget(UILogger::getInstance()->getTxtBrowser());
#if LINUX
    /** CAPTURE **/
	vlayout->addLayout(UICapture::getInstance()->create());
#else
	/** PROCESSOR **/


#endif
    /** CONTROL **/
    vlayout->addWidget(UIControlGroup::getInstance()->create());


    /** ENHANCE Group **/
    vlayout->addWidget(UIEnhanceGroup::getInstance()->create());
    

    hlayout->addLayout(vlayout);

    this->setLayout(hlayout);
}





