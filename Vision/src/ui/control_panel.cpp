#include "control_panel.h"
#include "cmd.h"
#include "ui_logger.h"
#include "ui_capture.h"
#include "ui_control_group.h"
#include "ui_enhance_group.h"
#include "ui_processor.h"
#include <QFont>
#include "../def/micro.h"
#include "../frame_displayer.h"

ControlPanel::ControlPanel(QWidget *parent)
    : QWidget(parent)
    , timer_imshow(new QTimer)
{
    setupUI();

    connect(timer_imshow, &QTimer::timeout, this, &ControlPanel::onTimerImshow);
    timer_imshow->start(3);
}


ControlPanel::~ControlPanel()
{
}

void ControlPanel::setupUI()
{
    /** Set the size the widget **/
    this->setWindowTitle("Vision Control Panel");
    this->resize(900, 500);
    /** Set the font style **/
    QFont font;
    font.setFamily(QStringLiteral("Arial"));
    font.setPointSize(9);
    this->setFont(font);
	//this->setStyleSheet(QString("background-color:rgb(30,30,30);color:rgb(240,240,240);"));
	
    
	/** Initialize others **/
    QBoxLayout* hlayout = new QHBoxLayout(this);
    QBoxLayout* vlayout = new QVBoxLayout(this);
	vlayout->setSizeConstraint(vlayout->SetFixedSize);

    // LOG: Text Browser
    hlayout->addWidget(UILogger::getInstance()->getTxtBrowser());
#if LINUX
    /** CAPTURE **/
	vlayout->addLayout(UICapture::getInstance()->create());
#else
	/** PROCESSOR **/
	vlayout->addLayout(UIProcessor::getInstance()->create());
#endif
    /** CONTROL **/
    vlayout->addWidget(UIControlGroup::getInstance()->create());

    /** ENHANCE **/
    vlayout->addWidget(UIEnhanceGroup::getInstance()->create());
    

    hlayout->addLayout(vlayout);

    this->setLayout(hlayout);
}

void ControlPanel::onTimerImshow()
{
    FrameDisplayer::getInstance()->showFrame();
}





