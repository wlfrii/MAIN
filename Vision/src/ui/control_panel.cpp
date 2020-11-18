#include "control_panel.h"
#include "../def/micro_define.h"
#include "cmd.h"
#include <QFont>
#include "../def/micro_define.h"
#include "ui_logger.h"
#include "ui_enhance_group.h"

ControlPanel::ControlPanel(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}


ControlPanel::~ControlPanel()
{
    for(auto label:labels)
        DELETE_PIONTER(label);
    for(auto layout:layouts)
        DELETE_PIONTER(layout);

    DELETE_PIONTER(pBtn_capture);
    DELETE_PIONTER(spBox_capture_num);
    DELETE_PIONTER(lEdit_capture_path);

    DELETE_PIONTER(gpBox_control);
    DELETE_PIONTER(chkBox_show_fps);

    DELETE_PIONTER(gpBox_enhance);
}

void ControlPanel::setupUI()
{
    /** Set the size the widget **/
    this->setWindowTitle("Vision Control Panel");
    this->resize(800, 500);
    /** Set the font style **/
    QFont font;
    font.setFamily(QStringLiteral("Arial"));
    font.setPointSize(10);
    this->setFont(font);
    /** Initialize others **/
    QBoxLayout* hlayout = new QHBoxLayout();
    layouts.push_back(hlayout);
    QBoxLayout* vlayout = new QVBoxLayout();
    layouts.push_back(vlayout);

    // LOG: Text Browser
    hlayout->addWidget(UILogger::getInstance()->getTxtBrowser());

    // ===================================================================
    /** CAPTURE **/
    pBtn_capture = new QPushButton();
    pBtn_capture->setText(tr("Capture Image"));
    connect(pBtn_capture, &QPushButton::clicked, this, &ControlPanel::onPushBtnCaptureClicked);
    QLabel* lb_capture_num = new QLabel();
    labels.push_back(lb_capture_num);
    lb_capture_num->setText(tr("Capture image number:"));
    spBox_capture_num = new QSpinBox();
    spBox_capture_num->setRange(1, 12);
    spBox_capture_num->setValue(1);
    spBox_capture_num->setSingleStep(1);
    spBox_capture_num->setMaximumWidth(50);
    QBoxLayout* hlayout_capture_num = new QHBoxLayout();
    layouts.push_back(hlayout_capture_num);
    hlayout_capture_num->addWidget(lb_capture_num);
    hlayout_capture_num->addWidget(spBox_capture_num);
    hlayout_capture_num->addWidget(pBtn_capture);
    QLabel* lb_capture_path = new QLabel();
    labels.push_back(lb_capture_path);
    lb_capture_path->setText(tr("Save path:"));
    lEdit_capture_path = new QLineEdit();
    lEdit_capture_path->setText("./capture");
    QBoxLayout* hlayout_capture_path = new QHBoxLayout();
    layouts.push_back(hlayout_capture_path);
    hlayout_capture_path->addWidget(lb_capture_path);
    hlayout_capture_path->addWidget(lEdit_capture_path);
    QBoxLayout* vlayout_capture = new QVBoxLayout();
    vlayout_capture->addLayout(hlayout_capture_num);
    vlayout_capture->addLayout(hlayout_capture_path);
    vlayout->addLayout(vlayout_capture);

    // ===================================================================
    /** CONTROL **/
    gpBox_control = new QGroupBox();
    gpBox_control->setTitle(tr("Control"));
    gpBox_control->setMaximumHeight(58);
    QBoxLayout* vlayout_control = new QVBoxLayout();
    layouts.push_back(vlayout_control);
    // FPS
    chkBox_show_fps = new QCheckBox();
    chkBox_show_fps->setText(tr("Show FPS"));
    connect(chkBox_show_fps, &QCheckBox::stateChanged, this, &ControlPanel::onChkBoxFPSSelected);
    vlayout_control->addWidget(chkBox_show_fps);

    gpBox_control->setLayout(vlayout_control);
    vlayout->addWidget(gpBox_control);

    // ===================================================================
    /** ENHANCE **/
    gpBox_enhance = UIEnhanceGroup::getInstance()->create();
    vlayout->addWidget(gpBox_enhance);
    connect(gpBox_enhance, &QGroupBox::clicked, this, &ControlPanel::onGroupBoxEnhanceSelected);

    hlayout->addLayout(vlayout);

    this->setLayout(hlayout);
}


/***************  SLOTS  *****************/
void ControlPanel::onChkBoxFPSSelected()
{
    if(chkBox_show_fps->isChecked()){
        UILogger::getInstance()->log(QString("Show FPS."));
        CMD::is_show_fps = true;
    }
    else{
        UILogger::getInstance()->log(QString("Close FPS."));
        CMD::is_show_fps = false;
    }
}


void ControlPanel::onPushBtnCaptureClicked()
{
    int num = spBox_capture_num->value();
    QString path = lEdit_capture_path->text();

    UILogger::getInstance()->log(QString("Capturing %1 image...").arg(num));
}


void ControlPanel::onGroupBoxEnhanceSelected()
{
    if(gpBox_enhance->isChecked()){
        UILogger::getInstance()->log(QString("Open to image enhancement mode."));
        CMD::is_enhance = true;
    }
    else{
        UILogger::getInstance()->log(QString("Close image enhancement mode."));
        CMD::is_enhance = false;
    }
}

