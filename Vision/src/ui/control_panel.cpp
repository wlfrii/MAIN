#include "control_panel.h"
#include "../def/micro_define.h"
#if LINUX && WITH_QT
#include "cmd.h"
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
    this->resize(600, 500);
    /** Set the font style **/
    QFont font;
    font.setFamily(QStringLiteral("Arial"));
    font.setPointSize(10);
    this->setFont(font);
    /** Initialize others **/
    vlayout = new QVBoxLayout();

    // CONTROL: FPS checkbox
    ckb_show_fps = new QCheckBox();
    ckb_show_fps->setText(tr("Show FPS"));
    vlayout->addWidget(ckb_show_fps);
    connect(ckb_show_fps, &QCheckBox::stateChanged, this, &ControlPanel::onChkBoxFPSSelected);
    // CONTROL: Enhance
    ckb_enhance = new QCheckBox();
    ckb_enhance->setText(tr("Enhance image"));
    vlayout->addWidget(ckb_enhance);
    connect(ckb_enhance, &QCheckBox::stateChanged, this, &ControlPanel::onChkBoxEnhanceSelected);
    // Enhance Group

    // LOG: Text Browser
    txt_browser = new QTextBrowser();
    vlayout->addWidget(txt_browser);

    this->setLayout(vlayout);
}

void ControlPanel::log(const QString &qstr)
{
    QString tmp = QString("vision: ") + qstr;
    txt_browser->append(tmp);
}

void ControlPanel::onChkBoxFPSSelected()
{
    if(ckb_show_fps->isChecked()){
        log(QString("Show FPS."));
        cmd.is_show_fps = true;
    }
    else{
        log(QString("Close FPS."));
        cmd.is_show_fps = false;
    }
}

void ControlPanel::onChkBoxEnhanceSelected()
{
    if(ckb_enhance->isChecked()){
        log(QString("Enter to image enhancement mode."));
        cmd.is_enhance = true;
    }
    else{
        log(QString("Exit image enhancement mode."));
        cmd.is_enhance = false;
    }
}

#endif
