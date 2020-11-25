#include "../def/micro_define.h"
#include <QString>
#include <QGroupBox>
#include <QCheckBox>
#include "cmd.h"
#include "ui_logger.h"
#include "ui_control_group.h"
#include "ui_control_rectify.h"

UIControlGroup::UIControlGroup()
{

}

UIControlGroup::~UIControlGroup()
{
}

UIControlGroup *UIControlGroup::getInstance()
{
	static UIControlGroup control_group;
    return &control_group;
}

QWidget *UIControlGroup::create()
{
	static bool is_init = false;
	if (!is_init) 
	{
		is_init = true;

		gpBox_control = new QGroupBox();
		gpBox_control->setTitle(tr("Control"));
		gpBox_control->setMaximumHeight(88);

		QBoxLayout* vlayout_control = new QVBoxLayout(this);

		// Control Group: FPS
		chkBox_show_fps = new QCheckBox();
		chkBox_show_fps->setText(tr("Show FPS"));
		connect(chkBox_show_fps, &QCheckBox::stateChanged, this, &UIControlGroup::onChkBoxFPSSelected);
		vlayout_control->addWidget(chkBox_show_fps);

		// Control Group: Rectify
		vlayout_control->addLayout(UIControlRectify::getInstance()->create());


		gpBox_control->setLayout(vlayout_control);
	}

    return gpBox_control;
}


void UIControlGroup::onChkBoxFPSSelected()
{
	if (chkBox_show_fps->isChecked()) {
		UILogger::getInstance()->log(QString("Show FPS."));
		CMD::is_show_fps = true;
	}
	else {
		UILogger::getInstance()->log(QString("Close FPS."));
		CMD::is_show_fps = false;
	}
}

