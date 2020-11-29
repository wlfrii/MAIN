#include "ui_control_rectify.h"
#include "ui_logger.h"
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <gpu_algorithm_pipeline_manager.h>

UIControlRectify::UIControlRectify()
{

}


UIControlRectify::~UIControlRectify()
{

}


UIControlRectify* UIControlRectify::getInstance()
{
	static UIControlRectify ui_rectify;
	return &ui_rectify;
}


QBoxLayout* UIControlRectify::create()
{
	if (!layout)
	{
		layout = new QHBoxLayout();

		chkBox = new QCheckBox();
		chkBox->setText(tr("Rectification"));
		chkBox->setChecked(false);
		connect(chkBox, &QCheckBox::stateChanged, this, &UIControlRectify::onChkBoxRectifySelected);
		
		pbtn_cam_param = new QPushButton();
		pbtn_cam_param->setText(tr("Load CamParams"));
		connect(pbtn_cam_param, &QPushButton::clicked, this, &UIControlRectify::onPushBtnCamParamClicked);
		pbtn_cam_param->setEnabled(false);

		pbtn_view_cam_param = new QPushButton();
		pbtn_view_cam_param->setText(tr("View CamParams"));
		connect(pbtn_view_cam_param, &QPushButton::clicked, this, &UIControlRectify::onPushBtnViewCamParamClicked);
		pbtn_view_cam_param->setEnabled(false);

		layout->addWidget(chkBox);
		layout->addWidget(pbtn_cam_param);
		layout->addWidget(pbtn_view_cam_param);
	}
	return layout;
}


void UIControlRectify::setProperty()
{
	
	UILogger::getInstance()->log(QString("Rectify: set map."));

	//gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::NonuniformProperty>(magnify, magnify0));
}


void UIControlRectify::onChkBoxRectifySelected()
{
	if (chkBox->isChecked())
	{
		pbtn_cam_param->setEnabled(true);
		pbtn_view_cam_param->setEnabled(true);
		UILogger::getInstance()->log(QString("Open rectification"));
	}
	else
	{
		pbtn_cam_param->setEnabled(false);
		pbtn_view_cam_param->setEnabled(false);
		UILogger::getInstance()->log(QString("Close rectification"));
	}
}


void UIControlRectify::onPushBtnCamParamClicked()
{
	UILogger::getInstance()->log(QString("TO DO"));
}


void UIControlRectify::onPushBtnViewCamParamClicked()
{
	UILogger::getInstance()->log(QString("TO DO"));
}