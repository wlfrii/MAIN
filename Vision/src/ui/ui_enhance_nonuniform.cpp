#include "ui_enhance_nonuniform.h"
#include "ui_logger.h"
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <gpu_algorithm_pipeline_manager.h>

UIEnhanceNonuniform::UIEnhanceNonuniform()
{

}


UIEnhanceNonuniform::~UIEnhanceNonuniform()
{

}


UIEnhanceNonuniform* UIEnhanceNonuniform::getInstance()
{
	static UIEnhanceNonuniform ui_nonuniform;
	return &ui_nonuniform;
}


QBoxLayout* UIEnhanceNonuniform::create()
{
	if (!layout)
	{
		layout = new QHBoxLayout();

		chkBox = new QCheckBox();
		chkBox->setText("Nonuniform");
		chkBox->setMinimumWidth(150);
		chkBox->setChecked(false);
		connect(chkBox, &QCheckBox::stateChanged, this, &UIEnhanceNonuniform::onChkBoxNonuniformSelected);

		QLabel *lb_magnify = new QLabel();
		lb_magnify->setText("Magnify: ");

		dspBox_magnify = new QDoubleSpinBox();
		dspBox_magnify->setMaximum(3.0);
		dspBox_magnify->setMinimum(1.5);
		dspBox_magnify->setValue(init.magnify);
		dspBox_magnify->setSingleStep(0.1);
		dspBox_magnify->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);	
		dspBox_magnify->setEnabled(false);
		connect(dspBox_magnify, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceNonuniform::onDoubleSpinBoxMagnifyValueChanged);

		QLabel *lb_magnify0 = new QLabel();
		lb_magnify0->setText("Magnify0: ");

		dspBox_magnify0 = new QDoubleSpinBox();
		dspBox_magnify0->setMaximum(1.3);
		dspBox_magnify0->setMinimum(0.7);
		dspBox_magnify0->setValue(init.magnify0);
		dspBox_magnify0->setSingleStep(0.05);
		dspBox_magnify0->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		dspBox_magnify0->setValue(init.magnify0);
		dspBox_magnify0->setEnabled(false);
		connect(dspBox_magnify0, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceNonuniform::onDoubleSpinBoxMagnify0ValueChanged);

		
		layout->addWidget(chkBox);
		layout->addWidget(lb_magnify);
		layout->addWidget(dspBox_magnify);
		layout->addWidget(lb_magnify0);
		layout->addWidget(dspBox_magnify0);
	}
	return layout;
}


void UIEnhanceNonuniform::reset()
{

}


void UIEnhanceNonuniform::setProperty()
{
	float magnify = dspBox_magnify->value();
	float magnify0 = dspBox_magnify0->value();

	UILogger::getInstance()->log(QString("Nonuniform: magnify = %1, magnify0 = %2").arg(magnify).arg(magnify0));

	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::NonuniformProperty>(magnify, magnify0));
}


void UIEnhanceNonuniform::onChkBoxNonuniformSelected()
{
	if (chkBox->isChecked()) {
		dspBox_magnify->setEnabled(true);
		dspBox_magnify0->setEnabled(true);
		UILogger::getInstance()->log("Open nonuniform adjutment.");
	}
	else {
		dspBox_magnify->setEnabled(false);
		dspBox_magnify0->setEnabled(false);
		UILogger::getInstance()->log("Close nonuniform adjutment.");
	}
}


void UIEnhanceNonuniform::onDoubleSpinBoxMagnifyValueChanged()
{
	setProperty();
}


void UIEnhanceNonuniform::onDoubleSpinBoxMagnify0ValueChanged()
{
	setProperty();
}