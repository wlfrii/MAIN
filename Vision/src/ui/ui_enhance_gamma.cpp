#include "ui_enhance_gamma.h"
#include "ui_logger.h"
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <libvisiongpu/gpu_algorithm_pipeline_manager.h>

UIEnhanceGamma::UIEnhanceGamma()
{

}


UIEnhanceGamma::~UIEnhanceGamma()
{

}


UIEnhanceGamma* UIEnhanceGamma::getInstance()
{
	static UIEnhanceGamma ui_gamma;
	return &ui_gamma;
}


QBoxLayout* UIEnhanceGamma::create()
{
	if (!hlayout)
	{
		hlayout = new QHBoxLayout();

		chkBox = new QCheckBox();
		chkBox->setText("Gamma");
		chkBox->setMinimumWidth(150);
		chkBox->setChecked(false);
		connect(chkBox, &QCheckBox::stateChanged, this, &UIEnhanceGamma::onChkBoxGammaSelected);

		QLabel *lb_alpha = new QLabel();
		lb_alpha->setText("Alpha: ");

		dspBox_alpha = new QDoubleSpinBox();
		dspBox_alpha->setEnabled(false);
		dspBox_alpha->setMaximum(1);
		dspBox_alpha->setMinimum(0);
		dspBox_alpha->setSingleStep(0.05);
		dspBox_alpha->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		dspBox_alpha->setValue(init.alpha);
		connect(dspBox_alpha, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceGamma::onDoubleSpinBoxAlphaValueChanged);

		QLabel *lb_ref_L = new QLabel();
		lb_ref_L->setText("Ref L: ");

		dspBox_ref_L = new QDoubleSpinBox();
		dspBox_ref_L->setEnabled(false);
		dspBox_ref_L->setMaximum(1);
		dspBox_ref_L->setMinimum(0);
		dspBox_ref_L->setSingleStep(0.1);
		dspBox_ref_L->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		dspBox_ref_L->setValue(init.ref_L);
		connect(dspBox_ref_L, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceGamma::onDoubleSpinBoxRefLValueChanged);

		hlayout->addWidget(chkBox);
		hlayout->addWidget(lb_alpha);
		hlayout->addWidget(dspBox_alpha);
		hlayout->addWidget(lb_ref_L);
		hlayout->addWidget(dspBox_ref_L);
	}
	return hlayout;
}


void UIEnhanceGamma::reset()
{
	dspBox_alpha->setValue(init.alpha);
	dspBox_ref_L->setValue(init.ref_L);
}


void UIEnhanceGamma::setProperty()
{
	float alpha = dspBox_alpha->value();
	float ref_L = dspBox_ref_L->value();

	UILogger::getInstance()->log(QString("Gamma: alpha = %1, reference L = %2").arg(alpha).arg(ref_L));

	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::GammaProperty>(alpha, ref_L));
}


void UIEnhanceGamma::onChkBoxGammaSelected()
{
	if (chkBox->isChecked()) {
		dspBox_alpha->setEnabled(true);
		dspBox_ref_L->setEnabled(true);
		UILogger::getInstance()->log("Open gamma transformation.");
	}
	else {
		dspBox_alpha->setEnabled(false);
		dspBox_ref_L->setEnabled(false);
		UILogger::getInstance()->log("Close gamma transformation.");
	}
}


void UIEnhanceGamma::onDoubleSpinBoxAlphaValueChanged()
{
	setProperty();
}


void UIEnhanceGamma::onDoubleSpinBoxRefLValueChanged()
{
	setProperty();
}