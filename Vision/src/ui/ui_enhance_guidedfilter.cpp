#include "ui_enhance_guidedfilter.h"
#include "ui_logger.h"
#include <gpu_algorithm_pipeline_manager.h>
#include <QSlider>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSpacerItem>
#include <QLabel>

UIEnhanceGuidedFilter::UIEnhanceGuidedFilter()
{
	init = { 0.2, 16, 4 };
}

UIEnhanceGuidedFilter::~UIEnhanceGuidedFilter()
{
}

UIEnhanceGuidedFilter* UIEnhanceGuidedFilter::getInstance()
{
	static UIEnhanceGuidedFilter ui_guided_filter;
	return &ui_guided_filter;
}


QBoxLayout* UIEnhanceGuidedFilter::create()
{
	if (!vlayout)
	{
		vlayout = new QVBoxLayout();

		chkBox = new QCheckBox();
		chkBox->setText("Guided Filter");
		chkBox->setChecked(false);
		connect(chkBox, &QCheckBox::stateChanged, this, &UIEnhanceGuidedFilter::onChkBoxGuidedFilterSelected);

		slider = new QSlider();
		slider->setMaximum(100);
		slider->setMinimum(0);
		slider->setValue(0);
		slider->setSingleStep(1);
		slider->setOrientation(Qt::Horizontal);
		slider->setMaximumWidth(200);
		slider->setEnabled(false);
		connect(slider, &QSlider::valueChanged, this, &UIEnhanceGuidedFilter::onSliderValueChanged);

		dspBox_eps = new QDoubleSpinBox();
		dspBox_eps->setMaximum(2);
		dspBox_eps->setMinimum(0);
		dspBox_eps->setValue(0);
		dspBox_eps->setSingleStep(0.02);
		dspBox_eps->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		dspBox_eps->setEnabled(false);
		connect(dspBox_eps, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceGuidedFilter::onDSpinBoxValueChanged);

		QHBoxLayout *hlayout1 = new QHBoxLayout();
		hlayout1->addWidget(chkBox);
		hlayout1->addWidget(slider);
		hlayout1->addWidget(dspBox_eps);

		spacer = new QSpacerItem(50, 18, QSizePolicy::Expanding);
		

		QLabel* lb_radius = new QLabel();
		lb_radius->setText("Filter radius: ");
		spBox_radius = new QSpinBox();
		spBox_radius->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		spBox_radius->setValue(init.radius);

		QLabel* lb_scale = new QLabel();
		lb_scale->setText("Downsampling scale: ");
		spBox_scale = new QSpinBox();
		spBox_scale->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
		spBox_scale->setValue(init.scale);

		QHBoxLayout *hlayout2 = new QHBoxLayout();
		hlayout2->addSpacerItem(spacer);
		hlayout2->addWidget(lb_radius);
		hlayout2->addWidget(spBox_radius);
		hlayout2->addWidget(lb_scale);
		hlayout2->addWidget(spBox_scale);

		vlayout->addLayout(hlayout1);
		vlayout->addLayout(hlayout2);
	}
	return vlayout;
}


void UIEnhanceGuidedFilter::reset()
{
	slider->setValue(50);
	dspBox_eps->setValue(init.eps);
	spBox_radius->setValue(init.radius);
	spBox_scale->setValue(init.scale);
}

void UIEnhanceGuidedFilter::setProperty()
{
	double eps = dspBox_eps->value();
	uint radius = spBox_radius->value();
	uint scale = spBox_scale->value();

	UILogger::getInstance()->log(QString("Guided filter: eps = %1, raidus = %2, scale = %3").arg(eps).arg(radius).arg(scale));

	gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(eps, radius, scale));
}

void UIEnhanceGuidedFilter::onChkBoxGuidedFilterSelected()
{
	if (chkBox->isChecked()) {
		slider->setEnabled(true);
		dspBox_eps->setEnabled(true);
		UILogger::getInstance()->log("Open guided flter.");
	}
	else {
		slider->setEnabled(false);
		dspBox_eps->setEnabled(false);
		UILogger::getInstance()->log("Close guided flter.");
	}
}

void UIEnhanceGuidedFilter::onSliderValueChanged()
{
	int val = slider->value();
	dspBox_eps->setValue(float(val) / 50);
}

void UIEnhanceGuidedFilter::onDSpinBoxValueChanged()
{
	double val = dspBox_eps->value();
	slider->setValue(round(val*50));
	setProperty();
}