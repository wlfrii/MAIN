#include "ui_enhance_guidedfilter.h"
#include "../def/micro_define.h"

UIEnhanceGuidedFilter::UIEnhanceGuidedFilter()
{
	init = { 0.2, 16, 4 };
}

UIEnhanceGuidedFilter::~UIEnhanceGuidedFilter()
{

}

UIEnhanceGuidedFilter* UIEnhanceGuidedFilter::getInstance()
{
	UIEnhanceGuidedFilter ui_guided_filter;
	return &ui_guided_filter;
}


QHBoxLayout* UIEnhanceGuidedFilter::create()
{
	if (!hlayout)
	{
		hlayout = new QHBoxLayout();

		chkBox = new QCheckBox();
		chkBox->setText("Guided Filter");
		chkBox->setChecked(false);
		connect(chkBox, &QCheckBox::stateChanged, this, &UIEnhanceGuidedFilter::onChkBoxGuidedFilterSelected);

		slider = new QSlider();
		slider->setMaximum(100);
		slider->setMinimum(0);
		slider->setValue(0);
		slider->setSingleStep(1);
		slider->setMaximumWidth(200);
		slider->setEnabled(false);
		connect(slider, &QSlider::valueChanged, this, &UIEnhanceGuidedFilter::onSliderValueChanged);

		dspBox_eps = new QDoubleSpinBox();
		dspBox_eps->setMaximum(2);
		dspBox_eps->setMinimum(0);
		dspBox_eps->setValue(0);
		dspBox_eps->setSingleStep(0.1);
		dspBox_eps->setMaximumWidth(50);
		dspBox_eps->setEnabled(false);
		connect(dspBox_eps, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &UIEnhanceGuidedFilter::onDSpinBoxValueChanged);

		QHBoxLayout *hlayout1 = new QHBoxLayout();
		hlayout1->addWidget(slider);
		hlayout1->addWidget(dspBox_eps);
		layouts.push_back(hlayout1);

		lb_radius = new QLabel();
		lb_radius->setText("Filter radius: ");
		spBox_radius = new QSpinBox();
		spBox_radius->setValue(init.radius);
		lb_scale = new QLabel();
		lb_scale->setText("Downsampling scale: ");
		spBox_scale = new QSpinBox();
		spBox_scale->setValue(init.scale);

		QHBoxLayout *hlayout2 = new QHBoxLayout();
		hlayout2->addWidget(lb_radius);
		hlayout2->addWidget(spBox_radius);
		hlayout2->addWidget(lb_scale);
		hlayout2->addWidget(spBox_scale);
		layouts.push_back(hlayout2);

		QVBoxLayout *vlayout1 = new QVBoxLayout();
		layouts.push_back(vlayout1);
		vlayout1->addLayout(hlayout1);
		vlayout1->addLayout(hlayout2);

		hlayout->addWidget(chkBox);
		hlayout->addLayout(vlayout1);
	}
	return hlayout;
}


void UIEnhanceGuidedFilter::reset()
{
	slider->setValue(50);
	dspBox_eps->setValue(init.eps);
	spBox_radius->setValue(init.radius);
	spBox_scale->setValue(init.scale);
}

void UIEnhanceGuidedFilter::onChkBoxGuidedFilterSelected()
{

}

void UIEnhanceGuidedFilter::onSliderValueChanged()
{

}

void UIEnhanceGuidedFilter::onDSpinBoxValueChanged()
{

}