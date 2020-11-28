#include "ui_enhance_property.h"
#include "ui_logger.h"
#include <gpu_algorithm_pipeline_manager.h>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>

UIEnhanceProperty::UIEnhanceProperty(QString name)
    : name(name)
{

}

UIEnhanceProperty::~UIEnhanceProperty()
{
}

QBoxLayout *UIEnhanceProperty::create()
{
    if(!hlayout)
    {
        hlayout = new QHBoxLayout();

        chkBox = new QCheckBox();
        chkBox->setText(name);
        chkBox->setChecked(false);
        connect(chkBox, &QCheckBox::stateChanged, this, &UIEnhanceProperty::onChkBoxSaturationSelected);

        slider = new QSlider();
        slider->setMaximum(100);
        slider->setMinimum(0);
        slider->setValue(50);
        slider->setSingleStep(1);
        slider->setOrientation(Qt::Horizontal);
        slider->setMaximumWidth(200);
        slider->setEnabled(false);
        connect(slider, &QSlider::valueChanged, this, &UIEnhanceProperty::onSliderValueChanged);

        spBox = new QSpinBox();
        spBox->setMaximum(100);
        spBox->setMinimum(0);
        spBox->setValue(50);
        spBox->setSingleStep(1);
        spBox->setMaximumWidth(UI_SPINBOX_MAX_WIDTH);
        spBox->setEnabled(false);
        connect(spBox, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &UIEnhanceProperty::onSpinBoxValueChanged);

        hlayout->addWidget(chkBox);
        hlayout->addWidget(slider);
        hlayout->addWidget(spBox);
    }
    return hlayout;
}

void UIEnhanceProperty::reset()
{
    slider->setValue(50);
    spBox->setValue(50);
}

void UIEnhanceProperty::setProperty(int value)
{
    static char satur = 50, contr = 50, bright = 50;
    if(name == "Saturation"){
        satur = value;
    }else if(name == "Contrast"){
        contr = value;
    }else{ // name = brightness
        bright = value;
    }
    UILogger::getInstance()->log(name + QString(" set to %1").arg(value));
    gpu::AlgoPipelineManager::getInstance()->setProperty(std::make_shared<gpu::ImageAdjustProperty>(satur, contr, bright));
}

void UIEnhanceProperty::onChkBoxSaturationSelected()
{
    if(chkBox->isChecked()){
        slider->setEnabled(true);
        spBox->setEnabled(true);
        UILogger::getInstance()->log("Open " + name + ".");
    }
    else{
        slider->setEnabled(false);
        spBox->setEnabled(false);
        UILogger::getInstance()->log("Close " + name + ".");
    }
}

void UIEnhanceProperty::onSliderValueChanged()
{
    int val = slider->value();
    spBox->setValue(val);
}

void UIEnhanceProperty::onSpinBoxValueChanged()
{
    int val = spBox->value();
    slider->setValue(val);
    UIEnhanceProperty::setProperty(val);
}


UIEnhanceSaturation* UIEnhanceSaturation::getInstance()
{
    static UIEnhanceSaturation ui_saturation;
    return &ui_saturation;
}

UIEnhanceContrast* UIEnhanceContrast::getInstance()
{
    static UIEnhanceContrast ui_contrast;
    return &ui_contrast;
}

UIEnhanceBrightness* UIEnhanceBrightness::getInstance()
{
    static UIEnhanceBrightness ui_brightness;
    return &ui_brightness;
}
