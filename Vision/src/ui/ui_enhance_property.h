#ifndef UIENHANCESATURATION_H
#define UIENHANCESATURATION_H
#include "ui_base.h"

class QCheckBox;
class QSlider;
class QSpinBox;

class UIEnhanceProperty : public UILayoutBase
{
protected:
    UIEnhanceProperty(QString name);

public:
    virtual ~UIEnhanceProperty();

public:
    QBoxLayout* create() override;
    void reset();

protected:
    void setProperty(int value);

private slots:
    void onChkBoxSaturationSelected();
    void onSliderValueChanged();
    void onSpinBoxValueChanged();

private:
    QString          name;
    QHBoxLayout     *hlayout;

    QCheckBox       *chkBox;
    QSlider         *slider;
    QSpinBox        *spBox;
};


class UIEnhanceSaturation : public UIEnhanceProperty
{
protected:
    UIEnhanceSaturation()
        : UIEnhanceProperty("Saturation")
    {}

public:
    ~UIEnhanceSaturation(){}

    static UIEnhanceSaturation* getInstance();
};

class UIEnhanceContrast : public UIEnhanceProperty
{
protected:
    UIEnhanceContrast()
        : UIEnhanceProperty("Contrast")
    {}
public:
    ~UIEnhanceContrast() {}

    static UIEnhanceContrast* getInstance();
};

class UIEnhanceBrightness : public UIEnhanceProperty
{
protected:
    UIEnhanceBrightness()
        : UIEnhanceProperty("Brightness")
    {}
public:
    ~UIEnhanceBrightness() {}

    static UIEnhanceBrightness* getInstance();
};

#endif // UIENHANCESATURATION_H
