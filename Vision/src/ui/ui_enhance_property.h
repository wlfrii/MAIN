#ifndef UIENHANCESATURATION_H
#define UIENHANCESATURATION_H
#include <QWidget>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>


class UIEnhanceProperty : public QWidget
{
protected:
    UIEnhanceProperty(QString name);

public:
    virtual ~UIEnhanceProperty();

public:
    QHBoxLayout* create();
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
