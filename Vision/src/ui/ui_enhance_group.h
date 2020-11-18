#ifndef UIENHANCEGROUP_H
#define UIENHANCEGROUP_H
#include "../def/micro_define.h"
#include <QGroupBox>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <QBoxLayout>
#include <QWidget>


class UIEnhanceGroup : public QWidget
{
    Q_OBJECT

protected:
    UIEnhanceGroup();
public:
    static UIEnhanceGroup *getInstance();
    QGroupBox *create();

private slots:
    void onChkBoxRectifySelected();

private:
    std::vector<QLabel*> labels;
    std::vector<QBoxLayout*> layouts;

    QGroupBox       *gpBox_enhance;
    QCheckBox       *chkBox_rectify;

};

#endif // ENHANCEGROUP_H
