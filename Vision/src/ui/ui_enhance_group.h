#ifndef ENHANCEGROUP_H
#define ENHANCEGROUP_H
#include "../def/micro_define.h"
#if LINUX && WITH_QT
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

#endif
#endif // ENHANCEGROUP_H
