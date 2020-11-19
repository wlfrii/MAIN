#ifndef UIENHANCEGROUP_H
#define UIENHANCEGROUP_H
#include "ui_base.h"

class QGroupBox;
class QCheckBox;

class UIEnhanceGroup : public UIWidgetBase
{
    Q_OBJECT

protected:
    UIEnhanceGroup();
public:
	~UIEnhanceGroup();

    static UIEnhanceGroup *getInstance();
    QWidget* create() override;

private slots:
    void onChkBoxRectifySelected();
	void onGroupBoxEnhanceSelected();

private:
    QGroupBox       *gpBox_enhance;
    QCheckBox       *chkBox_rectify;
};

#endif // ENHANCEGROUP_H
