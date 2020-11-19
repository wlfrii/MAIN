#ifndef UICONTROLGROUP_H
#define UICONTROLGROUP_H
#include "ui_base.h"

class QGroupBox;
class QCheckBox;

class UIControlGroup : public UIWidgetBase
{
    Q_OBJECT

protected:
	UIControlGroup();
public:
	~UIControlGroup();

    static UIControlGroup *getInstance();
    QWidget* create() override;

private slots:
	void onChkBoxFPSSelected();

private:
    QGroupBox       *gpBox_control;
    QCheckBox       *chkBox_show_fps;
};

#endif // UICONTROLGROUP_H
