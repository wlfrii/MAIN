#ifndef UI_ENHANCE_GROUP_H
#define UI_ENHANCE_GROUP_H
#include "ui_base.h"

class QGroupBox;

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
	void onGroupBoxEnhanceSelected();

private:
	QGroupBox	*gpBox_enhance;
};

#endif // UI_ENHANCE_GROUP_H
