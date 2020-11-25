#ifndef UICONTROLRECTIFY_H
#define UICONTROLRECTIFY_H
#include "ui_base.h"

class QCheckBox;
class QPushButton;

class UIControlRectify : public UILayoutBase
{
	Q_OBJECT
protected:
	UIControlRectify();
public:
	~UIControlRectify();

	static UIControlRectify* getInstance();

	QBoxLayout* create() override;

private:
	void setProperty();

private slots:
	void onChkBoxRectifySelected();
	void onPushBtnCamParamClicked();
	void onPushBtnViewCamParamClicked();

private:
	QHBoxLayout		*layout;
	QCheckBox       *chkBox;
	QPushButton		*pbtn_cam_param;
	QPushButton		*pbtn_view_cam_param;
};

#endif // UICONTROLRECTIFY_H
