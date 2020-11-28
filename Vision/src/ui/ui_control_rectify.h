#ifndef UI_CONTROL_RECTIFY_H
#define UI_CONTROL_RECTIFY_H
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

#endif // UI_CONTROL_RECTIFY_H
