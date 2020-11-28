#ifndef UI_CAPTURE_H
#define UI_CAPTURE_H
#include "ui_base.h"

class QPushButton;
class QSpinBox;
class QLineEdit;

class UICapture : public UILayoutBase
{
protected:
	UICapture();

public:
	~UICapture();

	static UICapture* getInstance();
	QBoxLayout* create() override;

private slots:
	void onPushBtnCaptureClicked();

private:
	QVBoxLayout		*vlayout;
	// Take photos
	QPushButton     *pBtn_capture;
	QSpinBox        *spBox_capture_num;

	QLineEdit       *lEdit_capture_path;
};

#endif // UI_CAPTURE_H
