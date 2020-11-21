#ifndef UIENHANCENONUNIFORM_H
#define UIENHANCENONUNIFORM_H
#include "ui_base.h"

class QCheckBox;
class QDoubleSpinBox;
//class QSpinBox;

class UIEnhanceNonuniform : public UILayoutBase
{
	Q_OBJECT
protected:
	UIEnhanceNonuniform();
public:
	~UIEnhanceNonuniform();

	static UIEnhanceNonuniform* getInstance();

	QBoxLayout* create() override;
	void reset();

private:
	void setProperty();

private slots:
	void onChkBoxNonuniformSelected();
	void onDoubleSpinBoxMagnifyValueChanged();
	void onDoubleSpinBoxMagnify0ValueChanged();

private:
	struct {
		float magnify = 2.0;
		float magnify0 = 1.0;
		int distance = 1000;
	}init;
	QBoxLayout		*layout;

	QCheckBox		*chkBox;
	QDoubleSpinBox	*dspBox_magnify;
	QDoubleSpinBox	*dspBox_magnify0;
	//QSpinBox		*spBox_dis;
};

#endif // UIENHANCENONUNIFORM_H
