#ifndef UIENHANCEGAMMA_H
#define UIENHANCEGAMMA_H
#include "ui_base.h"

class QCheckBox;
class QDoubleSpinBox;

class UIEnhanceGamma : public UILayoutBase
{
	Q_OBJECT
protected:
    UIEnhanceGamma();
public:
	~UIEnhanceGamma();

	static UIEnhanceGamma* getInstance();

	QBoxLayout* create() override;
	void reset();

private:
	void setProperty();

private slots:
	void onChkBoxGammaSelected();
	void onDoubleSpinBoxAlphaValueChanged();
	void onDoubleSpinBoxRefLValueChanged();

private:
	struct {
		float alpha = 0.55;
		float ref_L = 0.5;
	}init;
	QHBoxLayout		*hlayout;

	QCheckBox		*chkBox;
	QDoubleSpinBox	*dspBox_alpha;
	QDoubleSpinBox	*dspBox_ref_L;
};

#endif // UIENHANCEGAMMA_H
