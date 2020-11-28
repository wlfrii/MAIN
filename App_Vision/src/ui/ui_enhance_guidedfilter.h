#ifndef UI_ENHANCE_GUIDEDFILTER_H
#define UI_ENHANCE_GUIDEDFILTER_H
#include "ui_base.h"

class QCheckBox;
class QSlider;
class QDoubleSpinBox;
class QSpinBox;
class QSpacerItem;

class UIEnhanceGuidedFilter : public UILayoutBase
{
	Q_OBJECT

protected:
    UIEnhanceGuidedFilter();

public:
	~UIEnhanceGuidedFilter();
	static UIEnhanceGuidedFilter* getInstance();
	
	QBoxLayout* create() override;
	void reset();

private:
	void setProperty();

private slots:
	void onChkBoxGuidedFilterSelected();
	void onSliderValueChanged();
	void onDSpinBoxValueChanged();

private:
	struct {
		float eps;
		uchar radius;
		uchar scale;
	}init;
	QVBoxLayout		*vlayout;

	QCheckBox		*chkBox;
	QSlider			*slider;
	QDoubleSpinBox	*dspBox_eps;
	QSpinBox		*spBox_radius;
	QSpinBox		*spBox_scale;
	QSpacerItem		*spacer;
};

#endif // UI_ENHANCE_GUIDEDFILTER_H
