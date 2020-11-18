#ifndef UIENHANCEGUIDEDFILTER_H
#define UIENHANCEGUIDEDFILTER_H
#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QVBoxLayout>

class UIEnhanceGuidedFilter : public QWidget
{
protected:
    UIEnhanceGuidedFilter();

public:
	~UIEnhanceGuidedFilter();
	static UIEnhanceGuidedFilter* getInstance();
	
	QHBoxLayout* create();
	void reset();

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
	QHBoxLayout		*hlayout;
	QCheckBox		*chkBox;
	QSlider			*slider;
	QDoubleSpinBox	*dspBox_eps;
	QLabel			*lb_radius;
	QLabel			*lb_scale;
	QSpinBox		*spBox_radius;
	QSpinBox		*spBox_scale;

	std::vector<QBoxLayout*> layouts;
};

#endif // UIENHANCEGUIDEDFILTER_H
