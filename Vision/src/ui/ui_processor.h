#ifndef UI_PROCESSOR_H
#define UI_PROCESSOR_H
#include "ui_base.h"
#include <opencv2/opencv.hpp>

class QPushButton;

class UIProcessor : public UILayoutBase
{
	Q_OBJECT
protected:
	UIProcessor();

public:
	~UIProcessor();

	static UIProcessor* getInstance();

	QBoxLayout* create() override;

private slots:
	void onPushBtnLoadImageClicked();
	void onPushBtnLoadVideoClicked();

	void onPushBtnProcessClicked();

private:
	QVBoxLayout		*layout;
	QPushButton		*pbtn_load_image;
	QPushButton		*pbtn_load_video;

	QPushButton		*pbtn_process;

	cv::Mat			image;
	cv::Mat			image_processed;
};

#endif // UI_PROCESSOR_H
