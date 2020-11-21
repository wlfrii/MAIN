#include "ui_processor.h"
#include "ui_logger.h"
#include <gpu_algorithm_pipeline_manager.h>
#include <string>
#include <atomic>
#include <QPushButton>
#include <QFileDialog>
#include <QString>

namespace 
{
	const std::string window_name = "Vision UI Processor --- Original(Left) & Processed (right)";
	int win_width = 1920;
	int win_height = 540;
	void init_cv_window()
	{
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(window_name, win_width, win_height);
		cv::moveWindow(window_name, 0, 0);
	}

	void imshow(cv::Mat &image, cv::Mat &image_processed)
	{
		cv::Mat tmp(win_height, win_width, image.type());
		auto size = cv::Size(win_width / 2, win_height);
		cv::resize(image, tmp.colRange(0, win_width / 2), size);
		cv::resize(image_processed, tmp.colRange(win_width / 2, win_width), size);
		cv::imshow(window_name, tmp);
		cv::waitKey(10);
	}
}

UIProcessor::UIProcessor()
{

}


UIProcessor::~UIProcessor()
{

}


UIProcessor* UIProcessor::getInstance()
{
	static UIProcessor ui_processor;
	return &ui_processor;
}


QBoxLayout* UIProcessor::create()
{
	if (!layout)
	{
		layout = new QVBoxLayout();
		
		pbtn_load_image = new QPushButton();
		pbtn_load_image->setText(tr("Load Image"));
		connect(pbtn_load_image, &QPushButton::clicked, this, &UIProcessor::onPushBtnLoadImageClicked);

		pbtn_process = new QPushButton();
		pbtn_process->setText(tr("Process"));
		pbtn_process->setEnabled(false);
		connect(pbtn_process, &QPushButton::clicked, this, &UIProcessor::onPushBtnProcessClicked);

		QHBoxLayout* hlayout_image = new QHBoxLayout();
		hlayout_image->addWidget(pbtn_load_image);
		hlayout_image->addWidget(pbtn_process);

		pbtn_load_video = new QPushButton();
		pbtn_load_video->setText(tr("Load Video"));
		connect(pbtn_load_video, &QPushButton::clicked, this, &UIProcessor::onPushBtnLoadVideoClicked);

		layout->addLayout(hlayout_image);
		layout->addWidget(pbtn_load_video);
	}

	return layout;
}


void UIProcessor::onPushBtnLoadImageClicked()
{
	static QString image_path;
	image_path = QFileDialog::getOpenFileName(this, tr("File dialog"), image_path.isEmpty() ? "../" : image_path, tr("Image Files(*bmp *png *jpg)"));

	image = cv::imread(image_path.toStdString());
	if (image.empty())
	{
		UILogger::getInstance()->log(QString("Cannot load the specified image."));
		return;
	}
	cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
	::win_width = round(image.cols);
	::win_height = round(image.rows / 2);
	::init_cv_window();
	image.copyTo(image_processed);
	::imshow(image, image_processed);

	UILogger::getInstance()->log(QString("Load an image from \"%1\".").arg(image_path));
	pbtn_process->setEnabled(true);
}


void UIProcessor::onPushBtnLoadVideoClicked()
{

}


void UIProcessor::onPushBtnProcessClicked()
{
	static std::atomic<bool> flag(false);
	bool ret = true;
	while (!flag) {
		gpu::AlgoPipelineManager::getInstance()->process(image, image_processed, flag);
		if (!ret) break;
	}
	if (!ret) {
		UILogger::getInstance()->log(QString("Failed to process the image."));
		return;
	}
	flag.store(false, std::memory_order_relaxed);
	::imshow(image, image_processed);
}