#ifndef UI_BASE_H
#define UI_BASE_H
#include <QWidget>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>


constexpr unsigned int UI_SPINBOX_MAX_WIDTH = 50;
constexpr unsigned int UI_CHECKBOX_WIDTH = 150;

class UIBase : public QWidget
{
	Q_OBJECT
protected:
	UIBase(QWidget *parent = nullptr)
		: QWidget(parent)
	{}

public:
	virtual ~UIBase() {}
};


class UIWidgetBase : public UIBase
{
	Q_OBJECT
protected:
	UIWidgetBase(QWidget *parent = nullptr)
		: UIBase(parent)
	{}

public:
	virtual ~UIWidgetBase(){}

	virtual QWidget* create() = 0;
};


class UILayoutBase : public UIBase
{
	Q_OBJECT
protected:
	UILayoutBase(QWidget *parent = nullptr)
		: UIBase(parent)
	{}

public:
	virtual ~UILayoutBase() {}

	virtual QBoxLayout* create() = 0;
};

#endif // UI_BASE_H
