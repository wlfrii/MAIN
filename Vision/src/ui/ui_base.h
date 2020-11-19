#ifndef UI_DEFINE_H
#define UI_DEFINE_H
#include <QWidget>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>

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

#endif // UI_DEFINE_H
