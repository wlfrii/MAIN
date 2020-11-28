#pragma once

template<typename Tp> 
struct Placement
{
	Placement& operator=(const Tp& rhs)
	{
		assignBy(rhs);
		return *this;
	}

	void* alloc()		{ return (void*)&u; }
	void  free()		{ getObject()->~Tp(); }
	void  destory()		{ getObject()->~Tp(); }

	Tp* operator->() const		{ return getObject(); }
	Tp& operator*() const		{ return getRef();  }

	Tp* getObject() const		{ return (Tp*)&u;	}
	Tp& getRef() const			{ return (Tp&)u;	}

private:
	void assignBy(const Tp& rhs)
	{
		Tp* p = (Tp*)alloc();
		*p = rhs;
	}

private:
	union
	{
		char	c;
		short	s;
		int		i;
		long	l;
		long long ll;
		float	f;
		double	d;
		void*	p;

		char	buffer[sizeof(Tp)];
	}u;
};

template<typename Tp>
struct DefaultPlacement : Placement<Tp>
{
	Tp* init()
	{
		return new(Placement<Tp>::alloc()) Tp();
	}
};