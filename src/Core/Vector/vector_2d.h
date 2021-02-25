#pragma once
#include <iostream>
#include <glm/vec2.hpp>
#include "vector_base.h"

namespace dyno {

	template <typename T, int Dim> class SquareMatrix;

	template <typename T>
	class Vector<T, 2>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y);
		DYN_FUNC Vector(const Vector<T, 2>&);
		DYN_FUNC ~Vector();

		DYN_FUNC static int dims() { return 2; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 2> operator+ (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator+= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator- (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator-= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator* (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator*= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator/ (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator/= (const Vector<T, 2> &);

		DYN_FUNC Vector<T, 2>& operator= (const Vector<T, 2> &);

		DYN_FUNC bool operator== (const Vector<T, 2> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 2> &) const;

		DYN_FUNC const Vector<T, 2> operator* (T) const;
		DYN_FUNC const Vector<T, 2> operator- (T) const;
		DYN_FUNC const Vector<T, 2> operator+ (T) const;
		DYN_FUNC const Vector<T, 2> operator/ (T) const;

		DYN_FUNC Vector<T, 2>& operator+= (T);
		DYN_FUNC Vector<T, 2>& operator-= (T);
		DYN_FUNC Vector<T, 2>& operator*= (T);
		DYN_FUNC Vector<T, 2>& operator/= (T);

		DYN_FUNC const Vector<T, 2> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 2>& normalize();
		DYN_FUNC T cross(const Vector<T, 2> &)const;
		DYN_FUNC T dot(const Vector<T, 2>&) const;
		DYN_FUNC Vector<T, 2> minimum(const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2> maximum(const Vector<T, 2> &) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

	public:
		glm::tvec2<T> data_;
	};

	template class Vector<float, 2>;
	template class Vector<double, 2>;

	typedef Vector<float, 2> Vector2f;
	typedef Vector<double, 2> Vector2d;

} //end of namespace dyno

#include "vector_2d.inl"
