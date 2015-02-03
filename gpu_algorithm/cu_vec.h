#pragma once
template <typename T>
struct vec1 {
  T _x;
  __device__ __host__ vec1(T x = 0):_x(x) {}
  __device__ __host__ static vec1<T> zero(){
    return vec1<T>();
  }
  __device__ __host__ vec1<T> & operator= (const vec1<T> & rhs){
    _x = rhs._x;
    return *this;
  }
  __device__ __host__ vec1<T> operator*(const T scalar){
    return vec1<T>(_x * scalar);
  }
  __device__ __host__ vec1<T> operator*(const vec1<T> scalar){
    return vec1<T>(_x * scalar._x);
  }
  __device__ __host__ vec1<T>& operator+=(const vec1<T> & rhs){
    _x += rhs._x;
    return *this;
  }
  __device__ __host__ vec1<T> operator||(const T scalar){
    return vec1<T>(_x || scalar);
  }
  __device__ __host__ vec1<T> operator!(){
    return vec1<T>(!_x);
  }
  __device__ __host__ vec1<T> operator&&(const T scalar){
    return vec1<T>(_x && scalar);
  }
};
template <typename from_type,typename to_type>
__device__ __host__ static vec1<to_type> vector_promotion(const vec1<from_type> & arg){
    return vec1<to_type>((to_type)(arg._x));
}

template <typename T>
struct vec2 {
  T _x; T _y;
  __device__ __host__ vec2(T x = 0, T y = 0):_x(x),_y(y) {}
  __device__ __host__ static vec2<T> zero(){
    return vec2<T>();
  }
  __device__ __host__ vec2<T> & operator= (const vec2<T> & rhs){
    _x = rhs._x;
    _y = rhs._y;
    return *this;
  }
  __device__ __host__ vec2<T> operator*(const T scalar){
    return vec2<T>(_x * scalar, _y * scalar);
  }
  __device__ __host__ vec2<T> operator*(const vec1<T> scalar){
    return vec2<T>(_x * scalar._x, _y * scalar._x);
  }
  __device__ __host__ vec2<T> operator*(const vec2<T> element_wise_by_vector){
    return vec2<T>(_x * element_wise_by_vector._x, _y * element_wise_by_vector._y);
  }
  __device__ __host__ vec2<T>& operator+=(const vec2<T> & rhs){
    _x += rhs._x;
    _y += rhs._y;
    return *this;
  }
  __device__ __host__ vec2<T> operator||(const T scalar){
    return vec2<T>(_x || scalar, _y || scalar);
  }
  __device__ __host__ vec2<T> operator!(){
    return vec2<T>(!_x,!_y);
  }
  __device__ __host__ vec2<T> operator&&(const T scalar){
    return vec2<T>(_x && scalar, _y && scalar);
  }
};
template <typename from_type,typename to_type>
  __device__ __host__ static vec2<to_type> vector_promotion(const vec2<from_type> & arg){
    return vec2<to_type>((to_type)(arg._x),(to_type)(arg._y));
}

/**
 * scalar multiplication with correlated visibilities (can be up to 4 complex visibilties)
 */
template <typename T>
__device__ __host__ vec1<vec2<T> > operator*(const vec1<vec2<T> > & visibilities, const vec1<T> & scalars) {
  return vec1<vec2<T> >(vec2<T>(visibilities._x._x*scalars._x,visibilities._x._y*scalars._x));
}
template <typename T>
__device__ __host__ vec2<vec2<T> > operator*(const vec2<vec2<T> > & visibilities, const vec2<T> & scalars) {
  return vec2<vec2<T> >(vec2<T>(visibilities._x._x*scalars._x,visibilities._x._y*scalars._x),
			vec2<T>(visibilities._y._x*scalars._y,visibilities._y._y*scalars._y));
}