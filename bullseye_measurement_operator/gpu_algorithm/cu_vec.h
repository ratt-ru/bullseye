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

template <typename T>
struct vec4 {
  T _x; T _y; T _z; T _w;
  __device__ __host__ vec4(T x = 0, T y = 0, T z = 0, T w = 0):_x(x),_y(y),_z(z),_w(w) {}
  __device__ __host__ static vec4<T> zero(){
    return vec4<T>();
  }
  __device__ __host__ vec4<T> & operator= (const vec4<T> & rhs){
    _x = rhs._x;
    _y = rhs._y;
    _z = rhs._z;
    _w = rhs._w;
    return *this;
  }
  __device__ __host__ vec4<T> operator*(const T scalar){
    return vec4<T>(_x * scalar, _y * scalar, _z * scalar, _w * scalar);
  }
  __device__ __host__ vec4<T> operator*(const vec1<T> scalar){
    return vec4<T>(_x * scalar._x, _y * scalar._x, _z * scalar._x, _w * scalar._x);
  }
  __device__ __host__ vec4<T> operator*(const vec4<T> element_wise_by_vector){
    return vec4<T>(_x * element_wise_by_vector._x, _y * element_wise_by_vector._y,
		   _z * element_wise_by_vector._z, _w * element_wise_by_vector._w);
  }
  __device__ __host__ vec4<T>& operator+=(const vec4<T> & rhs){
    _x += rhs._x;
    _y += rhs._y;
    _z += rhs._z;
    _w += rhs._w;
    return *this;
  }
  __device__ __host__ vec4<T> operator||(const T scalar){
    return vec4<T>(_x || scalar, _y || scalar, _z || scalar, _w || scalar);
  }
  __device__ __host__ vec4<T> operator!(){
    return vec4<T>(!_x,!_y,!_z,!_w);
  }
  __device__ __host__ vec4<T> operator&&(const T scalar){
    return vec4<T>(_x && scalar, _y && scalar,_z && scalar, _w && scalar);
  }
};
template <typename from_type,typename to_type>
  __device__ __host__ static vec4<to_type> vector_promotion(const vec4<from_type> & arg){
    return vec4<to_type>((to_type)(arg._x),(to_type)(arg._y),(to_type)(arg._z),(to_type)(arg._w));
}